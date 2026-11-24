"""P0: audit RSNA DICOM, final Calculated labels, and lesion boxes.

This program does not train a model, create a split, move source data, or
modify DICOM/JSON inputs. On integrity failures it writes an explicit audit
report but never publishes a new cohort manifest as trustworthy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback
from typing import Any, Iterable

import numpy as np
import pydicom

from paths import AuditPaths, get_audit_paths


LABEL_NAMES = {
    "Lung Opacity": 1,
    "Normal": 0,
    "No Lung Opacity / Not Normal": 0,
}
MANIFEST_COLUMNS = (
    "image_id", "patient_id", "dicom_patient_id", "study_uid", "series_uid", "sop_uid",
    "dicom_path", "final_label", "target", "view_position", "rows", "columns",
    "photometric_interpretation", "samples_per_pixel", "bits_allocated",
    "bits_stored", "pixel_representation", "transfer_syntax_uid", "pixel_min",
    "pixel_max", "pixel_dtype", "num_boxes", "file_size_bytes",
)
BOX_COLUMNS = (
    "annotation_id", "image_id", "patient_id", "sop_uid", "box_number",
    "x", "y", "width", "height", "x2", "y2", "image_width", "image_height",
)
REQUIRED_UIDS = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bgrel.p0.audit")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (logging.FileHandler(log_path, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.info("========== RUN START | P0 Data Audit ==========")
    return logger


def record_issue(issues: list[dict[str, str]], kind: str, identifier: str, detail: str) -> None:
    issues.append({"kind": kind, "id": identifier, "detail": detail})


def extract_calculated_labels(annotations_json: Any) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Read only the adjudicated 'Calculated' group, not initial or team labels."""
    groups = annotations_json.get("labelGroups") if isinstance(annotations_json, dict) else None
    datasets = annotations_json.get("datasets") if isinstance(annotations_json, dict) else None
    if not isinstance(groups, list) or not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("Unexpected annotations JSON: labelGroups and exactly one dataset required")
    matching = [g for g in groups if g.get("name") == "Calculated"]
    if len(matching) != 1:
        raise ValueError(f"Expected one Calculated label group; found {len(matching)}")
    labels = matching[0].get("labels")
    if not isinstance(labels, list):
        raise ValueError("Calculated group labels must be a list")
    id_to_name: dict[str, str] = {}
    for label in labels:
        name, identifier = label.get("name"), label.get("id")
        if name in LABEL_NAMES and isinstance(identifier, str):
            if name in id_to_name.values() or identifier in id_to_name:
                raise ValueError("Duplicate final label name or ID")
            id_to_name[identifier] = name
    if set(id_to_name.values()) != set(LABEL_NAMES):
        raise ValueError(f"Unexpected Calculated labels: {sorted(id_to_name.values())}")
    annotations = datasets[0].get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Dataset.annotations must be a list")
    return id_to_name, [item for item in annotations if item.get("labelId") in id_to_name]


def index_mappings(mappings: Any, issues: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    if not isinstance(mappings, list):
        raise ValueError("Mappings JSON must be a list")
    # A DICOM study or series can legitimately contain multiple images.
    # Only image identifiers and SOP instance UIDs must be unique here.
    # RSNA mappings use a per-image UUID in subset_img_id. Do not confuse it
    # with the original patient grouping key embedded in img_id.
    seen: dict[str, set[str]] = {
        "SOPInstanceUID": set(), "img_id": set(), "subset_img_id": set(),
    }
    by_sop: dict[str, dict[str, Any]] = {}
    for position, item in enumerate(mappings):
        if not isinstance(item, dict):
            record_issue(issues, "invalid_mapping", str(position), "Expected mapping object")
            continue
        for key in (*REQUIRED_UIDS, "img_id", "subset_img_id"):
            value = item.get(key)
            if not isinstance(value, str) or not value:
                record_issue(issues, "missing_mapping_field", str(position), key)
            elif key in seen:
                if value in seen[key]:
                    record_issue(issues, "duplicate_mapping_field", value, key)
                else:
                    seen[key].add(value)
        sop = item.get("SOPInstanceUID")
        if isinstance(sop, str) and sop and sop not in by_sop:
            by_sop[sop] = item
    return by_sop


def index_annotations(
    annotations: Iterable[dict[str, Any]], names: dict[str, str],
    issues: list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids: set[str] = set()
    for item in annotations:
        ann_id = item.get("id")
        sop = item.get("SOPInstanceUID")
        if not isinstance(ann_id, str) or not ann_id or ann_id in ids:
            record_issue(issues, "duplicate_or_missing_annotation_id", str(ann_id), "Calculated annotation")
        else:
            ids.add(ann_id)
        if not isinstance(sop, str) or not sop:
            record_issue(issues, "missing_annotation_sop", str(ann_id), "No SOPInstanceUID")
            continue
        result[sop].append(item)
    for sop, entries in result.items():
        distinct = {names[e["labelId"]] for e in entries}
        if len(distinct) != 1:
            record_issue(issues, "conflicting_final_labels", sop, str(sorted(distinct)))
        if "Lung Opacity" not in distinct and len(entries) != 1:
            record_issue(issues, "repeated_negative_label", sop, f"{len(entries)} entries")
        for entry in entries:
            for key in REQUIRED_UIDS:
                if not entry.get(key):
                    record_issue(issues, "missing_annotation_uid", str(entry.get("id")), key)
    return result


def check_box(entry: dict[str, Any], dicom_rows: int, dicom_columns: int) -> dict[str, Any]:
    """Validate original 1024-coordinate boxes without rounding or resizing."""
    size_w, size_h = entry.get("width"), entry.get("height")
    coords = entry.get("data")
    if not isinstance(coords, dict) or any(k not in coords for k in ("x", "y", "width", "height")):
        raise ValueError("Missing box coordinates")
    values = [coords[k] for k in ("x", "y", "width", "height")]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError("Box coordinates must be finite numbers")
    x, y, width, height = map(float, values)
    if not isinstance(size_w, int) or not isinstance(size_h, int) or size_w <= 0 or size_h <= 0:
        raise ValueError("Invalid annotation canvas dimensions")
    if size_w != dicom_columns or size_h != dicom_rows:
        raise ValueError(f"Annotation canvas {size_w}x{size_h} differs from DICOM {dicom_columns}x{dicom_rows}")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("Nonpositive box or negative origin")
    if x + width > size_w + 1e-4 or y + height > size_h + 1e-4:
        raise ValueError("Bounding box exceeds annotation canvas")
    return {
        "x": x, "y": y, "width": width, "height": height,
        "x2": x + width, "y2": y + height,
        "image_width": size_w, "image_height": size_h,
    }


def checked_dicom(path: Path, mapping: dict[str, Any]) -> dict[str, Any]:
    """Read one image, verify identifiers, then decode pixels without preprocessing."""
    ds = pydicom.dcmread(path, force=False)
    for field in REQUIRED_UIDS:
        observed = str(getattr(ds, field, ""))
        if observed != mapping[field]:
            raise ValueError(f"{field} mismatch: DICOM={observed!r}, mappings={mapping[field]!r}")
    rows = int(ds.Rows)
    columns = int(ds.Columns)
    if rows <= 0 or columns <= 0:
        raise ValueError("Invalid DICOM image dimensions")
    pixels = ds.pixel_array
    if pixels.ndim != 2 or tuple(pixels.shape) != (rows, columns):
        raise ValueError(f"Unsupported pixel shape: {pixels.shape}; expected ({rows}, {columns})")
    if not np.isfinite(pixels).all():
        raise ValueError("Pixels contain NaN or infinity")
    return {
        "dicom_patient_id": str(getattr(ds, "PatientID", "")).strip(),
        "view_position": str(getattr(ds, "ViewPosition", "")).strip(),
        "rows": rows,
        "columns": columns,
        "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "")),
        "samples_per_pixel": str(getattr(ds, "SamplesPerPixel", "")),
        "bits_allocated": str(getattr(ds, "BitsAllocated", "")),
        "bits_stored": str(getattr(ds, "BitsStored", "")),
        "pixel_representation": str(getattr(ds, "PixelRepresentation", "")),
        "transfer_syntax_uid": str(getattr(ds.file_meta, "TransferSyntaxUID", "")),
        "pixel_min": int(np.min(pixels)) if np.issubdtype(pixels.dtype, np.integer) else float(np.min(pixels)),
        "pixel_max": int(np.max(pixels)) if np.issubdtype(pixels.dtype, np.integer) else float(np.max(pixels)),
        "pixel_dtype": str(pixels.dtype),
        "file_size_bytes": path.stat().st_size,
    }


def write_csv_atomic(destination: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.stem}_", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json_atomic(destination: Path, content: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.stem}_", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(content, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def audit(paths: AuditPaths, logger: logging.Logger, report_path: Path) -> dict[str, Any]:
    if (paths.processed / "split.sha256").exists():
        raise RuntimeError("P0 rebuild blocked: frozen split seal exists. Use an isolated reconstruction workspace.")
    issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    inputs = {
        "mappings": {"path": str(paths.mappings), "sha256": sha256(paths.mappings)},
        "annotations": {"path": str(paths.annotations), "sha256": sha256(paths.annotations)},
        "dicom_root": str(paths.dicom_root),
    }
    logger.info("Input JSON SHA-256 recorded; no source files will be modified")
    mappings = load_json(paths.mappings)
    annotations_json = load_json(paths.annotations)
    names, calculated = extract_calculated_labels(annotations_json)
    by_sop = index_mappings(mappings, issues)
    labeled = index_annotations(calculated, names, issues)
    logger.info("Metadata parsed: mappings=%s, calculated_annotations=%s, calculated_images=%s",
                len(mappings), len(calculated), len(labeled))

    mapped_sops = set(by_sop)
    for sop in sorted(set(labeled) - mapped_sops):
        record_issue(issues, "orphan_final_annotation", sop, "Calculated annotation has no mapping")
    expected_paths: set[Path] = set()
    for item in by_sop.values():
        if all(isinstance(item.get(k), str) and item[k] for k in REQUIRED_UIDS):
            expected_paths.add(paths.dicom_root / item["StudyInstanceUID"] / item["SeriesInstanceUID"] / (item["SOPInstanceUID"] + ".dcm"))
    actual_paths = set(paths.dicom_root.rglob("*.dcm"))
    unexpected = actual_paths - expected_paths
    logger.info("DICOM inventory: actual=%s, mapped_paths=%s, unexpected=%s",
                len(actual_paths), len(expected_paths), len(unexpected))
    for path in sorted(unexpected):
        record_issue(issues, "unmapped_dicom_file", str(path.relative_to(paths.dicom_root)), "No mapping at this path")

    manifest: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    final_labels = Counter()
    views = Counter()
    photometric = Counter()
    dimensions = Counter()
    transfers = Counter()
    matched_pseudonyms = 0
    mismatched_pseudonyms = 0
    missing_dicom_patient_ids = 0
    missing_views = 0
    patient_ids_dicom: dict[str, set[str]] = defaultdict(set)
    patient_ids_by_dicom: dict[str, set[str]] = defaultdict(set)
    used_annotations: set[str] = set()
    missing_file_count = 0
    dicom_error_count = 0

    for index, sop in enumerate(sorted(by_sop), start=1):
        mapping = by_sop[sop]
        image_id = mapping["img_id"]
        patient_id = image_id.split("_", 1)[0]
        if not image_id.endswith(".png") or "_" not in image_id or not patient_id:
            record_issue(issues, "invalid_image_id", sop, str(image_id))
        path = paths.dicom_root / mapping["StudyInstanceUID"] / mapping["SeriesInstanceUID"] / (sop + ".dcm")
        entries = labeled.get(sop, [])
        if not entries:
            exclusions.append({"sop_uid": sop, "image_id": image_id, "reason": "no_calculated_label"})
        if not path.is_file():
            missing_file_count += 1
            record_issue(issues, "missing_dicom_file", sop, str(path))
            continue
        try:
            info = checked_dicom(path, mapping)
        except Exception as error:
            dicom_error_count += 1
            record_issue(issues, "invalid_dicom", sop, f"{type(error).__name__}: {error}")
            continue
        dicom_patient = info["dicom_patient_id"]
        expected_pseudonym = mapping["subset_img_id"]
        if not dicom_patient:
            missing_dicom_patient_ids += 1
            record_issue(issues, "missing_dicom_patient_id", sop, "DICOM PatientID absent")
        else:
            # The DICOM PatientID is an image-level pseudonym in this dataset,
            # not the patient identifier used for the train/val/test split.
            patient_ids_dicom[patient_id].add(dicom_patient)
            patient_ids_by_dicom[dicom_patient].add(patient_id)
            if dicom_patient != expected_pseudonym:
                mismatched_pseudonyms += 1
                record_issue(
                    issues, "dicom_patient_id_mismatch", sop,
                    f"DICOM PatientID={dicom_patient!r}; "
                    f"mappings.subset_img_id={expected_pseudonym!r}",
                )
            else:
                matched_pseudonyms += 1
        if not info["view_position"]:
            missing_views += 1
            record_issue(warnings, "missing_view_position", sop, "DICOM ViewPosition absent")
        if info["samples_per_pixel"] != "1":
            record_issue(issues, "unsupported_samples_per_pixel", sop, info["samples_per_pixel"])
        if info["photometric_interpretation"] not in ("MONOCHROME1", "MONOCHROME2"):
            record_issue(issues, "unsupported_photometric", sop, info["photometric_interpretation"])
        if info["pixel_min"] == info["pixel_max"]:
            record_issue(warnings, "constant_pixel_image", sop, "All pixel values identical")
        views[info["view_position"] or "MISSING"] += 1
        photometric[info["photometric_interpretation"] or "MISSING"] += 1
        dimensions[f'{info["rows"]}x{info["columns"]}'] += 1
        transfers[info["transfer_syntax_uid"] or "MISSING"] += 1
        if not entries:
            continue
        final_label = names[entries[0]["labelId"]]
        if len({names[entry["labelId"]] for entry in entries}) != 1:
            continue
        record = {
            "image_id": image_id, "patient_id": patient_id, "study_uid": mapping["StudyInstanceUID"],
            "series_uid": mapping["SeriesInstanceUID"], "sop_uid": sop,
            "dicom_path": str(path.relative_to(paths.root).as_posix()) if path.is_relative_to(paths.root) else str(path),
            "final_label": final_label, "target": LABEL_NAMES[final_label], "num_boxes": 0,
            **info,
        }
        for entry in entries:
            if any(entry.get(field) != mapping[field] for field in REQUIRED_UIDS):
                record_issue(issues, "annotation_uid_mismatch", str(entry.get("id")), sop)
                continue
            used_annotations.add(entry["id"])
            if final_label != "Lung Opacity":
                if entry.get("data") is not None:
                    record_issue(issues, "negative_has_box", str(entry.get("id")), sop)
                continue
            try:
                box = check_box(entry, info["rows"], info["columns"])
            except Exception as error:
                record_issue(issues, "invalid_bounding_box", str(entry.get("id")), str(error))
                continue
            boxes.append({
                "annotation_id": entry["id"], "image_id": image_id, "patient_id": patient_id,
                "sop_uid": sop, "box_number": entry.get("annotationNumber") or "", **box,
            })
            record["num_boxes"] += 1
        if final_label == "Lung Opacity" and record["num_boxes"] == 0:
            record_issue(issues, "positive_without_valid_box", sop, "No usable Calculated lesion boxes")
        final_labels[final_label] += 1
        manifest.append(record)
        if index % 500 == 0:
            logger.info("DICOM verified: %s/%s, errors=%s, missing=%s", index, len(by_sop), dicom_error_count, missing_file_count)

    # Multiple image pseudonyms under one original patient are expected.
    # Report this relationship rather than incorrectly treating it as a leak.
    multiple_image_pseudonym_groups = sum(
        len(ids) > 1 for ids in patient_ids_dicom.values()
    )
    # Conversely, an image pseudonym attached to multiple original patients
    # is genuinely ambiguous for a patient-disjoint split and remains fatal.
    for dicom_id, patient_ids in patient_ids_by_dicom.items():
        if len(patient_ids) > 1:
            record_issue(
                issues, "dicom_id_maps_to_multiple_patient_ids",
                dicom_id, str(sorted(patient_ids)),
            )
    if len(used_annotations) != len(calculated) and not issues:
        record_issue(issues, "unaccounted_annotations", "cohort", f"Processed {len(used_annotations)}/{len(calculated)}")

    counters = Counter(item["kind"] for item in issues)
    warning_counts = Counter(item["kind"] for item in warnings)
    report = {
        "phase": "P0", "status": "PASS" if not issues else "FAIL", "timestamp_utc": now_utc(),
        "inputs": inputs,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pydicom": pydicom.__version__, "platform": platform.platform()},
        "counts": {
            "mapping_rows": len(mappings), "unique_mapping_sop": len(by_sop),
            "calculated_annotation_rows": len(calculated), "calculated_images": len(labeled),
            "unlabeled_exclusions": len(exclusions), "dicom_files_found": len(actual_paths),
            "dicom_files_without_expected_path": len(unexpected),
            "dicom_files_missing": missing_file_count, "dicom_decode_or_uid_errors": dicom_error_count,
            "included_image_rows": len(manifest), "valid_box_rows": len(boxes),
            "distinct_derived_patient_ids": len({r["patient_id"] for r in manifest}),
            "missing_view_position": missing_views,
            "dicom_patient_id_matches_subset_img_id": matched_pseudonyms,
            "dicom_patient_id_mismatches_subset_img_id": mismatched_pseudonyms,
            "missing_dicom_patient_id": missing_dicom_patient_ids,
            "distinct_dicom_patient_ids": len(patient_ids_by_dicom),
            "derived_patients_with_multiple_image_pseudonyms": multiple_image_pseudonym_groups,
        },
        "final_label_distribution": dict(sorted(final_labels.items())),
        "view_distribution": dict(sorted(views.items())),
        "photometric_distribution": dict(sorted(photometric.items())),
        "dimensions_distribution": dict(sorted(dimensions.items())),
        "transfer_syntax_distribution": dict(sorted(transfers.items())),
        "excluded_images": exclusions,
        "error_counts": dict(sorted(counters.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "issues": issues,
        "warnings": warnings,
        "outputs": {},
        "notes": [
            "Only Calculated annotations define final labels and lesion boxes.",
            "Pixel decoding validates readability, not image orientation or clinical correctness.",
            "Patient grouping uses the original image ID prefix before '_', as specified by the proposal.",
            "DICOM PatientID is checked against mappings.subset_img_id (an image pseudonym), not the original patient prefix.",
            "Multiple DICOM image pseudonyms for one derived patient are expected and not a split leak.",
            "No split, segmentation, model training, or preprocessing occurs in P0.",
        ],
    }
    if not issues:
        manifest_path = paths.processed / "master_manifest.csv"
        box_path = paths.processed / "boxes.csv"
        write_csv_atomic(manifest_path, MANIFEST_COLUMNS, manifest)
        write_csv_atomic(box_path, BOX_COLUMNS, boxes)
        report["outputs"] = {
            "master_manifest.csv": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "boxes.csv": {"path": str(box_path), "sha256": sha256(box_path)},
        }
    else:
        logger.error("P0 integrity check failed with %s issues. CSV cohort artifacts were not published.", len(issues))
    write_json_atomic(report_path, report)
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 audit: DICOM, final Calculated labels, and lesion boxes")
    parser.add_argument("--dicom-root", type=Path, help="Override DICOM source folder")
    parser.add_argument("--mappings", type=Path, help="Override source mappings JSON")
    parser.add_argument("--annotations", type=Path, help="Override source annotations JSON")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing previous P0 QA and CSV outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        paths = get_audit_paths(args.dicom_root, args.mappings, args.annotations)
    except Exception as error:
        print(f"P0 BLOCKED: {error}", file=sys.stderr)
        return 2
    if (paths.processed / "split.sha256").exists():
        print("P0 BLOCKED: frozen split seal exists; use an isolated reconstruction workspace", file=sys.stderr)
        return 2
    logger = setup_logger(paths.logs / "audit_data.log")
    report_path = paths.qa / "audit_report.json"
    try:
        target_files = (report_path, paths.processed / "master_manifest.csv", paths.processed / "boxes.csv")
        already_exist = [str(p) for p in target_files if p.exists()]
        if already_exist and not args.overwrite:
            logger.error("Outputs already exist; --overwrite required: %s", already_exist)
            return 2
        for path in (paths.dicom_root, paths.mappings, paths.annotations):
            if not path.exists():
                raise FileNotFoundError(str(path))
        logger.info("Project root: %s", paths.root)
        logger.info("DICOM root: %s", paths.dicom_root)
        logger.info("Mappings: %s", paths.mappings)
        logger.info("Annotations: %s", paths.annotations)
        logger.info("Source checksum: audit_data.py=%s", sha256(Path(__file__)))
        logger.info("Source checksum: paths.py=%s", sha256(Path(__file__).with_name("paths.py")))
        report = audit(paths, logger, report_path)
        logger.info("P0 Status: %s", report["status"])
        return 0 if report["status"] == "PASS" else 1
    except Exception as error:
        logger.exception("Unexpected failure; P0 has not passed")
        try:
            write_json_atomic(report_path, {
                "phase": "P0", "status": "FAIL", "timestamp_utc": now_utc(),
                "error_type": type(error).__name__, "error": str(error),
                "notes": ["Audit terminated unexpectedly; no PASS is authorized.",
                          "Inspect results/logs/audit_data.log for the complete traceback."],
            })
        except Exception:
            logger.exception("Could not write failure report")
        return 2
    finally:
        logger.info("=========== RUN END | P0 Data Audit ===========")
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
