"""P1: Create a reproducible patient-disjoint split from audited RSNA artifacts.

The source cohort is immutable. Preview first, inspect its QA report, and freeze
only after approval. The frozen split is never silently regenerated/overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
QA_DIR = PROJECT_ROOT / "results" / "qa"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
MANIFEST_PATH = PROCESSED_DIR / "master_manifest.csv"
BOXES_PATH = PROCESSED_DIR / "boxes.csv"
P0_REPORT_PATH = QA_DIR / "audit_report.json"
PREVIEW_PATH = QA_DIR / "split_preview.json"
AUDIT_PATH = QA_DIR / "split_audit.json"
SPLIT_PATH = PROCESSED_DIR / "split.csv"
SEAL_PATH = PROCESSED_DIR / "split.sha256"
LOG_PATH = LOG_DIR / "split_patients.log"

# The seed is a new P1 implementation choice, NOT a historic V2 assignment.
DEFAULT_SEED = 42
ALGORITHM_VERSION = "patient-stratified-greedy-refine-v1"
SPLITS = ("train", "val", "test")
RATIOS = (0.8, 0.1, 0.1)
LABELS = ("Lung Opacity", "Normal", "No Lung Opacity / Not Normal")
VIEWS = ("AP", "PA")
CELLS = tuple((label, view) for label in LABELS for view in VIEWS)
CSV_FIELDS = ("image_id", "patient_id", "split")
INPUT_COLUMNS = ("image_id", "patient_id", "sop_uid", "final_label", "target", "view_position", "num_boxes")
BOX_COLUMNS = ("image_id", "patient_id", "sop_uid")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bgrel.p1.split")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_logging(logger)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logging(logger: logging.Logger) -> None:
    """Release all phase-owned handlers before Windows removes temporary files."""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        try:
            handler.flush()
        finally:
            handler.close()


def parse_int(text: str, name: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an integer: {text!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must not be negative: {text!r}")
    return value


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"CSV has empty or duplicated headers: {path}")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
        result = []
        for index, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Invalid CSV field count at {path.name}:{index}")
            result.append(row)
    if not result:
        raise ValueError(f"CSV contains no data: {path}")
    return result


def validate_p0() -> tuple[list[dict[str, str]], dict[str, int], dict[str, str]]:
    for path in (MANIFEST_PATH, BOXES_PATH, P0_REPORT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required audited P0 artifact not found: {path}")
    qa = read_json(P0_REPORT_PATH)
    if qa.get("status") != "PASS":
        raise ValueError("P0 audit_report.json status is not PASS")
    if qa.get("issues") or qa.get("error_counts") or qa.get("warnings") or qa.get("warning_counts"):
        raise ValueError("P0 QA contains issues/errors/warnings; review before creating split")
    expected_outputs = qa.get("outputs")
    if not isinstance(expected_outputs, dict):
        raise ValueError("P0 QA does not contain outputs with hashes")
    hashes: dict[str, str] = {}
    for name, path in (("master_manifest.csv", MANIFEST_PATH), ("boxes.csv", BOXES_PATH)):
        claimed = expected_outputs.get(name)
        if not isinstance(claimed, dict) or not claimed.get("sha256"):
            raise ValueError(f"P0 QA missing expected SHA-256 for {name}")
        actual = file_hash(path)
        if actual != str(claimed["sha256"]).lower():
            raise ValueError(f"P0 input was altered: {name} SHA-256 mismatch")
        hashes[name] = actual
    hashes["audit_report.json"] = file_hash(P0_REPORT_PATH)

    images = read_csv(MANIFEST_PATH, INPUT_COLUMNS)
    boxes = read_csv(BOXES_PATH, BOX_COLUMNS)
    ids: dict[str, dict[str, str]] = {}
    sops: set[str] = set()
    box_expected: dict[str, int] = {}
    counts = Counter()
    for row in images:
        image_id, patient_id = row["image_id"], row["patient_id"]
        if not image_id or not patient_id or not row["sop_uid"]:
            raise ValueError("Manifest contains an empty image/patient/SOP identifier")
        if image_id in ids or row["sop_uid"] in sops:
            raise ValueError(f"Duplicated image_id or SOP UID: {image_id}")
        if image_id.split("_", 1)[0] != patient_id or "_" not in image_id:
            raise ValueError(f"Invalid patient grouping key at {image_id}")
        if row["final_label"] not in LABELS or row["view_position"] not in VIEWS:
            raise ValueError(f"Unknown final label/view at {image_id}")
        if row["target"] != ("1" if row["final_label"] == "Lung Opacity" else "0"):
            raise ValueError(f"Binary target mismatch at {image_id}")
        nboxes = parse_int(row["num_boxes"], f"num_boxes at {image_id}")
        if (row["target"] == "1" and nboxes == 0) or (row["target"] == "0" and nboxes != 0):
            raise ValueError(f"Unexpected lesion box count at {image_id}")
        ids[image_id] = row
        sops.add(row["sop_uid"])
        box_expected[image_id] = nboxes
        counts[(row["final_label"], row["view_position"])] += 1
    box_observed = Counter()
    for row in boxes:
        image_id = row["image_id"]
        if image_id not in ids:
            raise ValueError(f"Orphan box references unknown image: {image_id}")
        image = ids[image_id]
        if row["patient_id"] != image["patient_id"] or row["sop_uid"] != image["sop_uid"]:
            raise ValueError(f"Box mapping mismatch at {image_id}")
        box_observed[image_id] += 1
    for image_id, expected in box_expected.items():
        if box_observed[image_id] != expected:
            raise ValueError(f"Box count mismatch at {image_id}: expected {expected}, got {box_observed[image_id]}")
    qa_counts = qa.get("counts", {})
    for key, actual in (("included_image_rows", len(images)), ("valid_box_rows", len(boxes)),
                        ("distinct_derived_patient_ids", len({row["patient_id"] for row in images}))):
        if qa_counts.get(key) != actual:
            raise ValueError(f"P0 QA count mismatch {key}: reported={qa_counts.get(key)} actual={actual}")
    return images, {f"{l}|{v}": counts[(l, v)] for l, v in CELLS}, hashes


def patient_vectors(images: list[dict[str, str]]) -> dict[str, tuple[int, ...]]:
    """Vector: image count, patient count, then six label/view counts."""
    result: dict[str, list[int]] = {}
    cell_index = {pair: index for index, pair in enumerate(CELLS)}
    for row in images:
        patient = row["patient_id"]
        vector = result.setdefault(patient, [0] * (2 + len(CELLS)))
        vector[0] += 1
        vector[1] = 1
        vector[2 + cell_index[(row["final_label"], row["view_position"])]] += 1
    return {patient: tuple(vector) for patient, vector in result.items()}


def seeded_rank(patient: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}|{patient}".encode("utf-8")).digest()


def balance_score(counts: list[list[int]], totals: tuple[int, ...]) -> float:
    """Squared normalized deviation for images, patients, and six strata."""
    weights = (12.0, 4.0) + (1.0,) * len(CELLS)
    score = 0.0
    for split_idx, ratio in enumerate(RATIOS):
        for k, total in enumerate(totals):
            deviation = (counts[split_idx][k] - ratio * total) / max(1, total)
            score += weights[k] * deviation * deviation
    return score


def choose_split(vectors: dict[str, tuple[int, ...]], seed: int) -> tuple[dict[str, str], dict[str, Any]]:
    """Deterministic greedy grouping followed by strictly improving patient moves."""
    if not vectors:
        raise ValueError("No patient records")
    dimensions = 2 + len(CELLS)
    total = tuple(sum(vector[k] for vector in vectors.values()) for k in range(dimensions))
    # A cryptographic stable rank avoids reliance on Python's hash randomization.
    order = sorted(vectors, key=lambda patient: seeded_rank(patient, seed))
    occupancy = [[0] * dimensions for _ in SPLITS]
    assignments: dict[str, int] = {}
    for patient in order:
        vector = vectors[patient]
        options = []
        for j in range(len(SPLITS)):
            for k in range(dimensions):
                occupancy[j][k] += vector[k]
            score = balance_score(occupancy, total)
            for k in range(dimensions):
                occupancy[j][k] -= vector[k]
            options.append((score, j))
        j = min(options)[1]
        assignments[patient] = j
        for k in range(dimensions):
            occupancy[j][k] += vector[k]

    greedy_score = balance_score(occupancy, total)
    changed_by_pass: list[int] = []
    # Fixed ordered passes, no stochastic or metric-dependent tuning.
    for _ in range(5):
        changed = 0
        for patient in order:
            current = assignments[patient]
            vector = vectors[patient]
            score_before = balance_score(occupancy, total)
            best, best_score = current, score_before
            for j in range(len(SPLITS)):
                if j == current:
                    continue
                for k in range(dimensions):
                    occupancy[current][k] -= vector[k]
                    occupancy[j][k] += vector[k]
                candidate = balance_score(occupancy, total)
                for k in range(dimensions):
                    occupancy[current][k] += vector[k]
                    occupancy[j][k] -= vector[k]
                if candidate < best_score - 1e-15:
                    best, best_score = j, candidate
            if best != current:
                for k in range(dimensions):
                    occupancy[current][k] -= vector[k]
                    occupancy[best][k] += vector[k]
                assignments[patient] = best
                changed += 1
        changed_by_pass.append(changed)
        if changed == 0:
            break
    final_score = balance_score(occupancy, total)
    labels = {patient: SPLITS[split_idx] for patient, split_idx in assignments.items()}
    diagnostics = {"algorithm": ALGORITHM_VERSION, "seed": seed, "objective":
                   "normalized-weighted-squared-image-patient-label-view-deviation",
                   "greedy_objective": greedy_score, "final_objective": final_score,
                   "refinement_moves_per_pass": changed_by_pass, "weights": {"images": 12, "patients": 4, "each_label_view_cell": 1},
                   "max_refinement_passes": 5}
    return labels, diagnostics


def split_bytes(images: list[dict[str, str]], assignment: dict[str, str]) -> bytes:
    """Canonical UTF-8, LF line endings, independent of Windows locale."""
    lines = [",".join(CSV_FIELDS)]
    for row in sorted(images, key=lambda r: r["image_id"]):
        lines.append(",".join((row["image_id"], row["patient_id"], assignment[row["patient_id"]])))
    return ("\n".join(lines) + "\n").encode("utf-8")


def validate_split(images: list[dict[str, str]], assignments: dict[str, str], data: bytes) -> dict[str, Any]:
    patient_sets = {name: set() for name in SPLITS}
    image_counts = Counter()
    patient_counts = Counter()
    positive_counts = Counter()
    boxes_counts = Counter()
    strata = {name: Counter() for name in SPLITS}
    patient_label_sets: dict[str, set[str]] = defaultdict(set)
    for row in images:
        patient = row["patient_id"]
        name = assignments.get(patient)
        if name not in SPLITS:
            raise ValueError(f"No valid split assignment for patient {patient}")
        patient_sets[name].add(patient)
        image_counts[name] += 1
        positive_counts[name] += int(row["target"])
        boxes_counts[name] += int(row["num_boxes"])
        strata[name][(row["final_label"], row["view_position"])] += 1
        patient_label_sets[patient].add(row["final_label"])
    if any(patient_sets[a] & patient_sets[b] for a in SPLITS for b in SPLITS if a != b):
        raise ValueError("Patient overlap across train/val/test")
    if set(assignments) != set().union(*patient_sets.values()):
        raise ValueError("Patients missing or duplicated in assignments")
    if any(image_counts[name] == 0 for name in SPLITS):
        raise ValueError("At least one split is empty")
    for name in SPLITS:
        if any(strata[name][cell] == 0 for cell in CELLS):
            raise ValueError(f"Split {name} is missing a label/view subgroup")
    observed = list(csv.DictReader(data.decode("utf-8").splitlines()))
    if len(observed) != len(images) or set(row["image_id"] for row in observed) != set(row["image_id"] for row in images):
        raise ValueError("Split rows do not cover exactly the P0 image cohort")
    for row in observed:
        if assignments[row["patient_id"]] != row["split"]:
            raise ValueError(f"Inconsistent split record: {row['image_id']}")

    total_images = len(images)
    stats = {}
    for name, requested in zip(SPLITS, RATIOS):
        observed_ratio = image_counts[name] / total_images
        stats[name] = {
            "images": image_counts[name], "image_fraction": observed_ratio,
            "target_fraction": requested, "deviation_percentage_points": round(100.0 * (observed_ratio - requested), 5),
            "patients": len(patient_sets[name]), "positive_images": positive_counts[name],
            "negative_images": image_counts[name] - positive_counts[name],
            "lesion_boxes": boxes_counts[name],
            "label_view_counts": {f"{label}|{view}": strata[name][(label, view)] for label, view in CELLS},
            "label_counts": {label: sum(strata[name][(label, view)] for view in VIEWS) for label in LABELS},
            "view_counts": {view: sum(strata[name][(label, view)] for label in LABELS) for view in VIEWS},
        }
    return {
        "total_images": total_images,
        "total_patients": len(assignments),
        "mixed_final_label_patients": sum(len(s) > 1 for s in patient_label_sets.values()),
        "patient_overlap_count": 0,
        "patient_disjoint": True,
        "all_images_assigned_once": True,
        "all_six_label_view_cells_present_per_split": True,
        "partitions": stats,
    }


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".bgrel_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build_report(seed: int, hashes: dict[str, str], split_sha: str, stats: dict[str, Any],
                 diagnostics: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "phase": "P1 Patient-Disjoint Split", "status": status, "timestamp_utc": utc_now(),
        "proposal_split_target": {name: ratio for name, ratio in zip(SPLITS, RATIOS)},
        "seed": seed, "algorithm_version": ALGORITHM_VERSION,
        "input_hashes_sha256": hashes,
        "source_code_sha256": file_hash(Path(__file__)),
        "proposed_split_sha256": split_sha,
        "split_columns": list(CSV_FIELDS),
        "cohort": stats,
        "optimization": diagnostics,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "notes": [
            "Grouping key comes from audited P0 patient_id, not DICOM PatientID.",
            "Algorithm chooses metadata-balanced assignments; it never reads model predictions.",
            "P1 output is not a pristine historical holdout because V2 previously developed on RSNA.",
            "The seed is a new, recorded implementation choice, not a V2 recovery.",
            "Preview is not a frozen split; explicit freeze is required before training.",
        ],
    }


def ensure_no_frozen_outputs() -> None:
    existing = [str(path) for path in (SPLIT_PATH, SEAL_PATH, AUDIT_PATH) if path.exists()]
    if existing:
        raise FileExistsError("Frozen P1 output already exists; do not overwrite: " + ", ".join(existing))


def run(seed: int, preview: bool, freeze: bool, logger: logging.Logger) -> int:
    if preview == freeze:
        raise ValueError("Exactly one of preview or freeze mode must be selected")
    logger.info("P1 mode=%s; seed=%d; algorithm=%s", "preview" if preview else "freeze", seed, ALGORITHM_VERSION)
    logger.info("P0 manifest: %s", MANIFEST_PATH)
    logger.info("P0 boxes: %s", BOXES_PATH)
    ensure_no_frozen_outputs()
    images, cell_counts, hashes = validate_p0()
    logger.info("P0 hashes verified; images=%d; label/view cohort=%s", len(images), cell_counts)
    vectors = patient_vectors(images)
    assignments, diagnostics = choose_split(vectors, seed)
    result_bytes = split_bytes(images, assignments)
    split_sha = bytes_hash(result_bytes)
    stats = validate_split(images, assignments, result_bytes)
    for name in SPLITS:
        summary = stats["partitions"][name]
        logger.info("%s images=%d patients=%d positive=%d boxes=%d deviation_pp=%+.4f", name,
                    summary["images"], summary["patients"], summary["positive_images"],
                    summary["lesion_boxes"], summary["deviation_percentage_points"])
    logger.info("Proposed split SHA-256: %s", split_sha)
    if preview:
        report = build_report(seed, hashes, split_sha, stats, diagnostics, "PREVIEW_ONLY")
        if PREVIEW_PATH.exists():
            prior = read_json(PREVIEW_PATH)
            if (prior.get("status") != "PREVIEW_ONLY" or prior.get("seed") != seed
                    or prior.get("algorithm_version") != ALGORITHM_VERSION
                    or prior.get("input_hashes_sha256") != hashes
                    or prior.get("source_code_sha256") != file_hash(Path(__file__))
                    or prior.get("proposed_split_sha256") != split_sha):
                raise FileExistsError("A different P1 preview exists; investigate instead of overwriting it")
            logger.info("Existing preview matches; preserving original report: %s", PREVIEW_PATH)
        else:
            atomic_write(PREVIEW_PATH, json_bytes(report))
            logger.info("Preview QA written: %s; frozen assignments NOT created", PREVIEW_PATH)
        return 0

    if not PREVIEW_PATH.is_file():
        raise FileNotFoundError("P1 preview report is missing. Run --preview and review first")
    preview_report = read_json(PREVIEW_PATH)
    if (preview_report.get("status") != "PREVIEW_ONLY" or preview_report.get("seed") != seed
            or preview_report.get("algorithm_version") != ALGORITHM_VERSION
            or preview_report.get("input_hashes_sha256") != hashes
            or preview_report.get("source_code_sha256") != file_hash(Path(__file__))
            or preview_report.get("proposed_split_sha256") != split_sha):
        raise ValueError("Current split differs from reviewed preview; refuse to freeze")
    report = build_report(seed, hashes, split_sha, stats, diagnostics, "PASS")
    # A lock file prevents two cooperative invocations from publishing together.
    lock = PROCESSED_DIR / ".split_patients.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError as exc:
        raise FileExistsError("Another split operation/lock exists. Investigate before retrying") from exc
    try:
        ensure_no_frozen_outputs()
        # On a filesystem crash, a partial set of outputs blocks rerun and is not treated as PASS.
        atomic_write(SPLIT_PATH, result_bytes)
        atomic_write(SEAL_PATH, (split_sha + "  split.csv\n").encode("ascii"))
        atomic_write(AUDIT_PATH, json_bytes(report))
        logger.info("FROZEN split.csv, split.sha256 and split_audit.json written; no auto-overwrite")
    finally:
        lock.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preview", action="store_true", help="Write QA preview without freezing split")
    modes.add_argument("--freeze", action="store_true", help="Freeze only if preview hash still matches")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="New, deterministic P1 seed (default: 42)")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must not be negative")
    logger = setup_logging()
    try:
        logger.info("========== RUN START | P1 Patient Split ==========")
        try:
            return run(args.seed, args.preview, args.freeze, logger)
        except Exception:
            logger.exception("P1 FAILED. No split should be regarded as frozen without QA and hash checks")
            return 1
        finally:
            logger.info("=========== RUN END | P1 Patient Split ===========")
    finally:
        close_logging(logger)


if __name__ == "__main__":
    sys.exit(main())
