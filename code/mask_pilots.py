"""P2 TRAIN pilot workflows: QA20 lung-mask pilot and 36-image policy pilot."""
from __future__ import annotations


from refactor_helpers import verify_recorded_source_sha

import argparse
import csv
import importlib.metadata
import inspect
import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mask_policy import preliminary_decision, shape_features
from generate_lung_masks import (
    CANDIDATE_THRESHOLD,
    EXPECTED_WEIGHTS_SHA256,
    LEFT_INDEX,
    RIGHT_INDEX,
    WEIGHTS_NAME,
    read_csv,
    resize_binary_nearest,
    sha256,
)
from mask_qa import dicom_path, validate_dicom_record, validated_qa_inputs
from paths import PROJECT_ROOT

PILOT_SIZE = 20
MASK_SCHEMA_VERSION = 1
REVIEW_STATES = {"PASS", "WARN", "FAIL"}
AUTO_STATES = {"AUTO_SCREENED", "AUTO_FLAG", "AUTO_REJECT"}
AUDIT_FIELDS = (
    "image_id", "final_label", "view_position", "visual_review", "saved_auto_status",
    "observed_auto_status", "rel_policy", "skip_reasons", "mask_file",
    "mask_sha256", "height", "width", "left_fraction", "right_fraction",
    "left_largest_component_share", "right_largest_component_share",
)


def verified_evidence(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Validate P0/P1, QA200 and the two read-only screening records."""
    selected, manifest, _, qa_selection_audit = validated_qa_inputs(root)
    qa = root / "results/qa"
    reviewed_file = qa / "qa_review_completed.csv"
    review_manifest = json.loads((qa / "qa_review_manifest.json").read_text(encoding="utf-8"))
    auto_file = qa / "qa_auto_assessment.csv"
    auto_audit = json.loads((qa / "qa_auto_audit.json").read_text(encoding="utf-8"))
    qa_manifest_file = qa / "pspnet_manifest.json"
    qa_manifest = json.loads(qa_manifest_file.read_text(encoding="utf-8"))
    if review_manifest.get("status") != "VISUAL_SCREENING_ONLY":
        raise ValueError("The visual-screening provenance has changed")
    if review_manifest.get("review_csv_sha256") != sha256(reviewed_file):
        raise ValueError("Visual review CSV checksum mismatch")
    if review_manifest.get("qa_selection_sha256") != qa_selection_audit["selection_csv_sha256"]:
        raise ValueError("Visual review belongs to a different QA selection")
    if auto_audit.get("status") != "AUTOMATED_SCREEN_COMPLETE_NOT_ANATOMICAL_VALIDATION":
        raise ValueError("Automated QA provenance status changed")
    for attr, actual in (
        ("qa_auto_csv_sha256", sha256(auto_file)),
        ("prior_qa_manifest_sha256", sha256(qa_manifest_file)),
        ("prior_qa_images_csv_sha256", sha256(qa / "qa_images.csv")),
        ("selection_sha256", qa_selection_audit["selection_csv_sha256"]),
    ):
        if auto_audit.get(attr) != actual:
            raise ValueError(f"Auto QA provenance mismatch: {attr}")
    verify_recorded_source_sha(root, auto_audit.get("script_sha256"), "auto_mask_qa.py", "mask_policy.py")
    if auto_audit.get("split_sha256") != qa_selection_audit["split_sha256"]:
        raise ValueError("Frozen split/Auto QA mismatch")
    if auto_audit.get("checkpoint_sha256") != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Checkpoint/Auto QA mismatch")
    verify_recorded_source_sha(root, qa_manifest.get("generator_sha256"), "generate_lung_masks.py")
    verify_recorded_source_sha(root, qa_manifest.get("qa_script_sha256"), "run_mask_qa.py", "mask_qa.py")

    visual = read_csv(reviewed_file)
    automated = read_csv(auto_file)
    expected_ids = [r["image_id"] for r in selected]
    if (len(visual) != 200 or len(automated) != 200 or
            {r["image_id"] for r in visual} != set(expected_ids) or
            [r["image_id"] for r in automated] != expected_ids):
        raise ValueError("Review/auto QA does not exactly cover frozen QA200")
    if len({r["image_id"] for r in visual}) != 200:
        raise ValueError("Duplicate visual-review image IDs")
    auto_by_id = {r["image_id"]: r for r in automated}
    if len(auto_by_id) != 200:
        raise ValueError("Duplicate automated QA image IDs")
    records: list[dict[str, str]] = []
    for v in visual:
        a = auto_by_id[v["image_id"]]
        original = manifest[v["image_id"]]
        if v["review_status"] not in REVIEW_STATES or a["auto_status"] not in AUTO_STATES:
            raise ValueError("Unknown QA screening status")
        for field in ("final_label", "view_position"):
            if v[field] != a[field] or v[field] != original[field]:
                raise ValueError(f"QA metadata differs for {v['image_id']}: {field}")
        if a["rel_policy"] != ("REL_CANDIDATE_ONLY" if a["auto_status"] == "AUTO_SCREENED" else "SKIP_REL"):
            raise ValueError("Automated QA stored policy inconsistent")
        records.append({**v, "saved_auto_status": a["auto_status"],
                        "saved_auto_reasons": a["auto_reasons"]})
    return records, manifest, qa_manifest, auto_audit


def choose_pilot(reviewed: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deliberately include all known FAILs, auto flags, visual WARNs and PASSs."""
    failed = sorted((r for r in reviewed if r["review_status"] == "FAIL"), key=lambda r: r["image_id"])
    flagged = sorted((r for r in reviewed if r["review_status"] == "WARN" and
                      r["saved_auto_status"] != "AUTO_SCREENED"), key=lambda r: r["image_id"])
    warned = sorted((r for r in reviewed if r["review_status"] == "WARN" and
                     r["saved_auto_status"] == "AUTO_SCREENED"), key=lambda r: r["image_id"])
    passed = sorted((r for r in reviewed if r["review_status"] == "PASS" and
                     r["saved_auto_status"] == "AUTO_SCREENED"),
                    key=lambda r: (r["final_label"], r["view_position"], r["image_id"]))
    if len(failed) != 3 or len(flagged) != 3 or len(warned) < 6 or len(passed) < 8:
        raise ValueError("Known QA sample composition changed; do not silently reselect")
    chosen = failed + flagged + warned[:6]
    # Cover distinct class-by-projection groups for the visually screened PASS group.
    used = {r["image_id"] for r in chosen}
    strata_seen: set[tuple[str, str]] = set()
    for r in passed:
        key = (r["final_label"], r["view_position"])
        if key not in strata_seen and len(chosen) < PILOT_SIZE:
            chosen.append(r)
            strata_seen.add(key)
            used.add(r["image_id"])
    for r in passed:
        if len(chosen) == PILOT_SIZE:
            break
        if r["image_id"] not in used:
            chosen.append(r)
            used.add(r["image_id"])
    if len(chosen) != PILOT_SIZE or len(used) != PILOT_SIZE:
        raise ValueError("Pilot cohort is not unique or size 20")
    return chosen


def rel_decision(review: str, saved_auto: str, newly_observed: str) -> tuple[str, list[str]]:
    """A known failure or any screening uncertainty must fail closed for REL."""
    if review not in REVIEW_STATES or saved_auto not in AUTO_STATES or newly_observed not in AUTO_STATES:
        raise ValueError("Unknown screening state")
    reasons = []
    if review != "PASS":
        reasons.append("VISUAL_" + review)
    if saved_auto != "AUTO_SCREENED":
        reasons.append("QA200_" + saved_auto)
    if newly_observed != "AUTO_SCREENED":
        reasons.append("PILOT_" + newly_observed)
    return ("SKIP_REL" if reasons else "REL_CANDIDATE_ONLY"), reasons


def save_mask(path: Path, left: np.ndarray, right: np.ndarray) -> str:
    """Save and verify a lossless, original-grid binary mask without overwriting."""
    if left.shape != right.shape or left.ndim != 2 or left.dtype != np.bool_ or right.dtype != np.bool_:
        raise ValueError("Expected two same-shape 2D boolean lung masks")
    if left.shape != (1024, 1024):
        raise ValueError("Expected unchanged 1024x1024 original RSNA frame")
    if path.exists():
        raise FileExistsError(f"Mask already exists: {path}")
    with path.open("xb") as stream:
        np.savez_compressed(stream, left=left, right=right, schema_version=np.int32(MASK_SCHEMA_VERSION))
    with np.load(path, allow_pickle=False) as payload:
        if (int(payload["schema_version"]) != MASK_SCHEMA_VERSION or
                not np.array_equal(payload["left"], left) or
                not np.array_equal(payload["right"], right)):
            raise ValueError("Mask round-trip failed")
    return sha256(path)


def validate_rerun(reference: dict[str, str], l: np.ndarray, r: np.ndarray) -> None:
    """Compare re-inference occupancy against the prior immutable QA200 CSV."""
    if abs(float(l.mean()) - float(reference["left_fraction"])) > 2e-6:
        raise ValueError("Left lung area changed from QA200")
    if abs(float(r.mean()) - float(reference["right_fraction"])) > 2e-6:
        raise ValueError("Right lung area changed from QA200")


def run_lung_pilot(root: Path, device_name: str, logger: logging.Logger) -> None:
    import pydicom
    import torch
    import torchxrayvision as xrv

    reviewed, manifest_by_id, qa_manifest, auto_audit = verified_evidence(root)
    sample = choose_pilot(reviewed)
    qa = root / "results/qa"
    masks_root = root / "data/lung_masks"
    stage = masks_root / ".pilot_stage"
    final_masks = masks_root / "pilot"
    csv_out = qa / "pilot_mask_audit.csv"
    json_out = qa / "pilot_mask_manifest.json"
    for output in (stage, final_masks, csv_out, json_out):
        if output.exists():
            raise FileExistsError(f"Existing pilot artifact: {output}. Do not overwrite")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but not available")
    weights = root / "models/pretrained" / WEIGHTS_NAME
    if sha256(weights) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Checkpoint differs from recorded download")
    if importlib.metadata.version("torchxrayvision") != "1.5.7":
        raise ValueError("TorchXRayVision version changed")
    if torch.__version__ != qa_manifest["torch_version"]:
        raise ValueError("PyTorch version changed from QA200")
    if sha256(Path(inspect.getfile(xrv.baseline_models.chestx_det.PSPNet))) != qa_manifest["wrapper_sha256"]:
        raise ValueError("PSPNet wrapper changed from QA200")
    if (CANDIDATE_THRESHOLD != 0.5 or LEFT_INDEX != 4 or RIGHT_INDEX != 5):
        raise ValueError("PSPNet preprocessing contract changed")
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(weights.parent)).to(device_name).eval()
    if (len(model.targets) != 14 or model.targets[LEFT_INDEX] != "Left Lung" or
            model.targets[RIGHT_INDEX] != "Right Lung"):
        raise ValueError("Model output targets changed")
    stage.mkdir(parents=True, exist_ok=False)
    logger.info("P2 PILOT START | count=%d device=%s checkpoint=%s split=%s",
                len(sample), device_name, EXPECTED_WEIGHTS_SHA256, auto_audit["split_sha256"])
    qa_by_id = {r["image_id"]: r for r in read_csv(qa / "qa_images.csv")}
    results = []
    for index, item in enumerate(sample, start=1):
        image_id = item["image_id"]
        record = manifest_by_id[image_id]
        ds = pydicom.dcmread(dicom_path(root, record["dicom_path"]))
        display, normalized = validate_dicom_record(root, record, ds)
        image_h, image_w = display.shape
        if (image_h, image_w) != (1024, 1024):
            raise ValueError(f"Unexpected geometry for {image_id}")
        tensor = torch.from_numpy(normalized.copy())[None, None].to(device_name)
        with torch.inference_mode():
            logits = model(tensor)
            if tuple(logits.shape) != (1, 14, 512, 512) or not bool(torch.isfinite(logits).all()):
                raise ValueError(f"Non-finite or incorrectly shaped PSPNet output: {image_id}")
            probabilities = torch.sigmoid(logits[0, [LEFT_INDEX, RIGHT_INDEX]]).cpu().numpy()
        native_left = probabilities[0] > CANDIDATE_THRESHOLD
        native_right = probabilities[1] > CANDIDATE_THRESHOLD
        left = resize_binary_nearest(native_left, (image_w, image_h))
        right = resize_binary_nearest(native_right, (image_w, image_h))
        validate_rerun(qa_by_id[image_id], left, right)
        observed_status, observed_reasons = preliminary_decision(shape_features(native_left, native_right))
        policy, reasons = rel_decision(item["review_status"], item["saved_auto_status"], observed_status)
        mask_file, mask_digest = "", ""
        if policy == "REL_CANDIDATE_ONLY":
            target = stage / (Path(image_id).stem + ".npz")
            mask_digest = save_mask(target, left, right)
            mask_file = "data/lung_masks/pilot/" + target.name
        results.append({
            "image_id": image_id,
            "final_label": item["final_label"],
            "view_position": item["view_position"],
            "visual_review": item["review_status"],
            "saved_auto_status": item["saved_auto_status"],
            "observed_auto_status": observed_status,
            "rel_policy": policy,
            "skip_reasons": "|".join(reasons + observed_reasons),
            "mask_file": mask_file,
            "mask_sha256": mask_digest,
            "height": str(image_h),
            "width": str(image_w),
            "left_fraction": f"{float(left.mean()):.6f}",
            "right_fraction": f"{float(right.mean()):.6f}",
            "left_largest_component_share": f"{shape_features(native_left, native_right)['left_largest_component_share']:.6f}",
            "right_largest_component_share": f"{shape_features(native_left, native_right)['right_largest_component_share']:.6f}",
        })
        logger.info("P2 PILOT | %d/%d | %s | %s | %s", index, len(sample), image_id, policy, "|".join(reasons))
        print(f"P2 pilot: {index}/{len(sample)} | {policy}")
        del tensor, logits, probabilities

    # Publish masks only after all 20 records pass re-inference and serialization checks.
    os.rename(stage, final_masks)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    manifest = {
        "phase": "P2",
        "scope": "TRAIN_QA200_20_IMAGE_PILOT_ONLY",
        "status": "PILOT_COMPUTE_COMPLETE_REVIEW_REQUIRED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": PILOT_SIZE,
        "counts": dict(Counter(r["rel_policy"] for r in results)),
        "selection_image_ids": [r["image_id"] for r in sample],
        "split_sha256": auto_audit["split_sha256"],
        "qa_auto_csv_sha256": auto_audit["qa_auto_csv_sha256"],
        "visual_review_csv_sha256": sha256(qa / "qa_review_completed.csv"),
        "qa_auto_audit_sha256": sha256(qa / "qa_auto_audit.json"),
        "qa200_manifest_sha256": sha256(qa / "pspnet_manifest.json"),
        "checkpoint_sha256": EXPECTED_WEIGHTS_SHA256,
        "pilot_source_sha256": sha256(Path(__file__).resolve()),
        "pilot_csv_sha256": sha256(csv_out),
        "threshold": CANDIDATE_THRESHOLD,
        "torch_version": torch.__version__,
        "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        "device": device_name,
        "mask_format": "npz with original-grid boolean left/right and schema_version=1",
        "mask_field_rule": "Only REL_CANDIDATE_ONLY masks are persisted; skip cases always retain BCE image eligibility",
        "limitations": [
            "Known visual-review states exist only for frozen QA200; not available for all TRAIN images.",
            "Per-image geometry heuristics can miss anatomical undersegmentation.",
            "This pilot does not validate sampler safety, lesion exclusion or complete TRAIN processing.",
            "No lung-mask ground truth; no Dice/IoU claim.",
        ],
    }
    with json_out.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    logger.info("P2 PILOT RUN END: COMPUTE COMPLETE | candidate=%d skip=%d",
                manifest["counts"].get("REL_CANDIDATE_ONLY", 0), manifest["counts"].get("SKIP_REL", 0))
    print("P2 PILOT COMPUTE COMPLETE; REVIEW REQUIRED")
    print("Audit: results/qa/pilot_mask_manifest.json")





# --- Fresh policy pilot (formerly pilot_mask_policy.py) ---

from refactor_helpers import verify_recorded_source_sha

import argparse
import csv
import importlib.metadata
import inspect
import json
import logging
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mask_policy import shape_features
from generate_lung_masks import (
    CANDIDATE_THRESHOLD, EXPECTED_WEIGHTS_SHA256, LEFT_INDEX, RIGHT_INDEX,
    WEIGHTS_NAME, create_overlay, read_csv, resize_binary_nearest, sha256,
    verified_split,
)
from mask_policy import POLICY_ID, decide_geometry

from mask_qa import dicom_path, make_contact_sheets, validate_dicom_record
from paths import PROJECT_ROOT

SEED = 42
SAMPLE_PER_STRATUM = 6
SAMPLE_SIZE = 6 * SAMPLE_PER_STRATUM
STRATA = (
    ("Lung Opacity", "AP"), ("Lung Opacity", "PA"),
    ("Normal", "AP"), ("Normal", "PA"),
    ("No Lung Opacity / Not Normal", "AP"),
    ("No Lung Opacity / Not Normal", "PA"),
)
CSV_FIELDS = (
    "image_id", "final_label", "view_position", "geometry_status",
    "policy_for_diagnostic_only", "geometry_reasons", "left_fraction",
    "right_fraction", "left_largest_component_share",
    "right_largest_component_share", "left_vertical_span",
    "right_vertical_span", "side_area_ratio", "side_height_ratio",
    "overlap_fraction", "overlay",
)


def choose_fresh_train(
    manifest_by_id: dict[str, dict[str, str]],
    split_rows: list[dict[str, str]],
    excluded_patient_ids: set[str],
    seed: int = SEED,
) -> list[dict[str, str]]:
    """Sample six class/view strata, one image per new TRAIN patient."""
    split_by_id = {r["image_id"]: r for r in split_rows}
    if len(split_by_id) != len(split_rows) or set(split_by_id) != set(manifest_by_id):
        raise ValueError("Split and manifest image IDs do not match")
    by_stratum: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for image_id, row in manifest_by_id.items():
        spl = split_by_id[image_id]
        if row["patient_id"] != spl["patient_id"]:
            raise ValueError("Patient identity mismatch")
        if spl["split"] != "train" or row["patient_id"] in excluded_patient_ids:
            continue
        key = (row["final_label"], row["view_position"])
        if key not in STRATA:
            raise ValueError("Unexpected final label or projection")
        by_stratum[key][row["patient_id"]].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    used = set(excluded_patient_ids)
    for key in STRATA:
        available = sorted(p for p in by_stratum[key] if p not in used)
        if len(available) < SAMPLE_PER_STRATUM:
            raise ValueError(f"Not enough disjoint TRAIN patients for {key}")
        for patient in rng.sample(available, SAMPLE_PER_STRATUM):
            records = sorted(by_stratum[key][patient], key=lambda x: x["image_id"])
            row = records[rng.randrange(len(records))]
            selected.append(row)
            used.add(patient)
    if len(selected) != SAMPLE_SIZE or len({r["patient_id"] for r in selected}) != SAMPLE_SIZE:
        raise AssertionError("Fresh cohort is not patient-disjoint")
    return selected


def validate_policy_source(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict, dict]:
    """Verify frozen QA lineage; do not pretend this policy is approved for bulk."""
    reviewed, manifest_by_id, qa_manifest, auto_audit = verified_evidence(root)
    qa = root / "results/qa"
    policy_file = qa / "mask_policy_audit.json"
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    if policy.get("status") != "POLICY_AUDIT_COMPLETE_NOT_APPROVED_FOR_FULL_TRAIN":
        raise ValueError("Expected the unapproved full-TRAIN policy audit")
    if policy.get("policy_id") != POLICY_ID:
        raise ValueError("Mask policy ID changed")
    for key, p in (
        ("audit_csv_sha256", qa / "mask_policy_audit.csv"),
        ("pilot_manifest_sha256", qa / "pilot_mask_manifest.json"),
    ):
        if policy.get(key) != sha256(p):
            raise ValueError(f"Mask policy provenance mismatch: {key}")
    verify_recorded_source_sha(root, policy.get("source_sha256"), "mask_policy.py")
    if policy.get("split_sha256") != auto_audit["split_sha256"]:
        raise ValueError("Policy references different frozen split")
    return reviewed, manifest_by_id, qa_manifest, policy


def run_policy_pilot(root: Path, device_name: str, logger: logging.Logger) -> None:
    import pydicom
    import torch
    import torchxrayvision as xrv

    qa = root / "results/qa"
    outputs = {
        "csv": qa / "mask_policy_fresh_pilot.csv",
        "json": qa / "mask_policy_fresh_pilot.json",
        "overlays": qa / "overlays/policy_fresh",
        "sheets": qa / "qa_policy_fresh_sheets",
        "stage": qa / ".mask_policy_fresh_stage",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Fresh pilot artifacts already exist; do not overwrite")
    reviewed, manifest, qa_manifest, policy = validate_policy_source(root)
    split_rows, split_hash = verified_split(root)
    if split_hash != policy["split_sha256"]:
        raise ValueError("Split hash drift")
    smoke = json.loads((qa / "pspnet_smoke.json").read_text(encoding="utf-8"))
    excluded = {manifest[r["image_id"]]["patient_id"] for r in reviewed}
    excluded.update(manifest[r["image_id"]]["patient_id"] for r in smoke["images"])
    chosen = choose_fresh_train(manifest, split_rows, excluded)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    weights = root / "models/pretrained" / WEIGHTS_NAME
    if sha256(weights) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("PSPNet checkpoint hash changed")
    if importlib.metadata.version("torchxrayvision") != "1.5.7":
        raise ValueError("TorchXRayVision version changed")
    if torch.__version__ != qa_manifest["torch_version"]:
        raise ValueError("PyTorch version drift")
    if sha256(Path(inspect.getfile(xrv.baseline_models.chestx_det.PSPNet))) != qa_manifest["wrapper_sha256"]:
        raise ValueError("PSPNet wrapper source drift")
    if (CANDIDATE_THRESHOLD, LEFT_INDEX, RIGHT_INDEX) != (0.5, 4, 5):
        raise ValueError("PSPNet preprocessing contract drift")
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(weights.parent)).to(device_name).eval()
    if (len(model.targets) != 14 or model.targets[LEFT_INDEX] != "Left Lung" or
            model.targets[RIGHT_INDEX] != "Right Lung"):
        raise ValueError("Unexpected target mapping")
    stage = outputs["stage"]
    stage.mkdir(parents=True, exist_ok=False)
    ov = stage / "overlays"; ov.mkdir()
    sheets = stage / "sheets"  # make_contact_sheets creates this directory
    boxes_by_id = defaultdict(list)
    for box in read_csv(root / "data/processed/boxes.csv"):
        boxes_by_id[box["image_id"]].append(box)
    result = []
    logger.info("RUN START P2 fresh policy diagnostic | %s | split=%s | n=%d",
                POLICY_ID, split_hash, SAMPLE_SIZE)
    for index, row in enumerate(chosen, 1):
        image_id = row["image_id"]
        ds = pydicom.dcmread(dicom_path(root, row["dicom_path"]))
        display, normalized = validate_dicom_record(root, row, ds)
        if display.shape != (1024, 1024):
            raise ValueError(f"Unrecognized DICOM geometry: {image_id}")
        tensor = torch.from_numpy(normalized.copy())[None, None].to(device_name)
        with torch.inference_mode():
            logits = model(tensor)
            if tuple(logits.shape) != (1, 14, 512, 512) or not bool(torch.isfinite(logits).all()):
                raise ValueError(f"Invalid PSPNet output: {image_id}")
            probs = torch.sigmoid(logits[0, [LEFT_INDEX, RIGHT_INDEX]]).cpu().numpy()
        native_left = probs[0] > CANDIDATE_THRESHOLD
        native_right = probs[1] > CANDIDATE_THRESHOLD
        geometry = shape_features(native_left, native_right)
        decision = decide_geometry(geometry)
        left = resize_binary_nearest(native_left, display.shape[::-1])
        right = resize_binary_nearest(native_right, display.shape[::-1])
        create_overlay(display, left, right, boxes_by_id[image_id], ov / f"qa_{Path(image_id).stem}.png")
        result.append({
            "image_id": image_id,
            "final_label": row["final_label"],
            "view_position": row["view_position"],
            "geometry_status": decision.status,
            "policy_for_diagnostic_only": decision.rel_policy,
            "geometry_reasons": "|".join(decision.reasons),
            "left_fraction": f"{geometry['left_fraction']:.6f}",
            "right_fraction": f"{geometry['right_fraction']:.6f}",
            "left_largest_component_share": f"{geometry['left_largest_component_share']:.6f}",
            "right_largest_component_share": f"{geometry['right_largest_component_share']:.6f}",
            "left_vertical_span": f"{geometry['left_vertical_span']:.6f}",
            "right_vertical_span": f"{geometry['right_vertical_span']:.6f}",
            "side_area_ratio": f"{geometry['side_area_ratio']:.6f}",
            "side_height_ratio": f"{geometry['side_height_ratio']:.6f}",
            "overlap_fraction": f"{geometry['overlap_fraction']:.6f}",
            "overlay": f"results/qa/overlays/policy_fresh/qa_{Path(image_id).stem}.png",
        })
        logger.info("FRESH POLICY | %d/%d | %s | %s", index, SAMPLE_SIZE, image_id, decision.status)
        if index % 6 == 0:
            print(f"P2 fresh pilot: {index}/{SAMPLE_SIZE}")
        del tensor, logits, probs
    names = make_contact_sheets(result, ov, sheets)
    csv_stage = stage / "mask_policy_fresh_pilot.csv"
    with csv_stage.open("x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
        writer.writeheader(); writer.writerows(result)
    summary = {
        "phase": "P2", "scope": "FRESH_TRAIN_36_IMAGE_DIAGNOSTIC_ONLY",
        "status": "COMPUTE_COMPLETE_NOT_ANATOMICAL_VALIDATION",
        "full_train_authorized": False,
        "seed": SEED, "image_count": len(result),
        "unique_patient_count": len({r["patient_id"] for r in chosen}),
        "strata": {f"{a} | {b}": SAMPLE_PER_STRATUM for a, b in STRATA},
        "counts": dict(sorted(Counter(r["geometry_status"] for r in result).items())),
        "policy_id": POLICY_ID, "source_sha256": sha256(Path(__file__).resolve()),
        "mask_policy_source_sha256": sha256(root / "code/mask_policy.py"),
        "prior_policy_audit_sha256": sha256(qa / "mask_policy_audit.json"),
        "qa_review_manifest_sha256": sha256(qa / "qa_review_manifest.json"),
        "split_sha256": split_hash, "checkpoint_sha256": EXPECTED_WEIGHTS_SHA256,
        "torch_version": torch.__version__, "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        "csv_sha256": sha256(csv_stage),
        "overlays_sha256": {r["image_id"]: sha256(ov / f"qa_{Path(r['image_id']).stem}.png") for r in result},
        "sheets_sha256": {name: sha256(sheets / name) for name in names},
        "image_ids": [r["image_id"] for r in result],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Label-blind geometry screen can miss anatomical errors.",
            "No lung ground truth: this is not segmentation validation.",
            "36 quota-stratified TRAIN images do not establish full-cohort prevalence.",
            "No masks are saved for training and no full-TRAIN policy is approved.",
        ],
    }
    (stage / "mask_policy_fresh_pilot.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.rename(ov, outputs["overlays"])
    os.rename(sheets, outputs["sheets"])
    os.rename(csv_stage, outputs["csv"])
    os.rename(stage / "mask_policy_fresh_pilot.json", outputs["json"])
    stage.rmdir()
    logger.info("RUN END: COMPUTE COMPLETE | count=%d | status=%s", len(result), summary["status"])
    print("P2 fresh-TRAIN policy pilot complete; full-TRAIN policy NOT approved")






def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--lung-pilot',action='store_true')
    mode.add_argument('--policy-pilot',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    policy=bool(args.policy_pilot)
    filename='pilot_mask_policy.log' if policy else 'pilot_lung_masks.log'
    loggername='bgrel.p2.pilot_mask_policy' if policy else 'bgrel.p2.pilot_lung_masks'
    logfile=PROJECT_ROOT / 'results/logs' / filename
    logfile.parent.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger(loggername)
    logger.setLevel(logging.INFO);logger.propagate=False
    handler=logging.FileHandler(logfile,mode='a',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(handler)
    try:
        if policy:run_policy_pilot(PROJECT_ROOT,args.device,logger)
        else:run_lung_pilot(PROJECT_ROOT,args.device,logger)
        return 0
    except Exception:
        logger.exception('RUN END: FAIL | P2 pilot')
        raise
    finally:
        logger.removeHandler(handler);handler.close()

if __name__ == '__main__':
    raise SystemExit(main())
