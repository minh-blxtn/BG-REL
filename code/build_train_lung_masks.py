"""Build provisional P2 lung masks for the entire frozen TRAIN cohort.

The only permitted final mask consumers are downstream modules that also read
train_mask_audit.csv and honor SKIP_REL plus their own per-region constraints.
AUTO_SCREENED is never an anatomical PASS. Never use validation/test here.
"""

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

import numpy as np

from mask_policy import shape_features
from generate_lung_masks import (
    CANDIDATE_THRESHOLD, EXPECTED_WEIGHTS_SHA256, LEFT_INDEX, RIGHT_INDEX,
    WEIGHTS_NAME, read_csv, resize_binary_nearest, sha256, verified_split,
)
from mask_policy import POLICY_ID, apply_known_review, decide_geometry
from mask_pilots import validate_policy_source
from mask_pilots import save_mask
from mask_qa import dicom_path, validate_dicom_record
from paths import PROJECT_ROOT

FIELDS = (
    "image_id", "final_label", "view_position", "num_boxes",
    "geometry_status", "geometry_reasons", "review_override",
    "rel_policy", "skip_reason", "left_fraction", "right_fraction",
    "left_largest_component_share", "right_largest_component_share",
    "left_vertical_span", "right_vertical_span", "side_area_ratio",
    "side_height_ratio", "overlap_fraction", "mask_path", "mask_sha256",
)
RUN_NAME = "train_mask_run.json"
CSV_NAME = "train_mask_audit.csv"
FINAL_NAME = "train_mask_manifest.json"


def verified_inputs(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, str], dict]:
    """Verify P0/P1, PSPNet/QA, policy source, and fresh-pilot evidence."""
    reviewed, manifest, qa_manifest, policy = validate_policy_source(root)
    qa_dir = root / "results/qa"
    split_rows, split_hash = verified_split(root)
    fresh_file = qa_dir / "mask_policy_fresh_pilot.json"
    fresh = json.loads(fresh_file.read_text(encoding="utf-8"))
    if (fresh.get("scope") != "FRESH_TRAIN_36_IMAGE_DIAGNOSTIC_ONLY" or
            fresh.get("status") != "COMPUTE_COMPLETE_NOT_ANATOMICAL_VALIDATION" or
            fresh.get("full_train_authorized") is not False):
        raise ValueError("Fresh pilot provenance or status changed")
    if (fresh.get("split_sha256") != split_hash or
            fresh.get("checkpoint_sha256") != EXPECTED_WEIGHTS_SHA256 or
            fresh.get("prior_policy_audit_sha256") != sha256(qa_dir / "mask_policy_audit.json") or
            fresh.get("csv_sha256") != sha256(qa_dir / "mask_policy_fresh_pilot.csv")):
        raise ValueError("Fresh pilot data/source/checkpoint hashes disagree")
    verify_recorded_source_sha(root, fresh.get("mask_policy_source_sha256"), "mask_policy.py")
    verify_recorded_source_sha(root, fresh.get("source_sha256"), "pilot_mask_policy.py", "mask_pilots.py")
    if fresh.get("image_count") != 36 or len(fresh.get("image_ids", [])) != 36:
        raise ValueError("Fresh pilot sample size changed")
    if policy.get("split_sha256") != split_hash or policy.get("policy_id") != POLICY_ID:
        raise ValueError("Mask policy provenance changed")
    review_by_id = {r["image_id"]: r["review_status"] for r in reviewed}
    if len(review_by_id) != 200:
        raise ValueError("Expected 200 reviewed QA images")
    train = []
    for part in split_rows:
        if part["split"] != "train":
            continue
        record = manifest[part["image_id"]]
        if part["patient_id"] != record["patient_id"]:
            raise ValueError("Patient identity mismatch")
        train.append(record)
    train.sort(key=lambda r: r["image_id"])
    if len(train) != 23748 or len({r["image_id"] for r in train}) != len(train):
        raise ValueError("Unexpected frozen TRAIN image count or duplicates")
    if len({r["image_id"] for r in read_csv(qa_dir / "mask_policy_fresh_pilot.csv")}) != 36:
        raise ValueError("Fresh pilot CSV contains duplicate IDs")
    provenance = {
        "phase": "P2",
        "scope": "FROZEN_TRAIN_PROVISIONAL_MASK_CACHE_ONLY",
        "candidate_is_anatomical_pass": False,
        "REL_use_requires_P3_geometry_checks": True,
        "split_sha256": split_hash,
        "checkpoint_sha256": EXPECTED_WEIGHTS_SHA256,
        "checkpoint_name": WEIGHTS_NAME,
        "threshold": CANDIDATE_THRESHOLD,
        "target_indices": [LEFT_INDEX, RIGHT_INDEX],
        "policy_id": POLICY_ID,
        "source_sha256": sha256(Path(__file__).resolve()),
        "policy_source_sha256": sha256(root / "code/mask_policy.py"),
        "preprocessor_source_sha256": sha256(root / "code/generate_lung_masks.py"),
        "fresh_pilot_sha256": sha256(fresh_file),
        "qa200_review_sha256": sha256(qa_dir / "qa_review_completed.csv"),
        "image_count": len(train),
        "image_ids_sha256": __hash_ids([r["image_id"] for r in train]),
        "original_qa_torch_version": qa_manifest["torch_version"],
        "original_qa_pspnet_wrapper_sha256": qa_manifest["wrapper_sha256"],
    }
    return train, manifest, review_by_id, provenance


def __hash_ids(ids: list[str]) -> str:
    import hashlib
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def validate_progress(rows: list[dict[str, str]], train: list[dict[str, str]], masks_dir: Path) -> None:
    """Ensure a resumed run has no missing, extra, or changed mask files."""
    if len(rows) > len(train):
        raise ValueError("Progress extends beyond frozen TRAIN")
    recorded_masks = set()
    for i, record in enumerate(rows):
        if set(record) != set(FIELDS):
            raise ValueError("Unexpected progress CSV schema")
        if record["image_id"] != train[i]["image_id"]:
            raise ValueError("Progress is not a prefix of frozen TRAIN")
        policy = record["rel_policy"]
        if policy == "REL_CANDIDATE_ONLY":
            if record["skip_reason"]:
                raise ValueError("REL candidate unexpectedly has a skip reason")
            expected = f"{Path(record['image_id']).stem}.npz"
            if record["mask_path"] != f"data/lung_masks/train/{expected}":
                raise ValueError("Mask path changed")
            p = masks_dir / expected
            if not p.is_file() or sha256(p) != record["mask_sha256"]:
                raise ValueError(f"Mask missing or hash mismatch: {expected}")
            with np.load(p, allow_pickle=False) as mask:
                if (set(mask.files) != {"left", "right", "schema_version"} or
                        mask["left"].dtype != np.bool_ or
                        mask["right"].dtype != np.bool_ or
                        mask["left"].shape != (1024, 1024) or
                        mask["right"].shape != (1024, 1024) or
                        int(mask["schema_version"]) != 1):
                    raise ValueError(f"Invalid saved mask schema: {expected}")
            recorded_masks.add(expected)
        elif policy == "SKIP_REL":
            if record["mask_path"] or record["mask_sha256"] or not record["skip_reason"]:
                raise ValueError("Skipped image has an unexpected mask or no reason")
        else:
            raise ValueError("Unknown REL policy in progress")
    present = {p.name for p in masks_dir.glob("*.npz")} if masks_dir.exists() else set()
    if present != recorded_masks:
        raise ValueError("Untracked or missing mask files found; do not overwrite")


def make_record(row: dict[str, str], features: dict[str, float], review: str | None,
                geometry_status: str, geometry_reasons: tuple[str, ...], rel_policy: str,
                override: str, mask_path: str, mask_sha: str) -> dict[str, str]:
    """Persist the distinction between technical screening and REL permission."""
    if (rel_policy == "REL_CANDIDATE_ONLY") != bool(mask_path and mask_sha):
        raise ValueError("Mask artifacts and provisional REL policy are inconsistent")
    # Skip reasons apply only to rejected masks; missing visual review is
    # provenance, not a reason to skip a geometrically screened candidate.
    reasons = []
    if rel_policy == "SKIP_REL":
        reasons = [*geometry_reasons]
        if override and override != "NOT_VISUALLY_REVIEWED":
            reasons.append(override)
        if not reasons:
            raise ValueError("SKIP_REL requires a substantive skip reason")
    return {
        "image_id": row["image_id"],
        "final_label": row["final_label"],
        "view_position": row["view_position"],
        "num_boxes": row["num_boxes"],
        "geometry_status": geometry_status,
        "geometry_reasons": "|".join(geometry_reasons),
        "review_override": review or "NOT_REVIEWED",
        "rel_policy": rel_policy,
        "skip_reason": "|".join(reasons),
        "left_fraction": f"{features['left_fraction']:.6f}",
        "right_fraction": f"{features['right_fraction']:.6f}",
        "left_largest_component_share": f"{features['left_largest_component_share']:.6f}",
        "right_largest_component_share": f"{features['right_largest_component_share']:.6f}",
        "left_vertical_span": f"{features['left_vertical_span']:.6f}",
        "right_vertical_span": f"{features['right_vertical_span']:.6f}",
        "side_area_ratio": f"{features['side_area_ratio']:.6f}",
        "side_height_ratio": f"{features['side_height_ratio']:.6f}",
        "overlap_fraction": f"{features['overlap_fraction']:.6f}",
        "mask_path": mask_path,
        "mask_sha256": mask_sha,
    }


def generate(root: Path, train: list[dict[str, str]], reviews: dict[str, str],
             provenance: dict, limit: int, resume: bool, device: str,
             logger: logging.Logger) -> None:
    import pydicom
    import torch
    import torchxrayvision as xrv

    qa = root / "results/qa"
    masks_dir = root / "data/lung_masks/train"
    progress_file = qa / CSV_NAME
    run_file = qa / RUN_NAME
    final_file = qa / FINAL_NAME
    lock_file = qa / ".train_mask_generation.lock"
    if final_file.exists():
        raise FileExistsError("Completed TRAIN manifest exists; do not rerun")
    if resume:
        if not progress_file.is_file() or not run_file.is_file() or not masks_dir.is_dir():
            raise FileNotFoundError("Resume requested but run artifacts are missing")
    elif any(p.exists() for p in (masks_dir, progress_file, run_file)):
        raise FileExistsError("TRAIN masks/progress already exist; use --resume after review")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    weights = root / "models/pretrained" / WEIGHTS_NAME
    if sha256(weights) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("PSPNet weights hash changed")
    if importlib.metadata.version("torchxrayvision") != "1.5.7":
        raise ValueError("TorchXRayVision version changed")
    if torch.__version__ != provenance["original_qa_torch_version"]:
        raise ValueError("PyTorch version changed")
    if sha256(Path(inspect.getfile(xrv.baseline_models.chestx_det.PSPNet))) != provenance["original_qa_pspnet_wrapper_sha256"]:
        raise ValueError("PSPNet wrapper source hash mismatch")
    if (CANDIDATE_THRESHOLD, LEFT_INDEX, RIGHT_INDEX) != (0.5, 4, 5):
        raise ValueError("PSPNet channel/threshold contract changed")
    if lock_file.exists():
        raise FileExistsError("TRAIN mask lock exists; investigate before retry")
    lock = os.open(lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(lock)
    try:
        if resume:
            if json.loads(run_file.read_text(encoding="utf-8")) != provenance:
                raise ValueError("Run config changed; cannot resume")
            prior = read_csv(progress_file)
            validate_progress(prior, train, masks_dir)
            begin = len(prior)
        else:
            masks_dir.mkdir(parents=True, exist_ok=False)
            with progress_file.open("x", encoding="utf-8", newline="") as fp:
                csv.writer(fp).writerow(FIELDS)
            run_file.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            prior = []
            begin = 0
        model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(weights.parent)).to(device).eval()
        if (len(model.targets) != 14 or model.targets[LEFT_INDEX] != "Left Lung" or
                model.targets[RIGHT_INDEX] != "Right Lung"):
            raise ValueError("PSPNet target mapping mismatch")
        stop = min(len(train), begin + limit)
        logger.info("RUN START | P2 provisional full TRAIN | start=%d stop=%d total=%d", begin, stop, len(train))
        for idx in range(begin, stop):
            row = train[idx]
            image_id = row["image_id"]
            ds = pydicom.dcmread(dicom_path(root, row["dicom_path"]))
            display, normalized = validate_dicom_record(root, row, ds)
            if display.shape != (1024, 1024):
                raise ValueError(f"Unexpected original geometry: {image_id}")
            tensor = torch.from_numpy(normalized.copy())[None, None].to(device)
            with torch.inference_mode():
                logits = model(tensor)
                if tuple(logits.shape) != (1, 14, 512, 512) or not bool(torch.isfinite(logits).all()):
                    raise ValueError(f"Invalid PSPNet logits: {image_id}")
                prob = torch.sigmoid(logits[0, [LEFT_INDEX, RIGHT_INDEX]]).cpu().numpy()
            left_native = prob[0] > CANDIDATE_THRESHOLD
            right_native = prob[1] > CANDIDATE_THRESHOLD
            features = shape_features(left_native, right_native)
            geometry = decide_geometry(features)
            review = reviews.get(image_id)
            rel_policy, override = apply_known_review(geometry, review)
            mask_path = ""
            mask_hash = ""
            if rel_policy == "REL_CANDIDATE_ONLY":
                left = resize_binary_nearest(left_native, display.shape[::-1])
                right = resize_binary_nearest(right_native, display.shape[::-1])
                mask_name = Path(image_id).stem + ".npz"
                path = masks_dir / mask_name
                mask_hash = save_mask(path, left, right)
                mask_path = f"data/lung_masks/train/{mask_name}"
            record = make_record(row, features, review, geometry.status,
                                 geometry.reasons, rel_policy, override, mask_path, mask_hash)
            with progress_file.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writerow(record)
                handle.flush()
                os.fsync(handle.fileno())
            if (idx + 1) % 50 == 0 or idx + 1 == stop:
                logger.info("PROGRESS | %d/%d | %s", idx + 1, len(train), image_id)
                print(f"P2 TRAIN: {idx + 1}/{len(train)}")
            del tensor, logits, prob
        if stop == len(train):
            entries = read_csv(progress_file)
            validate_progress(entries, train, masks_dir)
            result = {
                **provenance,
                "status": "GENERATED_PROVISIONAL_NOT_REL_APPROVED",
                "processed": len(entries),
                "geometry_status_counts": dict(Counter(r["geometry_status"] for r in entries)),
                "rel_policy_counts": dict(Counter(r["rel_policy"] for r in entries)),
                "csv_sha256": sha256(progress_file),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            with final_file.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            logger.info("RUN END: TRAIN MASKS GENERATED PROVISIONAL | %d", len(entries))
            print("P2 TRAIN GENERATION COMPLETED; REL NOT YET APPROVED")
        else:
            logger.info("RUN END: PARTIAL READY FOR RESUME | %d/%d", stop, len(train))
            print("P2 PARTIAL BATCH SAVED; USE --resume FOR NEXT BATCH")
    finally:
        lock_file.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true", help="Read only validation")
    group.add_argument("--generate", action="store_true", help="Write incremental TRAIN artifacts")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--accept-provisional", action="store_true",
                        help="Explicitly acknowledge that masks are not anatomically validated")
    parser.add_argument("--batch-limit", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.batch_limit < 1:
        parser.error("--batch-limit must be >=1")
    if args.generate and not args.accept_provisional:
        parser.error("--generate requires --accept-provisional")
    if args.preflight and (args.resume or args.accept_provisional):
        parser.error("--preflight does not accept write-mode arguments")

    log = logging.getLogger("bgrel.p2.build_train_lung_masks")
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = None
    if args.generate:
        logfile = PROJECT_ROOT / "results/logs/build_train_lung_masks.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        log.addHandler(handler)
    try:
        train, _, reviews, provenance = verified_inputs(PROJECT_ROOT)
        print(f"P2 PREFLIGHT PASS | TRAIN={len(train)} | prior visual screens={len(reviews)}")
        print(f"Split SHA256: {provenance['split_sha256']}")
        print(f"Policy: {provenance['policy_id']} | provisional only; not anatomical PASS")
        weights_path = PROJECT_ROOT / "models/pretrained" / WEIGHTS_NAME
        if not weights_path.is_file() or sha256(weights_path) != EXPECTED_WEIGHTS_SHA256:
            raise ValueError("Checkpoint missing or SHA-256 changed")
        if args.preflight:
            for path in (
                PROJECT_ROOT / "data/lung_masks/train",
                PROJECT_ROOT / "results/qa" / CSV_NAME,
                PROJECT_ROOT / "results/qa" / RUN_NAME,
                PROJECT_ROOT / "results/qa" / FINAL_NAME,
            ):
                if path.exists():
                    print(f"EXISTING TRAIN ARTIFACT: {path}")
            print("P2 READ-ONLY PREFLIGHT COMPLETE; NO MASKS WRITTEN")
        if args.generate:
            generate(PROJECT_ROOT, train, reviews, provenance,
                     args.batch_limit, args.resume, args.device, log)
        return 0
    except Exception:
        if handler:
            log.exception("RUN END: FAIL | P2 TRAIN provisional generator")
        raise
    finally:
        if handler:
            log.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
