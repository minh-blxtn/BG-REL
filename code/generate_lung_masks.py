"""P2 PSPNet TRAIN-only smoke inference and visual QA.

This version deliberately supports only --smoke. It does not write training masks.
All binary masks and overlays are provisional until reviewed by a human.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from paths import PROJECT_ROOT
from paths import dicom_from_manifest


TARGETS = ("Lung Opacity", "Normal", "No Lung Opacity / Not Normal")
VIEWS = ("AP", "PA")
WEIGHTS_NAME = "pspnet_chestxray_best_model_4.pth"
EXPECTED_WEIGHTS_SHA256 = "019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc"
LEFT_INDEX = 4
RIGHT_INDEX = 5
CANDIDATE_THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verified_split(root: Path) -> tuple[list[dict[str, str]], str]:
    split_file = root / "data/processed/split.csv"
    seal_file = root / "data/processed/split.sha256"
    audit_file = root / "results/qa/split_audit.json"
    expected = seal_file.read_text(encoding="utf-8").split()[0].lower()
    actual = sha256(split_file)
    if actual != expected:
        raise ValueError(f"Frozen split SHA-256 mismatch: {actual} != {expected}")
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError("P1 audit does not have status PASS")
    rows = read_csv(split_file)
    if not rows or len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Frozen split is empty or has duplicate image IDs")
    patients: dict[str, str] = {}
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Frozen split contains unknown partition")
        previous = patients.setdefault(row["patient_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("Patient leakage detected in frozen split")
    return rows, actual


def select_smoke_rows(
    manifest: list[dict[str, str]],
    split: list[dict[str, str]],
    boxes: list[dict[str, str]],
    count: int,
) -> list[dict[str, str]]:
    """Pick six label-by-view strata, then challenging multi-box TRAIN cases."""
    if not 6 <= count <= 12:
        raise ValueError("Smoke count must be between 6 and 12")
    manifest_by_id = {row["image_id"]: row for row in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("Manifest has duplicate image IDs")
    if set(manifest_by_id) != {row["image_id"] for row in split}:
        raise ValueError("Frozen split and manifest image ID sets differ")
    box_counts: dict[str, int] = defaultdict(int)
    for box in boxes:
        if box["image_id"] not in manifest_by_id:
            raise ValueError("Box belongs to an unknown image ID")
        box_counts[box["image_id"]] += 1
    train_ids = {row["image_id"] for row in split if row["split"] == "train"}
    allowed_strata = {(lab, view) for lab in TARGETS for view in VIEWS}
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in split:
        if row["split"] != "train":
            continue
        item = manifest_by_id[row["image_id"]]
        if row["patient_id"] != item["patient_id"]:
            raise ValueError("Patient ID mismatch between manifest and split")
        key = (item["final_label"], item["view_position"])
        if key not in allowed_strata:
            raise ValueError(f"Unexpected TRAIN class/view stratum: {key}")
        strata[key].append(item)
    selected: list[dict[str, str]] = []
    patients: set[str] = set()
    ids: set[str] = set()

    def add(candidates: list[dict[str, str]]) -> bool:
        for item in candidates:
            if item["patient_id"] not in patients and item["image_id"] not in ids:
                selected.append(item)
                patients.add(item["patient_id"])
                ids.add(item["image_id"])
                return True
        return False

    for label in TARGETS:
        for view in VIEWS:
            candidates = sorted(
                strata.get((label, view), []),
                key=lambda item: (-box_counts[item["image_id"]], item["image_id"]),
            )
            if not add(candidates):
                raise ValueError(f"No eligible TRAIN image for stratum {label}/{view}")

    # Enrich beyond the six mandatory strata with multi-box positive cases.
    extras = sorted(
        (item for item in manifest if
         (item["final_label"] == "Lung Opacity" and
          item["image_id"] in train_ids and
          box_counts[item["image_id"]] >= 2)),
        key=lambda item: (-box_counts[item["image_id"]], item["image_id"]),
    )
    for view in VIEWS:
        if len(selected) >= count:
            break
        add([item for item in extras if item["view_position"] == view])
    while len(selected) < count:
        if not add(extras):
            raise ValueError("Insufficient distinct-patient TRAIN multi-box samples")
    return selected


def normalize_dicom_pixels(ds: object) -> tuple[np.ndarray, np.ndarray]:
    """Return oriented display pixels and PSPNet input in [-1024, 1024]."""
    bits = int(ds.BitsStored)
    if bits < 1 or bits > 16 or int(ds.PixelRepresentation) != 0:
        raise ValueError("Unsupported stored pixel type; review before inference")
    if int(ds.SamplesPerPixel) != 1:
        raise ValueError("Expected single-channel DICOM")
    maxval = float((1 << bits) - 1)
    pixels = np.asarray(ds.pixel_array, dtype=np.float32)
    if pixels.ndim != 2 or pixels.shape != (int(ds.Rows), int(ds.Columns)):
        raise ValueError("Unexpected DICOM pixel geometry")
    if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > maxval:
        raise ValueError("Unexpected DICOM pixel intensity range")
    photo = str(ds.PhotometricInterpretation)
    if photo == "MONOCHROME1":
        pixels = maxval - pixels
    elif photo != "MONOCHROME2":
        raise ValueError(f"Unsupported photometric interpretation: {photo}")
    normalized = pixels * (2048.0 / maxval) - 1024.0
    if normalized.min() < -1024.001 or normalized.max() > 1024.001:
        raise ValueError("PSPNet normalization went out of expected range")
    display = np.rint(pixels * (255.0 / maxval)).clip(0, 255).astype(np.uint8)
    return display, normalized.astype(np.float32, copy=False)


def resize_binary_nearest(mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Binary mask must be 2D boolean")
    width, height = target_size
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    return np.asarray(resized) > 0


def create_overlay(
    display: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    boxes: list[dict[str, str]],
    destination: Path,
) -> None:
    height, width = display.shape
    if left.shape != display.shape or right.shape != display.shape:
        raise ValueError("Overlay and masks must have identical geometry")
    image = Image.fromarray(display).convert("RGBA")
    for mask, color in ((left, (0, 210, 255)), (right, (255, 135, 0))):
        layer = Image.new("RGBA", (width, height), color + (0,))
        layer.putalpha(Image.fromarray(mask.astype(np.uint8) * 75))
        image = Image.alpha_composite(image, layer)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x0, y0 = float(box["x"]), float(box["y"])
        x1, y1 = float(box["x2"]), float(box["y2"])
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Expert bounding box is outside DICOM geometry")
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=3)
    image.convert("RGB").save(destination, format="PNG")


def run_smoke(root: Path, count: int, device_name: str, logger: logging.Logger) -> None:
    import pydicom
    import torch
    import torchxrayvision as xrv

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    device = torch.device(device_name)
    split, split_hash = verified_split(root)
    manifest_file = root / "data/processed/master_manifest.csv"
    boxes_file = root / "data/processed/boxes.csv"
    manifest = read_csv(manifest_file)
    boxes = read_csv(boxes_file)
    selected = select_smoke_rows(manifest, split, boxes, count)
    weights_dir = root / "models/pretrained"
    weights_file = weights_dir / WEIGHTS_NAME
    if sha256(weights_file) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("PSPNet checkpoint SHA-256 differs from verified download")
    qa_dir = root / "results/qa"
    overlay_dir = qa_dir / "overlays"
    summary_file = qa_dir / "pspnet_smoke.json"
    if summary_file.exists():
        raise FileExistsError(f"Smoke report exists: {summary_file}. Do not overwrite.")
    outputs = [overlay_dir / f"smoke_{Path(item['image_id']).stem}.png" for item in selected]
    if any(path.exists() for path in outputs):
        raise FileExistsError("One or more QA overlay filenames already exist")
    logger.info("P2 | split_sha256=%s | model_sha256=%s | samples=%s | device=%s",
                split_hash, EXPECTED_WEIGHTS_SHA256, count, device_name)
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(weights_dir)).to(device).eval()
    if len(model.targets) != 14 or model.targets[LEFT_INDEX] != "Left Lung" or model.targets[RIGHT_INDEX] != "Right Lung":
        raise ValueError("PSPNet anatomical targets changed")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    all_boxes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for box in boxes:
        all_boxes[box["image_id"]].append(box)
    records = []
    for item, destination in zip(selected, outputs):
        dicom_path = dicom_from_manifest(item["dicom_path"], root=root)
        ds = pydicom.dcmread(dicom_path)
        if str(ds.SOPInstanceUID) != item["sop_uid"]:
            raise ValueError(f"SOP UID mismatch for {item['image_id']}")
        display, normalized = normalize_dicom_pixels(ds)
        if display.shape != (int(item["rows"]), int(item["columns"])):
            raise ValueError("Manifest and DICOM geometry differ")
        tensor = torch.from_numpy(normalized.copy())[None, None].to(device)
        with torch.inference_mode():
            logits = model(tensor)
            if logits.ndim != 4 or tuple(logits.shape) != (1, 14, 512, 512):
                raise ValueError(f"Unexpected PSPNet output shape: {tuple(logits.shape)}")
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite PSPNet segmentation logits")
            pair = torch.sigmoid(logits[0, [LEFT_INDEX, RIGHT_INDEX]]).cpu().numpy()
            logits_min = float(logits.min().item())
            logits_max = float(logits.max().item())
        low_res_left = pair[0] > CANDIDATE_THRESHOLD
        low_res_right = pair[1] > CANDIDATE_THRESHOLD
        size = (display.shape[1], display.shape[0])
        left = resize_binary_nearest(low_res_left, size)
        right = resize_binary_nearest(low_res_right, size)
        create_overlay(display, left, right, all_boxes[item["image_id"]], destination)
        total = display.size
        record = {
            "image_id": item["image_id"],
            "final_label": item["final_label"],
            "view_position": item["view_position"],
            "num_boxes": len(all_boxes[item["image_id"]]),
            "shape_original": [int(display.shape[0]), int(display.shape[1])],
            "input_min": float(normalized.min()),
            "input_max": float(normalized.max()),
            "logits_min": logits_min,
            "logits_max": logits_max,
            "left_fraction": round(float(left.sum()) / total, 6),
            "right_fraction": round(float(right.sum()) / total, 6),
            "overlap_fraction": round(float(np.logical_and(left, right).sum()) / total, 6),
            "empty_left": not bool(left.any()),
            "empty_right": not bool(right.any()),
            "overlay": destination.relative_to(root).as_posix(),
            "qa_status": "UNREVIEWED",
        }
        records.append(record)
        logger.info("P2 | image=%s | view=%s | label=%s | left=%.4f | right=%.4f | overlap=%.4f",
                    item["image_id"], item["view_position"], item["final_label"],
                    record["left_fraction"], record["right_fraction"], record["overlap_fraction"])
        del logits, tensor
    report = {
        "phase": "P2",
        "scope": "TRAIN_ONLY_SMOKE",
        "status": "REVIEW_REQUIRED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split_hash,
        "master_manifest_sha256": sha256(manifest_file),
        "boxes_sha256": sha256(boxes_file),
        "checkpoint_sha256": EXPECTED_WEIGHTS_SHA256,
        "checkpoint_name": WEIGHTS_NAME,
        "generator_source_sha256": sha256(Path(__file__)),
        "pspnet_wrapper_source_sha256": sha256(Path(inspect.getfile(xrv.baseline_models.chestx_det.PSPNet))),
        "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        "torch_version": torch.__version__,
        "device": str(device),
        "target_channels": {"left_lung": LEFT_INDEX, "right_lung": RIGHT_INDEX},
        "probability_transform": "sigmoid(logits)",
        "candidate_threshold": CANDIDATE_THRESHOLD,
        "inverse_resize": "nearest-neighbor",
        "image_count": len(records),
        "images": records,
        "limitations": [
            "Not a QA PASS: images require manual orientation and boundary review.",
            "No ground-truth lung masks; Dice/IoU cannot be reported.",
            "Smoke only. No training masks generated.",
        ],
    }
    summary_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("P2 | smoke complete | report=%s | overlays=%s", summary_file, len(records))


def configure_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bgrel.p2.smoke")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8", mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="TRAIN-only QA smoke test")
    parser.add_argument("--count", type=int, default=8, help="6..12 TRAIN images; default 8")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Only --smoke is implemented and permitted in this review build")
    logger = configure_logger(PROJECT_ROOT / "results/logs/generate_lung_masks.log")
    try:
        logger.info("RUN START | P2 | SMOKE")
        run_smoke(PROJECT_ROOT, args.count, args.device, logger)
        logger.info("RUN END: PASS | P2 SMOKE | REVIEW_REQUIRED")
        print("P2 SMOKE COMPUTE PASS; VISUAL REVIEW REQUIRED")
        print("Report: results/qa/pspnet_smoke.json")
        print("Overlays: results/qa/overlays/smoke_*.png")
        return 0
    except Exception:
        logger.exception("RUN END: FAIL | P2 SMOKE")
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
