
"""BG-REL V3.2: fixed-control regional evaluation on frozen VALIDATION.

Compute REM, Top-1, Delta-L, Delta-C and RIE for B0 and A3.
Reuse the frozen P2 PSPNet configuration and P3 sampling algorithm.

The code does not train models, modify P3, or access TEST pixels.
Controls are geometry-screened unannotated regions, not known healthy tissue.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from mask_policy import shape_features
from dataset import decode_dicom, load_frozen, prepare_image, sha256
from generate_lung_masks import (
    CANDIDATE_THRESHOLD,
    EXPECTED_WEIGHTS_SHA256,
    LEFT_INDEX,
    RIGHT_INDEX,
    WEIGHTS_NAME,
    normalize_dicom_pixels,
    resize_binary_nearest,
)
from mask_policy import decide_geometry
import importlib.util

_MODEL_SPEC = importlib.util.spec_from_file_location(
    "_bgrel_evaluation_model",
    Path(__file__).with_name("model.py"),
)

if _MODEL_SPEC is None or _MODEL_SPEC.loader is None:
    raise ImportError("Cannot load BG-REL classifier module")

_bgrel_model = importlib.util.module_from_spec(_MODEL_SPEC)
_MODEL_SPEC.loader.exec_module(_bgrel_model)

MobileNetV2OS16 = _bgrel_model.MobileNetV2OS16
from regions import scale_boxes_xyxy, shared_region_scores
from sample_controls import (
    EXPECTED_BOXES,
    Lung,
    Rect,
    rect_from_row,
    sample_lesion,
)
from paths import PROJECT_ROOT as ROOT
from paths import dicom_from_manifest

RUNS = ("b0_dev_s42", "a3_l100_s42")
P8A = ROOT / "results/evaluation/p8a_val"
OUTPUT = ROOT / "results/evaluation/regions_val"
STAGING = ROOT / "results/evaluation/.regions_val_stage"

SIGMA = 12.0
BOOTSTRAPS = 2000
SEED = 42

FIELDS = (
    "image_id", "patient_id", "annotation_id", "view_position",
    "run_id", "lesion_area", "k", "tier1", "tier2",
    "mask_status", "skip_reason",
    "lesion_logit", "control_mean_logit",
    "rem", "top1",
    "delta_l_blur", "delta_c_blur", "rie_blur",
    "delta_l_fill", "delta_c_fill", "rie_fill",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_inputs():
    """Read only frozen TRAIN/VAL metadata, never TEST pixels."""
    _, val = load_frozen(ROOT)
    require(len(val) == 2969, "Frozen VAL count changed")

    positive = [r for r in val if r["target"] == "1"]
    require(len(positive) == 711, "Positive VAL count changed")

    box_file = ROOT / "data/processed/boxes.csv"
    require(
        sha256(box_file) == EXPECTED_BOXES,
        "Annotation boxes changed",
    )

    val_by_id = {r["image_id"]: r for r in val}
    boxes = defaultdict(list)

    for box in read_csv(box_file):
        sid = box["image_id"]
        if sid in val_by_id:
            require(
                val_by_id[sid]["target"] == "1",
                "Negative VAL image has a lesion box",
            )
            boxes[sid].append(box)

    require(
        sum(len(boxes[r["image_id"]]) for r in positive) == 1122,
        "VAL lesion count changed",
    )

    for row in positive:
        sid = row["image_id"]
        require(boxes[sid], f"Missing boxes: {sid}")
        require(
            len(boxes[sid]) == int(row["num_boxes"]),
            f"Box count mismatch: {sid}",
        )

    p8a_manifest = read_json(P8A / "manifest.json")
    require(
        p8a_manifest["scope"] == "VALIDATION_ONLY"
        and p8a_manifest["status"] == "VAL_EVALUATION_COMPLETE",
        "P8a provenance invalid",
    )

    predictions = {}
    for run_id in RUNS:
        relative = f"{run_id}/predictions.csv"
        path = P8A / relative

        require(
            sha256(path) == p8a_manifest["file_sha256"][relative],
            f"Prediction hash changed: {run_id}",
        )

        records = read_csv(path)
        predictions[run_id] = {r["image_id"]: r for r in records}

        require(
            len(records) == 2969
            and set(predictions[run_id]) == set(val_by_id),
            "P8a prediction cohort changed",
        )

    p3_manifest = read_json(ROOT / "results/qa/sampler_manifest.json")
    require(
        sha256(ROOT / "code/sample_controls.py")
        == p3_manifest["source_sha256"],
        "Frozen P3 sampler source changed",
    )

    return sorted(positive, key=lambda r: r["image_id"]), boxes, predictions


def load_models(device):
    models = {}

    for run_id in RUNS:
        folder = ROOT / "results/runs" / run_id
        summary = read_json(folder / "summary.json")
        checkpoint_path = folder / "best.pt"

        require(
            sha256(checkpoint_path) == summary["checkpoint_sha256"],
            f"Checkpoint hash mismatch: {run_id}",
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        require(
            checkpoint["epoch"] == summary["best_epoch"],
            "Checkpoint epoch mismatch",
        )

        model = MobileNetV2OS16(pretrained=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        models[run_id] = model.to(device).eval()

    return models


def load_pspnet(device):
    import torchxrayvision as xrv

    weights = ROOT / "models/pretrained" / WEIGHTS_NAME
    require(weights.is_file(), "PSPNet weights missing")
    require(
        sha256(weights) == EXPECTED_WEIGHTS_SHA256,
        "PSPNet weights hash mismatch",
    )

    model = xrv.baseline_models.chestx_det.PSPNet(
        cache_dir=str(weights.parent)
    ).to(device).eval()

    require(
        len(model.targets) == 14
        and model.targets[LEFT_INDEX] == "Left Lung"
        and model.targets[RIGHT_INDEX] == "Right Lung",
        "PSPNet anatomical channels changed",
    )

    return model


@torch.inference_mode()
def predict_lungs(pspnet, row, device):
    """Use original P2 DICOM normalization and fixed segmentation."""
    import pydicom

    path = dicom_from_manifest(row["dicom_path"], root=ROOT)
    require(path.is_file(), f"Missing DICOM: {path}")

    dicom = pydicom.dcmread(str(path))
    require(
        str(dicom.SOPInstanceUID) == row["sop_uid"],
        "DICOM SOP UID mismatch",
    )
    require(
        str(dicom.PatientID) == row["dicom_patient_id"],
        "DICOM patient mismatch",
    )

    display, normalized = normalize_dicom_pixels(dicom)
    require(
        display.shape == (1024, 1024),
        "Invalid DICOM dimensions",
    )

    image = torch.from_numpy(
        normalized.copy()
    )[None, None].to(device)

    logits = pspnet(image)

    require(
        tuple(logits.shape) == (1, 14, 512, 512)
        and bool(torch.isfinite(logits).all()),
        "Invalid PSPNet output",
    )

    probs = torch.sigmoid(
        logits[0, [LEFT_INDEX, RIGHT_INDEX]]
    ).cpu().numpy()

    left_small = probs[0] > CANDIDATE_THRESHOLD
    right_small = probs[1] > CANDIDATE_THRESHOLD

    features = shape_features(left_small, right_small)
    decision = decide_geometry(features)

    if decision.rel_policy != "REL_CANDIDATE_ONLY":
        return None, decision.status

    left = resize_binary_nearest(left_small, (1024, 1024))
    right = resize_binary_nearest(right_small, (1024, 1024))

    require(
        left.dtype == np.bool_ and right.dtype == np.bool_,
        "Unexpected lung mask type",
    )

    return {"left": Lung(left), "right": Lung(right)}, decision.status


def region_mask(rect):
    """Rasterize fractional XYXY using source-pixel coverage."""
    values = (rect.x0, rect.y0, rect.x1, rect.y1)

    require(
        all(math.isfinite(v) for v in values)
        and rect.valid(),
        "Invalid intervention rectangle",
    )

    x0, y0 = math.floor(rect.x0), math.floor(rect.y0)
    x1, y1 = math.ceil(rect.x1), math.ceil(rect.y1)

    mask = np.zeros((1024, 1024), dtype=np.bool_)
    mask[y0:y1, x0:x1] = True
    require(mask.any(), "Empty intervention mask")

    return mask


def modify_image(image, rect, mode, blurred):
    mask = region_mask(rect)
    changed = image.copy()

    if mode == "blur":
        changed[mask] = blurred[mask]
    elif mode == "fill":
        changed[mask] = float(image.mean())
    else:
        raise ValueError(f"Unknown intervention: {mode}")

    return changed


@torch.inference_mode()
def probabilities(model, images, device):
    """Inference with fixed batch size 4, matching P8a."""
    result = []

    for start in range(0, len(images), 4):
        chunk = images[start:start + 4]
        valid_count = len(chunk)

        require(valid_count > 0, "Empty inference batch")

        # Pad the final batch, but return only real predictions.
        padded = chunk + [chunk[-1]] * (4 - valid_count)

        batch = torch.stack(
            [prepare_image(image) for image in padded]
        ).to(device)

        require(
            batch.shape == (4, 3, 224, 224),
            "Unexpected inference batch shape",
        )

        output = model(batch).float()

        require(
            output.shape == (4,)
            and bool(torch.isfinite(output).all()),
            "Invalid intervention logits",
        )

        result.extend(
            torch.sigmoid(output[:valid_count]).cpu().tolist()
        )

    require(
        len(result) == len(images),
        "Inference result count mismatch",
    )

    return result


@torch.inference_mode()
def original_region_scores(model, raw, rectangles, device):
    """Compute region logits and P8a-compatible image probability."""
    require(rectangles, "No regions supplied")

    image = prepare_image(raw)[None].to(device)

    # P8a uses batches of four. The diagnostic confirmed that
    # repeating the image to four samples reproduces its logit.
    batch = image.expand(4, -1, -1, -1)

    features_batch = model.forward_features(batch)

    require(
        features_batch.shape == (4, 1280, 14, 14),
        "Unexpected feature map shape",
    )

    image_logits = model.classifier(
        features_batch.mean(dim=(2, 3))
    ).flatten()

    original_probability = float(
        torch.sigmoid(image_logits[0].float()).item()
    )

    # Region pooling refers only to the first image.
    features = features_batch[:1]

    boxes = torch.tensor(
        [
            [r.x0, r.y0, r.x1, r.y1]
            for r in rectangles
        ],
        dtype=torch.float32,
        device=device,
    )

    boxes = scale_boxes_xyxy(
        boxes,
        source_size=(1024, 1024),
        input_size=(224, 224),
    )

    owners = torch.zeros(
        len(rectangles),
        dtype=torch.int64,
        device=device,
    )

    scores = shared_region_scores(
        features,
        model.classifier,
        boxes,
        owners,
        input_size=(224, 224),
    )

    require(
        len(scores) == len(rectangles)
        and bool(torch.isfinite(scores).all()),
        "Invalid region scores",
    )

    return (
        original_probability,
        scores.float().cpu().numpy(),
    )


def intervention_metrics(model, raw, blurred, lesion, controls,
                         original_probability, device):
    result = {}

    for mode in ("blur", "fill"):
        variants = [
            modify_image(raw, lesion, mode, blurred)
        ]
        variants.extend(
            modify_image(raw, rect, mode, blurred)
            for rect in controls
        )

        output = probabilities(model, variants, device)
        lesion_delta = original_probability - output[0]
        control_delta = float(np.mean([
            original_probability - p for p in output[1:]
        ]))

        result[f"delta_l_{mode}"] = float(lesion_delta)
        result[f"delta_c_{mode}"] = float(control_delta)
        result[f"rie_{mode}"] = float(
            lesion_delta - control_delta
        )

    return result


def patient_bootstrap(rows, key):
    """Patient-equal-weight, paired B0/A3 bootstrap."""
    paired = defaultdict(dict)

    for row in rows:
        if row["k"] > 0:
            identity = (row["image_id"], row["annotation_id"])
            paired[identity][row["run_id"]] = row

    patients = defaultdict(list)
    b0_values = []
    a3_values = []

    for identity, values in paired.items():
        require(
            set(values) == set(RUNS),
            "Missing paired model result",
        )
        b0 = float(values[RUNS[0]][key])
        a3 = float(values[RUNS[1]][key])

        require(
            values[RUNS[0]]["patient_id"]
            == values[RUNS[1]]["patient_id"],
            "Patient pairing mismatch",
        )

        patients[values[RUNS[0]]["patient_id"]].append(
            a3 - b0
        )

        b0_values.append(b0)
        a3_values.append(a3)

    require(patients, f"No eligible samples for {key}")

    per_patient = np.array(
        [
            np.mean(patients[pid])
            for pid in sorted(patients)
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(SEED)
    n = len(per_patient)
    estimates = np.empty(BOOTSTRAPS, dtype=np.float64)

    for i in range(BOOTSTRAPS):
        sampled = rng.integers(0, n, size=n)
        estimates[i] = per_patient[sampled].mean()

    return {
        "n_lesions": len(paired),
        "n_patients": n,
        "b0_lesion_mean": float(np.mean(b0_values)),
        "a3_lesion_mean": float(np.mean(a3_values)),
        "lesion_mean_difference": float(
            np.mean(np.asarray(a3_values) - np.asarray(b0_values))
        ),
        "patient_mean_difference": float(per_patient.mean()),
        "patient_ci95": [
            float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5)),
        ],
    }


def write_results(records, counts, eligible_images):
    require(records, "No regional evaluation results")
    require(
        not STAGING.exists() and not OUTPUT.exists(),
        "Evaluation output/staging already exists",
    )

    STAGING.mkdir(parents=True, exist_ok=False)

    csv_path = STAGING / "regions.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)

    metrics = {}
    for key in (
        "rem", "top1",
        "delta_l_blur", "delta_c_blur", "rie_blur",
        "delta_l_fill", "delta_c_fill", "rie_fill",
    ):
        metrics[key] = patient_bootstrap(records, key)

    report = {
        "phase": "P8b",
        "status": "REGIONAL_EVALUATION_COMPLETE",
        "scope": "FROZEN_VALIDATION_POSITIVE_IMAGES",
        "n_positive_images": 711,
        "n_lesion_boxes": 1122,
        "eligible_images": eligible_images,
        "k_distribution": {
            str(k): int(counts[k]) for k in range(3)
        },
        "mask_status_counts": dict(counts["mask_status"]),
        "skip_reasons": dict(counts["skip_reason"]),
        "methods": metrics,
        "sampling": {
            "algorithm": "frozen_P3_sample_lesion",
            "seed": SEED,
            "budget_per_tier": 128,
            "step": 8,
            "max_controls": 2,
            "controls_model_score_independent": True,
        },
        "intervention": {
            "source_grid": [1024, 1024],
            "gaussian_sigma": SIGMA,
            "mean_fill": "original_image_global_mean",
            "unit": "per_lesion_control_group",
            "control_effect": "mean_across_matched_controls",
            "rie": "delta_l_minus_delta_c",
        },
        "statistics": {
            "method": "paired_patient_cluster_percentile_bootstrap",
            "n_resamples": BOOTSTRAPS,
            "patient_equal_weighted": True,
            "seed": SEED,
        },
        "limitations": [
            "Control is unannotated, not clinically certified healthy",
            "PSPNet masks are geometrically screened, not anatomical ground truth",
            "K=0 lesions excluded from matched regional metrics",
            "Only seed 42 model pair was evaluated",
            "RIE is an intervention contrast, not definitive causal attribution",
            "Mean-fill and Gaussian blur may introduce perturbation artifacts",
        ],
        "inputs": {
            "split_sha256": sha256(
                ROOT / "data/processed/split.csv"
            ),
            "boxes_sha256": sha256(
                ROOT / "data/processed/boxes.csv"
            ),
            "pspnet_sha256": EXPECTED_WEIGHTS_SHA256,
            "b0_checkpoint_sha256": sha256(
                ROOT / "results/runs/b0_dev_s42/best.pt"
            ),
            "a3_checkpoint_sha256": sha256(
                ROOT / "results/runs/a3_l100_s42/best.pt"
            ),
            "regions_csv_sha256": sha256(csv_path),
            "source_sha256": sha256(
                ROOT / "code/evaluate_regions.py"
            ),
        },
        "test_pixels_opened": False,
    }

    with (STAGING / "summary.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(
            report, handle,
            indent=2, ensure_ascii=False, allow_nan=False
        )
        handle.write("\n")

    STAGING.rename(OUTPUT)

    print("P8B_REGIONAL_EVALUATION_PASS")
    print(json.dumps({
        "eligible_images": eligible_images,
        "k_distribution": report["k_distribution"],
        "rem": metrics["rem"],
        "top1": metrics["top1"],
        "rie_blur": metrics["rie_blur"],
        "rie_fill": metrics["rie_fill"],
    }, indent=2))


def run(device):
    require(
        not OUTPUT.exists() and not STAGING.exists(),
        "Output already exists; refusing overwrite",
    )

    positive, boxes, predictions = get_inputs()
    models = load_models(device)
    pspnet = load_pspnet(device)

    records = []
    counts = Counter()
    counts["mask_status"] = Counter()
    counts["skip_reason"] = Counter()
    eligible_images = 0

    with torch.inference_mode():
        for image_index, row in enumerate(positive, 1):
            sid = row["image_id"]
            raw = decode_dicom(ROOT, row)

            lungs, mask_status = predict_lungs(
                pspnet, row, device
            )
            counts["mask_status"][mask_status] += 1

            boxes_for_image = sorted(
                boxes[sid],
                key=lambda b: (
                    int(b["box_number"]),
                    b["annotation_id"],
                ),
            )

            rects = [
                rect_from_row(b)
                for b in boxes_for_image
            ]

            groups = []

            for annotation, lesion in zip(
                boxes_for_image, rects
            ):
                sample = sample_lesion(
                    sid,
                    annotation["annotation_id"],
                    lesion,
                    rects,
                    lungs,
                )

                controls = [
                    candidate["rect"]
                    for candidate in sample["accepted"]
                ]

                counts[sample["k"]] += 1

                if sample["k"] == 0:
                    counts["skip_reason"][
                        sample["skip"] or "NO_MATCH"
                    ] += 1

                groups.append((
                    annotation,
                    lesion,
                    controls,
                    sample,
                ))

            if any(group[3]["k"] > 0 for group in groups):
                eligible_images += 1

            blurred = gaussian_filter(
                raw, sigma=SIGMA,
                mode="nearest", truncate=3.0
            )

            for run_id in RUNS:
                model = models[run_id]
                prefix = run_id

                for annotation, lesion, controls, sample in groups:
                    result = {
                        "image_id": sid,
                        "patient_id": row["patient_id"],
                        "annotation_id": annotation["annotation_id"],
                        "view_position": row["view_position"],
                        "run_id": prefix,
                        "lesion_area": float(lesion.area),
                        "k": sample["k"],
                        "tier1": int(sample["tier_count"]["1"]),
                        "tier2": int(sample["tier_count"]["2"]),
                        "mask_status": mask_status,
                        "skip_reason": sample["skip"],
                    }

                    if sample["k"] == 0:
                        for field in FIELDS[11:]:
                            result[field] = ""
                        records.append(result)
                        continue

                    rectangles = [lesion, *controls]
                    p0, scores = original_region_scores(
                        model, raw, rectangles, device
                    )

                    reference = float(
                        predictions[run_id][sid]["probability"]
                    )
                    require(
                        abs(p0 - reference) < 1e-4,
                        f"P8a probability mismatch: {run_id}/{sid}",
                    )

                    lesion_logit = float(scores[0])
                    control_logits = scores[1:]

                    result.update({
                        "lesion_logit": lesion_logit,
                        "control_mean_logit": float(
                            control_logits.mean()
                        ),
                        "rem": float(
                            lesion_logit - control_logits.mean()
                        ),
                        "top1": int(
                            lesion_logit > control_logits.max()
                        ),
                    })

                    result.update(intervention_metrics(
                        model, raw, blurred,
                        lesion, controls, p0, device
                    ))
                    records.append(result)

            if image_index % 25 == 0 or image_index == len(positive):
                print(
                    f"P8b {image_index}/{len(positive)} | "
                    f"matched lesions={counts[1] + counts[2]}",
                    flush=True,
                )

    require(
        sum(counts[k] for k in range(3)) == 1122,
        "Incomplete lesion evaluation",
    )

    write_results(records, counts, eligible_images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda"
    )
    args = parser.parse_args()

    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable")

    device = torch.device(args.device)

    if args.preflight:
        positive, boxes, _ = get_inputs()
        models = load_models(device)
        pspnet = load_pspnet(device)

        # One real VALIDATION image validates segmentation and sampling.
        row = positive[0]
        lungs, status = predict_lungs(pspnet, row, device)
        source_boxes = boxes[row["image_id"]]
        rects = [rect_from_row(b) for b in source_boxes]
        sample = sample_lesion(
            row["image_id"],
            source_boxes[0]["annotation_id"],
            rects[0],
            rects,
            lungs,
        )

        print("P8B_PREFLIGHT_PASS")
        print("Positive VAL images:", len(positive))
        print("Lesion boxes:", sum(
            len(boxes[r["image_id"]]) for r in positive
        ))
        print("Models:", ", ".join(models))
        print("Sample mask status:", status)
        print("Sample matched controls:", sample["k"])
        print("Output exists:", OUTPUT.exists())
    else:
        run(device)


if __name__ == "__main__":
    main()
