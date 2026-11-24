
"""BG-REL V3.2: lesion-union intervention on frozen VALIDATION.

Compare B0 and A3 under identical Gaussian blur and mean-fill.
Use original 1024x1024 DICOM intensities before model preprocessing.
Never train, sample controls, or decode TEST pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from dataset import decode_dicom, load_frozen, prepare_image, sha256
from model import MobileNetV2OS16

RUNS = ("b0_dev_s42", "a3_l100_s42")
P8A = ROOT / "results/evaluation/p8a_val"
OUT = ROOT / "results/evaluation/lesion_val"
STAGE = OUT.with_name(".lesion_val_stage")

BOXES_SHA = (
    "6372a1facc3e1b78fbcd97a5bd103d2fc33ed257c92f8076b1d36e9168078e45"
)

SIGMA = 12.0  # Original 1024-pixel coordinates; fixed, not tuned.
N_BOOTSTRAP = 2000
SEED = 42

COLUMNS = [
    "image_id", "patient_id", "view_position",
    "n_boxes", "lesion_union_fraction",
    "b0_probability", "a3_probability",
    "b0_blur_delta", "a3_blur_delta",
    "b0_fill_delta", "a3_fill_delta",
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def frozen_inputs():
    _, val = load_frozen(ROOT)
    val_by_id = {r["image_id"]: r for r in val}

    require(len(val) == 2969, "Frozen VAL size changed")

    box_path = ROOT / "data/processed/boxes.csv"
    require(
        sha256(box_path) == BOXES_SHA,
        "Frozen annotation boxes changed",
    )

    boxes = defaultdict(list)
    for row in read_csv(box_path):
        if row["image_id"] in val_by_id:
            require(
                val_by_id[row["image_id"]]["target"] == "1",
                "Box associated with negative VALIDATION image",
            )
            boxes[row["image_id"]].append(row)

    positive = sorted(
        (r for r in val if r["target"] == "1"),
        key=lambda r: r["image_id"],
    )

    require(len(positive) > 0, "No positive VAL images")
    require(
        all(boxes[r["image_id"]] for r in positive),
        "Positive VAL image missing lesion boxes",
    )

    p8a_manifest = json.loads(
        (P8A / "manifest.json").read_text(encoding="utf-8")
    )

    require(
        p8a_manifest.get("scope") == "VALIDATION_ONLY"
        and p8a_manifest.get("status") == "VAL_EVALUATION_COMPLETE",
        "P8a manifest invalid",
    )

    predictions = {}

    for run_id in RUNS:
        relative = f"{run_id}/predictions.csv"
        path = P8A / relative
        require(
            sha256(path) == p8a_manifest["file_sha256"][relative],
            f"P8a predictions changed: {run_id}",
        )
        rows = read_csv(path)
        require(len(rows) == 2969, "P8a predictions incomplete")
        by_id = {r["image_id"]: r for r in rows}
        require(len(by_id) == 2969, "Duplicate prediction IDs")
        require(
            set(by_id) == set(val_by_id),
            "Prediction cohort differs from frozen VAL",
        )
        predictions[run_id] = by_id

    for row in positive:
        sid = row["image_id"]
        for run_id in RUNS:
            p = predictions[run_id][sid]
            require(
                p["patient_id"] == row["patient_id"]
                and int(p["target"]) == 1,
                "Prediction/VAL identity or label mismatch",
            )

    return positive, boxes, predictions


def models_on(device):
    result = {}

    for run_id in RUNS:
        folder = ROOT / "results/runs" / run_id
        summary = json.loads(
            (folder / "summary.json").read_text(encoding="utf-8")
        )
        checkpoint_path = folder / "best.pt"

        require(
            sha256(checkpoint_path) == summary["checkpoint_sha256"],
            f"{run_id}: checkpoint integrity failure",
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model = MobileNetV2OS16(pretrained=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        result[run_id] = model

    return result


def union_mask(rows):
    """Rasterize original XYXY boxes to covering source pixels."""
    mask = np.zeros((1024, 1024), dtype=np.bool_)

    for row in rows:
        x0 = float(row["x"])
        y0 = float(row["y"])
        x1 = float(row["x2"])
        y1 = float(row["y2"])

        require(
            all(math.isfinite(v) for v in (x0, y0, x1, y1))
            and 0 <= x0 < x1 <= 1024
            and 0 <= y0 < y1 <= 1024,
            "Invalid lesion geometry",
        )

        # Covering rasterization; any intersected source pixel is masked.
        ix0 = int(math.floor(x0))
        iy0 = int(math.floor(y0))
        ix1 = int(math.ceil(x1))
        iy1 = int(math.ceil(y1))
        mask[iy0:iy1, ix0:ix1] = True

    require(mask.any(), "Empty lesion union")
    return mask


def make_interventions(image, mask):
    blurred = gaussian_filter(
        image, sigma=SIGMA, mode="nearest", truncate=3.0
    )

    blur_image = image.copy()
    blur_image[mask] = blurred[mask]

    fill_image = image.copy()
    fill_image[mask] = float(image.mean())

    return blur_image, fill_image


@torch.inference_mode()
def infer(model, images, device):
    tensor = torch.stack(
        [prepare_image(image) for image in images]
    ).to(device)

    logits = model(tensor)
    require(
        tuple(logits.shape) == (len(images),)
        and bool(torch.isfinite(logits).all()),
        "Non-finite or unexpected model output",
    )

    return torch.sigmoid(logits.float()).cpu().numpy()


def bootstrap_patient(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        groups[row["patient_id"]].append(
            float(row["a3_" + metric + "_delta"])
            - float(row["b0_" + metric + "_delta"])
        )

    patients = sorted(groups)
    patient_values = np.asarray(
        [np.mean(groups[pid]) for pid in patients],
        dtype=np.float64,
    )

    rng = np.random.default_rng(SEED)
    n = len(patient_values)

    samples = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)
        samples[i] = patient_values[indices].mean()

    return {
        "n_patients": n,
        "patient_mean_delta_a3_minus_b0": float(
            patient_values.mean()
        ),
        "ci95_lower": float(np.percentile(samples, 2.5)),
        "ci95_upper": float(np.percentile(samples, 97.5)),
        "bootstrap_samples": N_BOOTSTRAP,
        "seed": SEED,
    }


def summarize(rows):
    results = {}

    for method in ("blur", "fill"):
        b0 = np.asarray(
            [float(r[f"b0_{method}_delta"]) for r in rows]
        )
        a3 = np.asarray(
            [float(r[f"a3_{method}_delta"]) for r in rows]
        )

        results[method] = {
            "b0_mean_delta": float(b0.mean()),
            "a3_mean_delta": float(a3.mean()),
            "b0_median_delta": float(np.median(b0)),
            "a3_median_delta": float(np.median(a3)),
            "b0_fraction_positive": float(np.mean(b0 > 0)),
            "a3_fraction_positive": float(np.mean(a3 > 0)),
            "image_mean_delta_a3_minus_b0": float(
                np.mean(a3 - b0)
            ),
            "patient_bootstrap": bootstrap_patient(
                rows, method
            ),
        }

    return {
        "phase": "P8b_lesion_intervention",
        "partition": "val",
        "status": "LESION_INTERVENTION_COMPLETE",
        "n_positive_images": len(rows),
        "n_patients": len({r["patient_id"] for r in rows}),
        "n_lesion_boxes": sum(int(r["n_boxes"]) for r in rows),
        "methods": results,
        "gaussian_sigma_source_pixels": SIGMA,
        "occlusion": "union_of_all_annotated_lesion_boxes",
        "rasterization": "floor_start_ceil_end_covering",
        "mean_fill": "original_image_global_mean",
        "normalization": "frozen_dataset_prepare_image",
        "interpretation": (
            "Exploratory lesion intervention; positive delta is "
            "probability reduction, not proof of causal lesion reliance."
        ),
        "not_computed": [
            "control_intervention",
            "RIE",
            "REM",
            "Top1",
        ],
        "test_pixels_opened": False,
    }


def run(device):
    require(not OUT.exists(), "Output exists; refusing overwrite")
    require(
        not STAGE.exists(),
        "Staging directory exists; investigate previous run",
    )

    positive, boxes, predictions = frozen_inputs()
    models = models_on(device)

    # Preflight completes before writing output.
    STAGE.mkdir(parents=True, exist_ok=False)

    records = []

    for i, row in enumerate(positive, 1):
        sid = row["image_id"]
        raw = decode_dicom(ROOT, row)
        mask = union_mask(boxes[sid])
        blur, fill = make_interventions(raw, mask)
        images = [raw, blur, fill]

        measurements = {}

        for run_id, prefix in (
            ("b0_dev_s42", "b0"),
            ("a3_l100_s42", "a3"),
        ):
            p = infer(models[run_id], images, device)

            reference = float(
                predictions[run_id][sid]["probability"]
            )

            require(
                abs(float(p[0]) - reference) < 1e-4,
                f"Unperturbed probability mismatch: {run_id}/{sid}",
            )

            measurements[f"{prefix}_probability"] = float(p[0])
            measurements[f"{prefix}_blur_delta"] = float(
                p[0] - p[1]
            )
            measurements[f"{prefix}_fill_delta"] = float(
                p[0] - p[2]
            )

        records.append({
            "image_id": sid,
            "patient_id": row["patient_id"],
            "view_position": row["view_position"],
            "n_boxes": len(boxes[sid]),
            "lesion_union_fraction": float(mask.mean()),
            **measurements,
        })

        if i % 50 == 0 or i == len(positive):
            print(
                f"Lesion evaluation {i}/{len(positive)}",
                flush=True,
            )

    with (STAGE / "per_image.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)

    report = summarize(records)
    report["input_checkpoint_sha256"] = {
        run_id: sha256(
            ROOT / "results/runs" / run_id / "best.pt"
        )
        for run_id in RUNS
    }

    report["per_image_sha256"] = sha256(
        STAGE / "per_image.csv"
    )

    with (STAGE / "summary.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(
            report, handle, indent=2,
            ensure_ascii=False, allow_nan=False,
        )
        handle.write("\n")

    STAGE.rename(OUT)

    print("LESION_EVALUATION_PASS")
    print(json.dumps(report["methods"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable")

    device = torch.device(args.device)

    if args.preflight:
        positive, boxes, _ = frozen_inputs()
        models_on(device)
        print("LESION_PREFLIGHT_PASS")
        print("Positive VAL images:", len(positive))
        print("Lesion boxes:", sum(len(boxes[r["image_id"]]) for r in positive))
        print("Output exists:", OUT.exists())
    else:
        run(device)


if __name__ == "__main__":
    main()
