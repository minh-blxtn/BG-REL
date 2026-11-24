"""Independent P7/B1 frozen-VALIDATION evaluation and paired patient bootstrap.

This module never accesses TEST pixels or writes to existing B0/A3 P8 directories.
Outputs are published exclusively to results/evaluation/b1_val.

    python code/evaluate_b1_val.py --preflight
    python code/evaluate_b1_val.py --run
"""
from __future__ import annotations
from refactor_helpers import read_json, require

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RSNAImageDataset, sha256
from metrics import binary_metrics
from model import MobileNetV2OS16
import evaluate_p8a as p8a

ROOT = Path(__file__).resolve().parents[1]
RUN = "b1_boxpos_s42"
VAL_COUNT = 2969
OUTPUT = ROOT / "results/evaluation/b1_val"
STAGE = ROOT / "results/evaluation/.b1_val_stage"
OLD_P8A = ROOT / "results/evaluation/p8a_val"
N_BOOTSTRAPS = 2000
SEED = 42






def verify_b1(device: torch.device) -> dict:
    folder = ROOT / "results/runs" / RUN
    config = read_json(folder / "config.json")
    summary = read_json(folder / "summary.json")
    require(summary.get("status") == "FINISHED" and summary.get("variant") == "b1", "B1 not finalized")
    require(config.get("run_id") == RUN and config.get("seed") == SEED, "B1 config mismatch")
    require(config.get("test_images_opened") is False and summary.get("test_images_opened") is False, "TEST seal invalid")
    require(summary.get("best_epoch") == 3, "B1 selected epoch differs from reviewed evidence")
    require(sha256(folder / "best.pt") == summary["checkpoint_sha256"], "B1 best checkpoint hash mismatch")
    require(sha256(folder / "metrics.csv") == summary["metrics_sha256"], "B1 metrics hash mismatch")
    sources = summary.get("source_sha256", {})
    for name in ("dataset.py", "model.py", "metrics.py", "regions.py"):
        require(sha256(ROOT / "code" / name) == sources.get(name), f"B1 source drift: {name}")
    require(config.get("split_sha256") == sha256(ROOT / "data/processed/split.csv"), "B1 split mismatch")
    cp = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    require(cp.get("epoch") == summary["best_epoch"], "B1 checkpoint epoch mismatch")
    require(cp.get("config") == config, "B1 checkpoint config mismatch")
    require(math.isclose(float(cp["val_auc"]), float(summary["best_val_auroc"]), rel_tol=0, abs_tol=1e-9), "B1 AUC mismatch")
    model = MobileNetV2OS16(pretrained=False)
    model.load_state_dict(cp["model"], strict=True)
    model.to(device).eval()
    return {"model": model, "config": config, "checkpoint": cp, "checkpoint_sha256": summary["checkpoint_sha256"]}


def verify_old_predictions(val: list[dict]) -> dict[str, list[dict]]:
    manifest = read_json(OLD_P8A / "manifest.json")
    require(manifest.get("scope") == "VALIDATION_ONLY" and manifest.get("status") == "VAL_EVALUATION_COMPLETE", "Original P8a manifest invalid")
    expected_ids = [r["image_id"] for r in val]
    result = {}
    for run_id in ("b0_dev_s42", "a3_l100_s42"):
        relative = f"{run_id}/predictions.csv"
        path = OLD_P8A / relative
        require(sha256(path) == manifest["file_sha256"][relative], f"Old P8a hash mismatch: {run_id}")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == len(val) and [r["image_id"] for r in rows] == expected_ids, f"Old P8a cohort/order mismatch: {run_id}")
        for x, v in zip(rows, val):
            require(x["patient_id"] == v["patient_id"] and int(x["target"]) == int(v["target"]), "Old P8a patient/target mismatch")
        result[run_id] = rows
    return result


def predict_val(val: list[dict], model: MobileNetV2OS16, device: torch.device) -> list[float]:
    loader = DataLoader(RSNAImageDataset(ROOT, val, train=False, seed=SEED, epoch=0),
                        batch_size=4, shuffle=False, num_workers=0, drop_last=False,
                        pin_memory=(device.type == "cuda"))
    output = []
    with torch.inference_mode():
        for images, targets in loader:
            start = len(output)
            require([int(x) for x in targets.tolist()] ==
                    [int(r["target"]) for r in val[start:start + len(targets)]], "VAL order mismatch")
            z = model(images.to(device, non_blocking=True)).float().cpu()
            require(tuple(z.shape) == tuple(targets.shape) and bool(torch.isfinite(z).all()), "Invalid B1 logits")
            output.extend(z.tolist())
            if len(output) % 400 < len(targets):
                print(f"B1 VAL {len(output)}/{len(val)}", flush=True)
    require(len(output) == VAL_COUNT, "Incomplete B1 VAL predictions")
    return output


def paired_bootstrap(val: list[dict], b1: list[dict], previous: dict) -> dict:
    y = np.asarray([int(row["target"]) for row in val], dtype=np.int64)
    scores = {
        "b0_dev_s42": np.asarray([float(r["logit"]) for r in previous["b0_dev_s42"]]),
        "a3_l100_s42": np.asarray([float(r["logit"]) for r in previous["a3_l100_s42"]]),
        RUN: np.asarray([float(r["logit"]) for r in b1]),
    }
    patient_indices = defaultdict(list)
    for i, row in enumerate(val):
        patient_indices[row["patient_id"]].append(i)
    groups = [np.asarray(patient_indices[k], dtype=int) for k in sorted(patient_indices)]
    keys = list(scores)
    point = {name: dict(zip(("auroc", "ap"), binary_metrics(y, z))) for name, z in scores.items()}
    rng = np.random.default_rng(SEED)
    resamples = {name: {"auroc": [], "ap": []} for name in keys}
    invalid = 0
    for _ in range(N_BOOTSTRAPS):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in chosen])
        sy = y[indices]
        if np.unique(sy).size != 2:
            invalid += 1
            continue
        for name in keys:
            auc, ap = binary_metrics(sy, scores[name][indices])
            resamples[name]["auroc"].append(float(auc))
            resamples[name]["ap"].append(float(ap))
    require(invalid == 0, "Invalid bootstrap resamples: inspect before accepting results")
    comparisons = {}
    for old in ("b0_dev_s42", "a3_l100_s42"):
        diffs = {}
        for metric in ("auroc", "ap"):
            a = np.asarray(resamples[RUN][metric]) - np.asarray(resamples[old][metric])
            diffs[metric] = {
                "delta_b1_minus_reference": point[RUN][metric] - point[old][metric],
                "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            }
        comparisons[f"{RUN}_minus_{old}"] = diffs
    return {
        "scope": "FROZEN_VALIDATION_ONLY", "seed": SEED,
        "n_images": VAL_COUNT, "n_patients": len(groups),
        "n_resamples": N_BOOTSTRAPS, "invalid_resamples": invalid,
        "method": "paired_patient_cluster_percentile_bootstrap",
        "point_estimates": point, "comparisons": comparisons,
        "test_pixels_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    device = torch.device(args.device)
    val = p8a.check_frozen_data()
    record = verify_b1(device)
    previous = verify_old_predictions(val)
    require(not STAGE.exists(), f"Unresolved stage folder: {STAGE}")
    if args.preflight:
        print("B1_P8A_PREFLIGHT_PASS", "val_images=2969", "reference_models=B0,A3", "test_pixels_opened=False")
        return
    require(not OUTPUT.exists(), f"Refusing overwrite: {OUTPUT}")
    predictions = predict_val(val, record["model"], device)
    STAGE.mkdir(parents=True, exist_ok=False)
    run_dir = STAGE / RUN
    run_dir.mkdir()
    metrics = p8a.summarize_run(RUN, val,
                               [int(r["target"]) for r in val], predictions, record, run_dir)
    with (run_dir / "predictions.csv").open(encoding="utf-8", newline="") as fp:
        new_rows = list(csv.DictReader(fp))
    bootstrap = paired_bootstrap(val, new_rows, previous)
    p8a.write_json(STAGE / "paired_bootstrap.json", bootstrap)
    hashes = {str(f.relative_to(STAGE)).replace("\\", "/"): sha256(f)
              for f in STAGE.rglob("*") if f.is_file()}
    manifest = {
        "phase": "P7_B1_P8a", "status": "B1_VAL_EVALUATION_COMPLETE",
        "scope": "VALIDATION_ONLY", "n_images": VAL_COUNT,
        "run_id": RUN, "file_sha256": hashes,
        "split_sha256": sha256(ROOT / "data/processed/split.csv"),
        "source_sha256": sha256(ROOT / "code/evaluate_b1_val.py"),
        "test_pixels_opened": False,
    }
    p8a.write_json(STAGE / "manifest.json", manifest)
    STAGE.rename(OUTPUT)
    print("B1_P8A_VAL_COMPLETE")
    print(json.dumps({"metrics": metrics, "comparisons": bootstrap["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
