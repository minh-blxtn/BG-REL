
"""P8a: frozen-validation image-level evaluation for B0 and A3.

This module never decodes TEST images, trains models, changes thresholds,
or overwrites existing results. All outputs are staged before publication.
"""

from __future__ import annotations
from refactor_helpers import require

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from dataset import RSNAImageDataset, load_frozen, sha256
from metrics import binary_metrics
from model import MobileNetV2OS16

RUN_IDS = ("b0_dev_s42", "a3_l100_s42")
EXPECTED_VAL = 2969
OUTPUT_ROOT = ROOT / "results" / "evaluation" / "p8a_val"
STAGE_ROOT = OUTPUT_ROOT.with_name(".p8a_val_stage")

PREDICTION_COLUMNS = (
    "image_id",
    "patient_id",
    "view_position",
    "target",
    "logit",
    "probability",
    "predicted_at_0p5",
)

CRITICAL_SOURCES = ("dataset.py", "model.py", "metrics.py")




def read_json(path: Path) -> dict:
    require(path.is_file(), f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    require(not path.exists(), f"Refusing overwrite: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            data, handle, indent=2,
            ensure_ascii=False, sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def write_csv(path: Path, fields, records) -> None:
    require(not path.exists(), f"Refusing overwrite: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def check_frozen_data():
    """Read frozen metadata and return TRAIN/VAL only."""
    train, val = load_frozen(ROOT)

    require(len(train) == 23748, "Unexpected TRAIN count")
    require(len(val) == EXPECTED_VAL, "Unexpected VAL count")

    train_patients = {row["patient_id"] for row in train}
    val_patients = {row["patient_id"] for row in val}

    require(
        not train_patients.intersection(val_patients),
        "TRAIN/VAL patient leakage",
    )

    val_ids = [row["image_id"] for row in val]
    require(
        len(val_ids) == len(set(val_ids)),
        "Duplicate VAL image IDs",
    )

    require(
        all(row["target"] in ("0", "1") for row in val),
        "Unexpected VAL target",
    )

    return val


def verify_run(run_id: str) -> dict:
    folder = ROOT / "results" / "runs" / run_id
    config = read_json(folder / "config.json")
    summary = read_json(folder / "summary.json")

    expected_status = (
        "FINISHED"
        if run_id == "b0_dev_s42"
        else "FINISHED_P6_DEVELOPMENT_PILOT"
    )

    require(
        summary.get("status") == expected_status,
        f"{run_id}: incomplete run",
    )
    require(
        summary.get("best_epoch") == 3,
        f"{run_id}: unexpected best epoch",
    )
    require(
        summary.get("test_images_opened") is False,
        f"{run_id}: TEST seal declaration changed",
    )

    best_path = folder / "best.pt"
    metrics_path = folder / "metrics.csv"

    require(
        sha256(best_path) == summary["checkpoint_sha256"],
        f"{run_id}: checkpoint hash mismatch",
    )
    require(
        sha256(metrics_path) == summary["metrics_sha256"],
        f"{run_id}: metrics hash mismatch",
    )

    # The active train_baseline.py is a T2-A experiment.
    # Verify only the shared sources used to reconstruct B0/A3.
    recorded_sources = summary["source_sha256"]

    for name in CRITICAL_SOURCES:
        require(
            name in recorded_sources,
            f"{run_id}: missing source provenance for {name}",
        )
        require(
            sha256(ROOT / "code" / name) == recorded_sources[name],
            f"{run_id}: source changed: {name}",
        )

    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True,
    )

    require(
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("model"), dict),
        f"{run_id}: invalid checkpoint schema",
    )
    require(
        checkpoint.get("epoch") == summary["best_epoch"],
        f"{run_id}: epoch mismatch",
    )
    require(
        checkpoint.get("config") == config,
        f"{run_id}: checkpoint/config mismatch",
    )

    require(
        math.isclose(
            float(checkpoint["val_auc"]),
            float(summary["best_val_auroc"]),
            abs_tol=1e-8,
            rel_tol=0.0,
        ),
        f"{run_id}: saved AUROC mismatch",
    )

    model = MobileNetV2OS16(pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    with torch.inference_mode():
        smoke = model(torch.zeros(1, 3, 224, 224))

    require(
        tuple(smoke.shape) == (1,)
        and bool(torch.isfinite(smoke).all()),
        f"{run_id}: model inference smoke failed",
    )

    return {
        "model": model,
        "config": config,
        "summary": summary,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(best_path),
    }


def prepare_models(device: torch.device) -> dict:
    models = {}

    for run_id in RUN_IDS:
        record = verify_run(run_id)
        record["model"] = record["model"].to(device).eval()
        models[run_id] = record

    return models


def evaluate(val: list[dict], models: dict, device: torch.device):
    dataset = RSNAImageDataset(
        ROOT, val, train=False, seed=42, epoch=0
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    logits = {run_id: [] for run_id in RUN_IDS}
    observed_targets = []
    processed = 0

    with torch.inference_mode():
        for batch_number, (images, targets) in enumerate(loader, 1):
            require(
                tuple(images.shape[1:]) == (3, 224, 224),
                "Unexpected input shape",
            )

            batch_size = len(targets)
            expected = np.array(
                [
                    int(row["target"])
                    for row in val[processed:processed + batch_size]
                ],
                dtype=np.int64,
            )
            actual = targets.numpy().astype(np.int64)

            require(
                np.array_equal(expected, actual),
                "VAL sample order or target mismatch",
            )

            images = images.to(device, non_blocking=True)

            # FP32; matches B0/A3 validation recipes.
            for run_id in RUN_IDS:
                model = models[run_id]["model"]
                output = model(images).float().cpu()

                require(
                    output.shape == targets.shape,
                    f"{run_id}: logit shape mismatch",
                )
                require(
                    bool(torch.isfinite(output).all()),
                    f"{run_id}: non-finite logits",
                )

                logits[run_id].extend(output.tolist())

            observed_targets.extend(actual.tolist())
            processed += batch_size

            if batch_number % 100 == 0:
                print(f"P8a VAL {processed}/{len(val)}", flush=True)

    require(
        processed == EXPECTED_VAL,
        "Incomplete VALIDATION inference",
    )

    return observed_targets, logits


def summarize_run(run_id, val, targets, logits, record, folder):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)

    require(
        len(z) == EXPECTED_VAL and len(y) == EXPECTED_VAL,
        "Incomplete output arrays",
    )

    probability = 1.0 / (1.0 + np.exp(-np.clip(z, -80, 80)))
    auc, ap = binary_metrics(y, z)

    saved = record["checkpoint"]
    require(
        abs(auc - float(saved["val_auc"])) < 1e-6,
        f"{run_id}: AUROC does not reproduce best checkpoint",
    )
    require(
        abs(ap - float(saved["val_ap"])) < 1e-6,
        f"{run_id}: AP does not reproduce best checkpoint",
    )

    labels = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        y, labels, labels=[0, 1]
    ).ravel()

    prediction_rows = []

    for index, row in enumerate(val):
        prediction_rows.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "view_position": row["view_position"],
            "target": int(y[index]),
            "logit": repr(float(z[index])),
            "probability": repr(float(probability[index])),
            "predicted_at_0p5": int(labels[index]),
        })

    write_csv(
        folder / "predictions.csv",
        PREDICTION_COLUMNS,
        prediction_rows,
    )

    fpr, tpr, roc_thresholds = roc_curve(y, probability)
    write_csv(
        folder / "roc.csv",
        ("fpr", "tpr", "threshold"),
        (
            {
                "fpr": float(a),
                "tpr": float(b),
                "threshold": (
                    float(c) if math.isfinite(float(c)) else ""
                ),
            }
            for a, b, c in zip(fpr, tpr, roc_thresholds)
        ),
    )

    precision, recall, pr_thresholds = precision_recall_curve(
        y, probability
    )
    write_csv(
        folder / "pr.csv",
        ("precision", "recall", "threshold"),
        (
            {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "threshold": (
                    float(pr_thresholds[i])
                    if i < len(pr_thresholds) else ""
                ),
            }
            for i in range(len(precision))
        ),
    )

    metrics = {
        "run_id": run_id,
        "partition": "val",
        "n_images": len(y),
        "n_patients": len({r["patient_id"] for r in val}),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "auroc": float(auc),
        "average_precision": float(ap),
        "threshold": 0.5,
        "threshold_policy": "descriptive_only_not_optimized",
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity_at_0p5": float(tp / (tp + fn)),
        "specificity_at_0p5": float(tn / (tn + fp)),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "best_epoch": int(saved["epoch"]),
        "checkpoint_auroc_reproduced": True,
        "checkpoint_ap_reproduced": True,
        "test_pixels_opened": False,
    }

    write_json(folder / "metrics.json", metrics)
    return metrics


def run(device: torch.device) -> None:
    require(
        not OUTPUT_ROOT.exists(),
        f"Evaluation output already exists: {OUTPUT_ROOT}",
    )
    require(
        not STAGE_ROOT.exists(),
        f"Unresolved staging directory: {STAGE_ROOT}",
    )

    val = check_frozen_data()
    models = prepare_models(device)

    # All input checks pass before creating output folders.
    STAGE_ROOT.mkdir(parents=True, exist_ok=False)

    targets, logits = evaluate(val, models, device)
    results = {}

    for run_id in RUN_IDS:
        folder = STAGE_ROOT / run_id
        folder.mkdir()

        results[run_id] = summarize_run(
            run_id,
            val,
            targets,
            logits[run_id],
            models[run_id],
            folder,
        )

    b0 = results["b0_dev_s42"]
    a3 = results["a3_l100_s42"]

    comparison = {
        "partition": "val",
        "runs": list(RUN_IDS),
        "n_common_images": EXPECTED_VAL,
        "matched_image_order": True,
        "delta_auroc_a3_minus_b0": (
            a3["auroc"] - b0["auroc"]
        ),
        "delta_ap_a3_minus_b0": (
            a3["average_precision"] - b0["average_precision"]
        ),
        "note": (
            "Development comparison only. "
            "No statistical superiority or TEST claim."
        ),
    }

    write_json(STAGE_ROOT / "comparison.json", comparison)

    file_hashes = {
        str(path.relative_to(STAGE_ROOT)).replace("\\", "/"): sha256(path)
        for path in sorted(STAGE_ROOT.rglob("*"))
        if path.is_file()
    }

    manifest = {
        "phase": "P8a",
        "status": "VAL_EVALUATION_COMPLETE",
        "split_sha256": sha256(
            ROOT / "data" / "processed" / "split.csv"
        ),
        "source_sha256": sha256(
            ROOT / "code" / "evaluate_p8a.py"
        ),
        "file_sha256": file_hashes,
        "test_pixels_opened": False,
        "scope": "VALIDATION_ONLY",
    }

    write_json(STAGE_ROOT / "manifest.json", manifest)

    # Atomic publication. Existing results are never replaced.
    STAGE_ROOT.rename(OUTPUT_ROOT)

    print("P8A_VAL_COMPLETE")
    print(json.dumps(comparison, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument('--bootstrap-preflight', action='store_true')
    mode.add_argument('--bootstrap-run', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda"
    )
    args = parser.parse_args()

    if args.bootstrap_preflight or args.bootstrap_run:
        bootstrap_check(args.bootstrap >= 100, 'Bootstrap count too small')
        bootstrap_check(args.seed >= 0, 'Invalid random seed')
        if args.bootstrap_preflight:
            y, _, _, groups, _ = bootstrap_load_predictions()
            print('P8A_BOOTSTRAP_PREFLIGHT_PASS')
            print('Images:', len(y))
            print('Patients:', len(groups))
            print('Output exists:', BOOTSTRAP_OUTPUT.exists())
        else:
            bootstrap_run(args.bootstrap, args.seed)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device(args.device)

    if args.preflight:
        val = check_frozen_data()
        records = prepare_models(device)
        require(
            len(val) == EXPECTED_VAL,
            "Unexpected validation count",
        )
        print("P8A_PREFLIGHT_PASS")
        print("Validation images:", len(val))
        print("Verified runs:", ", ".join(records))
        print("TEST images decoded: 0")
    else:
        run(device)



# --- Paired patient bootstrap (formerly p8a_bootstrap.py) ---
BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BOOTSTRAP_ROOT / 'code'))

BOOTSTRAP_BASE = BOOTSTRAP_ROOT / 'results' / 'evaluation' / 'p8a_val'

BOOTSTRAP_RUNS = ('b0_dev_s42', 'a3_l100_s42')

BOOTSTRAP_OUTPUT = BOOTSTRAP_BASE / 'paired_patient_bootstrap.json'

def bootstrap_check(condition, message):
    if not condition:
        raise RuntimeError(message)

def bootstrap_load_predictions():
    manifest = json.loads((BOOTSTRAP_BASE / 'manifest.json').read_text(encoding='utf-8'))
    bootstrap_check(manifest.get('status') == 'VAL_EVALUATION_COMPLETE' and manifest.get('scope') == 'VALIDATION_ONLY', 'P8a evaluation manifest invalid')
    records = []
    metrics = []
    for run_id in BOOTSTRAP_RUNS:
        relative = f'{run_id}/predictions.csv'
        path = BOOTSTRAP_BASE / relative
        expected_hash = manifest['file_sha256'].get(relative)
        bootstrap_check(expected_hash and sha256(path) == expected_hash, f'Prediction integrity failed: {run_id}')
        with path.open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        metric_path = BOOTSTRAP_BASE / run_id / 'metrics.json'
        metric_hash = manifest['file_sha256'].get(f'{run_id}/metrics.json')
        bootstrap_check(metric_hash and sha256(metric_path) == metric_hash, f'Metrics integrity failed: {run_id}')
        metric = json.loads(metric_path.read_text(encoding='utf-8'))
        bootstrap_check(len(rows) == 2969, 'Incomplete prediction cohort')
        bootstrap_check(metric['n_images'] == 2969, 'Metrics count mismatch')
        bootstrap_check(len({r['image_id'] for r in rows}) == 2969, 'Duplicate image ID')
        records.append(rows)
        metrics.append(metric)
    b0, a3 = records
    for x, y in zip(b0, a3):
        bootstrap_check((x['image_id'], x['patient_id'], x['target'], x['view_position']) == (y['image_id'], y['patient_id'], y['target'], y['view_position']), 'Paired image identity mismatch')
    targets = np.asarray([int(row['target']) for row in b0], dtype=np.int64)
    bootstrap_check(set(targets.tolist()) == {0, 1}, 'Only one class')
    z0 = np.asarray([float(row['logit']) for row in b0], dtype=np.float64)
    z1 = np.asarray([float(row['logit']) for row in a3], dtype=np.float64)
    bootstrap_check(np.isfinite(z0).all() and np.isfinite(z1).all(), 'Invalid logit')
    for z, metric, run_id in zip((z0, z1), metrics, BOOTSTRAP_RUNS):
        auc, ap = binary_metrics(targets, z)
        bootstrap_check(math.isclose(auc, metric['auroc'], abs_tol=1e-10, rel_tol=0), f'{run_id}: AUROC mismatch')
        bootstrap_check(math.isclose(ap, metric['average_precision'], abs_tol=1e-10, rel_tol=0), f'{run_id}: AP mismatch')
    patients = defaultdict(list)
    for i, row in enumerate(b0):
        patients[row['patient_id']].append(i)
    groups = [np.asarray(patients[pid], dtype=np.int64) for pid in sorted(patients)]
    return (targets, z0, z1, groups, manifest)

def bootstrap_interval(values):
    a = np.asarray(values, dtype=np.float64)
    return {'lower_95': float(np.percentile(a, 2.5)), 'upper_95': float(np.percentile(a, 97.5))}

def bootstrap_run(n_bootstrap, seed):
    bootstrap_check(not BOOTSTRAP_OUTPUT.exists(), f'Refusing to overwrite existing result: {BOOTSTRAP_OUTPUT}')
    y, z0, z1, groups, manifest = bootstrap_load_predictions()
    point0 = binary_metrics(y, z0)
    point1 = binary_metrics(y, z1)
    rng = np.random.default_rng(seed)
    n_patients = len(groups)
    samples = {'b0_auroc': [], 'a3_auroc': [], 'b0_ap': [], 'a3_ap': [], 'delta_auroc': [], 'delta_ap': []}
    invalid = 0
    attempts = 0
    while len(samples['delta_auroc']) < n_bootstrap:
        attempts += 1
        bootstrap_check(attempts <= n_bootstrap * 2, 'Too many invalid resamples')
        chosen = rng.integers(low=0, high=n_patients, size=n_patients)
        indices = np.concatenate([groups[int(i)] for i in chosen])
        labels = y[indices]
        if len(np.unique(labels)) != 2:
            invalid += 1
            continue
        b0_auc, b0_ap = binary_metrics(labels, z0[indices])
        a3_auc, a3_ap = binary_metrics(labels, z1[indices])
        samples['b0_auroc'].append(b0_auc)
        samples['a3_auroc'].append(a3_auc)
        samples['b0_ap'].append(b0_ap)
        samples['a3_ap'].append(a3_ap)
        samples['delta_auroc'].append(a3_auc - b0_auc)
        samples['delta_ap'].append(a3_ap - b0_ap)
        done = len(samples['delta_auroc'])
        if done % 250 == 0:
            print(f'Bootstrap {done}/{n_bootstrap}', flush=True)
    result = {'phase': 'P8a', 'scope': 'VALIDATION_ONLY', 'status': 'PAIRED_PATIENT_BOOTSTRAP_COMPLETE', 'method': 'paired_patient_cluster_percentile_bootstrap', 'confidence_level': 0.95, 'n_images': int(len(y)), 'n_patients': n_patients, 'n_bootstrap': n_bootstrap, 'seed': seed, 'invalid_resamples': invalid, 'auroc': {'b0': {'point': float(point0[0]), **bootstrap_interval(samples['b0_auroc'])}, 'a3': {'point': float(point1[0]), **bootstrap_interval(samples['a3_auroc'])}, 'delta_a3_minus_b0': {'point': float(point1[0] - point0[0]), **bootstrap_interval(samples['delta_auroc'])}}, 'average_precision': {'b0': {'point': float(point0[1]), **bootstrap_interval(samples['b0_ap'])}, 'a3': {'point': float(point1[1]), **bootstrap_interval(samples['a3_ap'])}, 'delta_a3_minus_b0': {'point': float(point1[1] - point0[1]), **bootstrap_interval(samples['delta_ap'])}}, 'input_manifest_sha256': sha256(BOOTSTRAP_BASE / 'manifest.json'), 'limitations': ['Validation-only exploratory inference', 'Percentile CI is not a non-inferiority test', 'No inference about regional lesion reliance', 'Paired seed-42 models only; no between-seed uncertainty'], 'test_pixels_opened': False}
    with BOOTSTRAP_OUTPUT.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print('P8A_BOOTSTRAP_PASS')
    print(json.dumps({'n_patients': n_patients, 'delta_auroc': result['auroc']['delta_a3_minus_b0'], 'delta_ap': result['average_precision']['delta_a3_minus_b0'], 'output': str(BOOTSTRAP_OUTPUT)}, indent=2))

def bootstrap_main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    bootstrap_check(args.bootstrap >= 100, 'Bootstrap count too small')
    bootstrap_check(args.seed >= 0, 'Invalid random seed')
    if args.preflight:
        y, _, _, groups, _ = bootstrap_load_predictions()
        print('P8A_BOOTSTRAP_PREFLIGHT_PASS')
        print('Images:', len(y))
        print('Patients:', len(groups))
        print('Output exists:', BOOTSTRAP_OUTPUT.exists())
    else:
        bootstrap_run(args.bootstrap, args.seed)

if __name__ == "__main__":
    main()
