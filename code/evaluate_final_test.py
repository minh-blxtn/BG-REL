"""P9: frozen, single-release image-level FINAL TEST for B0/B1/A3.

Usage (Windows venv):
    python code/evaluate_final_test.py --preflight
    python code/evaluate_final_test.py --run
    python code/evaluate_final_test.py --verify-output

Preflight checks metadata, model weights, source hashes, and VAL evidence; it
NEVER decodes TEST DICOMs. --run is the only mode that decodes TEST images.
No model selection, training, or parameter tuning takes place in this module.
Published results and earlier VAL experiments are never overwritten.
"""
from __future__ import annotations
from refactor_helpers import require

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from dataset import (  # noqa: E402
    MASTER_SHA, SPLIT_SHA, RSNAImageDataset, load_frozen, read_csv, sha256
)
from metrics import binary_metrics  # noqa: E402
from evaluate_p8a import verify_run  # noqa: E402
from evaluate_b1_val import verify_b1  # noqa: E402

RUN_IDS = ("b0_dev_s42", "b1_boxpos_s42", "a3_l100_s42")
EXPECTED_CHECKPOINTS = {
    "b0_dev_s42": "f6bda8a74c5d72b948a9ee4ba35cf53c1c0176e7d15dbb457ecc78a4d79f2929",
    "b1_boxpos_s42": "f1b2ab19211d22dc4ebe53cb978d36e15fa436b58e16f389954f5d2da0e688f1",
    "a3_l100_s42": "7bd87ba3841f0b8c8287a859dc3c942027e6fb318b2d5122dbfe1321bca2743f",
}
EXPECTED_VAL_MANIFESTS = {
    "p8a_val": "717c17801a336e966cfb45625727c35b47d0c1d78fe2d67e6c78adf1bc426983",
    "b1_val": "3964ef8bfc8a6f89d7be97f560ca9943c10426b620620f1e6e570e062a946948",
}
EXPECTED_TEST = 2967
EXPECTED_POSITIVE = 710
EXPECTED_PATIENTS = 1224
SEED = 42
BATCH_SIZE = 4
N_BOOTSTRAPS = 2000
THRESHOLD = 0.5  # Reporting-only fixed decision threshold; no tuning on TEST.
OUTPUT = ROOT / "results" / "evaluation" / "final_test"
STAGE = OUTPUT.with_name(".final_test_stage")
FIELDS = ("image_id", "patient_id", "view_position", "target", "logit",
          "probability", "predicted_at_0p5")




def read_json(path: Path) -> dict:
    require(path.is_file(), f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    require(not path.exists(), f"Refusing overwrite: {path}")
    with path.open("x", encoding="utf-8") as fp:
        json.dump(value, fp, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        fp.write("\n")


def write_csv(path: Path, fields: tuple[str, ...], rows) -> None:
    require(not path.exists(), f"Refusing overwrite: {path}")
    with path.open("x", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_validation_evidence() -> dict:
    """Freeze both completed image-level VAL experiments, without image I/O."""
    proof = {}
    base = ROOT / "results" / "evaluation"
    for name, expected_sha in EXPECTED_VAL_MANIFESTS.items():
        folder = base / name
        path = folder / "manifest.json"
        require(sha256(path) == expected_sha,
                f"Frozen VAL manifest changed: {name}")
        metadata = read_json(path)
        expected_status = ("VAL_EVALUATION_COMPLETE" if name == "p8a_val"
                           else "B1_VAL_EVALUATION_COMPLETE")
        require(metadata.get("status") == expected_status
                and metadata.get("scope") == "VALIDATION_ONLY",
                f"VAL manifest status invalid: {name}")
        require(metadata.get("test_pixels_opened") is False,
                f"VAL TEST-seal flag changed: {name}")
        require(metadata.get("split_sha256") == SPLIT_SHA,
                f"VAL split changed: {name}")
        for relative, expected_hash in metadata["file_sha256"].items():
            file_path = folder / relative
            require(file_path.is_file() and sha256(file_path) == expected_hash,
                    f"VAL artifact hash changed: {file_path}")
        proof[name] = {"manifest_sha256": expected_sha,
                       "n_artifacts_verified": len(metadata["file_sha256"])}
    return proof


def frozen_test_metadata() -> list[dict[str, str]]:
    """Select TEST rows using frozen metadata only; no TEST pixels opened."""
    base = ROOT / "data" / "processed"
    require(sha256(base / "split.csv") == SPLIT_SHA,
            "Frozen split.csv SHA256 mismatch")
    require(sha256(base / "master_manifest.csv") == MASTER_SHA,
            "Frozen master_manifest.csv SHA256 mismatch")
    require((base / "split.sha256").read_text(encoding="utf-8").split()[0]
            == SPLIT_SHA, "Frozen split seal changed")
    # The established loader enforces label, hash and patient-disjoint audits
    # but deliberately returns only train/val rows. It does not decode pixels.
    train, val = load_frozen(ROOT)
    splits = read_csv(base / "split.csv")
    master = read_csv(base / "master_manifest.csv")
    require(len(splits) == len(master) == 29684, "Full cohort changed")
    by_id = {x["image_id"]: x for x in master}
    require(len(by_id) == len(master), "Duplicate master image IDs")
    test = []
    seen = set()
    test_patients = set()
    for row in splits:
        image_id = row["image_id"]
        require(image_id not in seen, "Duplicate split image IDs")
        seen.add(image_id)
        if row["split"] != "test":
            continue
        image = by_id.get(image_id)
        require(image is not None and image["patient_id"] == row["patient_id"],
                f"TEST metadata patient mismatch: {image_id}")
        require(image["target"] in ("0", "1") and
                (image["final_label"] == "Lung Opacity") == (image["target"] == "1"),
                f"TEST target/label mismatch: {image_id}")
        test_patients.add(image["patient_id"])
        test.append(image)
    require(len(seen) == len(splits), "Incomplete split IDs")
    require(len(test) == EXPECTED_TEST, "TEST image count changed")
    require(len(test_patients) == EXPECTED_PATIENTS, "TEST patient count changed")
    require(sum(x["target"] == "1" for x in test) == EXPECTED_POSITIVE,
            "TEST positive count changed")
    require(len({x["image_id"] for x in test}) == EXPECTED_TEST,
            "Duplicate TEST rows")
    require(not test_patients.intersection(x["patient_id"] for x in train),
            "TEST/TRAIN patient leakage")
    require(not test_patients.intersection(x["patient_id"] for x in val),
            "TEST/VAL patient leakage")
    return sorted(test, key=lambda r: r["image_id"])


def verify_models(device: torch.device) -> dict:
    """Use the same checkpoint verification as completed VAL experiments."""
    result = {}
    for run_id in RUN_IDS:
        if run_id == "b1_boxpos_s42":
            record = verify_b1(device)
        else:
            record = verify_run(run_id)
            record["model"] = record["model"].to(device).eval()
        require(record["config"].get("split_sha256") == SPLIT_SHA,
                f"Frozen model split mismatch: {run_id}")
        require(record["config"].get("seed") == SEED
                and record["config"].get("batch") == BATCH_SIZE,
                f"Frozen training identity mismatch: {run_id}")
        stored = ROOT / "results" / "runs" / run_id / "best.pt"
        require(sha256(stored) == EXPECTED_CHECKPOINTS[run_id],
                f"Unexpected frozen checkpoint: {run_id}")
        record["model"].eval()
        result[run_id] = record
    return result


def frozen_protocol(val_proof: dict, models: dict) -> dict:
    return {
        "phase": "P9_FINAL_CLASSIFICATION_TEST",
        "protocol_status": "FROZEN_BEFORE_TEST_PIXEL_ACCESS",
        "split_sha256": SPLIT_SHA,
        "master_sha256": MASTER_SHA,
        "validation_evidence": val_proof,
        "models": {run: {"checkpoint_sha256": EXPECTED_CHECKPOINTS[run],
                         "best_epoch": 3}
                   for run in RUN_IDS},
        "image_input": [3, 224, 224],
        "image_processing": "dataset.py RSNAImageDataset(train=False, seed=42, epoch=0)",
        "inference_batch": BATCH_SIZE,
        "inference_precision": "FP32",
        "model_selection": "Previously frozen best VAL-AUROC epoch 3 for all models",
        "threshold": THRESHOLD,
        "threshold_policy": "fixed_descriptive_only_no_test_tuning",
        "primary_metric": "AUROC",
        "secondary_metric": "non_interpolated_Average_Precision",
        "comparison_pairs": ["b1_boxpos_s42_minus_b0_dev_s42",
                             "a3_l100_s42_minus_b0_dev_s42",
                             "a3_l100_s42_minus_b1_boxpos_s42"],
        "statistics": {"method": "paired_patient_cluster_percentile_bootstrap",
                       "resamples": N_BOOTSTRAPS, "seed": SEED,
                       "ci": 0.95, "bootstrap_unit": "patient"},
        "test_count_expected": EXPECTED_TEST,
        "test_positive_expected": EXPECTED_POSITIVE,
        "test_patients_expected": EXPECTED_PATIENTS,
        "regional_test": "NOT_IN_THIS_PROTOCOL",
        "training_or_tuning_after_test": "PROHIBITED",
        "code_sha256": {name: sha256(ROOT / "code" / name)
                        for name in ("dataset.py", "metrics.py", "model.py",
                                     "evaluate_p8a.py", "evaluate_b1_val.py",
                                     "evaluate_final_test.py")},
    }


def predict(test: list[dict], models: dict, device: torch.device) -> dict:
    """First and only phase in which TEST DICOM pixels are decoded."""
    loader = DataLoader(RSNAImageDataset(ROOT, test, train=False, seed=SEED, epoch=0),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=(device.type == "cuda"))
    outputs: dict[str, list[float]] = {run_id: [] for run_id in RUN_IDS}
    processed = 0
    with torch.inference_mode():
        for image, target in loader:
            n = len(target)
            require(tuple(image.shape[1:]) == (3, 224, 224),
                    "TEST image preprocessing changed")
            require([int(x) for x in target.tolist()] ==
                    [int(row["target"]) for row in test[processed:processed+n]],
                    "TEST sample order/target mismatch")
            image = image.to(device, non_blocking=True)
            for run_id in RUN_IDS:
                z = models[run_id]["model"](image).float().cpu()
                require(tuple(z.shape) == tuple(target.shape) and
                        bool(torch.isfinite(z).all()),
                        f"Nonfinite/invalid TEST model output: {run_id}")
                outputs[run_id].extend(z.tolist())
            processed += n
            if processed % 400 < n or processed == len(test):
                print(f"FINAL TEST {processed}/{len(test)}", flush=True)
    require(processed == EXPECTED_TEST and
            all(len(values) == EXPECTED_TEST for values in outputs.values()),
            "Incomplete TEST prediction output")
    return outputs


def summaries(test: list[dict], logits: dict, stage: Path, models: dict) -> dict:
    labels = np.array([int(r["target"]) for r in test], dtype=np.int64)
    metrics = {}
    for run_id in RUN_IDS:
        values = np.asarray(logits[run_id], dtype=np.float64)
        probabilities = 1. / (1. + np.exp(-np.clip(values, -80, 80)))
        auc, ap = binary_metrics(labels, values)
        predictions = (probabilities >= THRESHOLD).astype(np.int64)
        tn, fp, fn, tp = [int(n) for n in confusion_matrix(
            labels, predictions, labels=[0, 1]).ravel()]
        require(tn + fp + fn + tp == EXPECTED_TEST, "Confusion matrix size mismatch")
        run_folder = stage / run_id
        run_folder.mkdir()
        prediction_rows = ({
            "image_id": x["image_id"], "patient_id": x["patient_id"],
            "view_position": x["view_position"], "target": int(labels[i]),
            "logit": repr(float(values[i])), "probability": repr(float(probabilities[i])),
            "predicted_at_0p5": int(predictions[i]),
        } for i, x in enumerate(test))
        write_csv(run_folder / "predictions.csv", FIELDS, prediction_rows)
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        write_csv(run_folder / "roc.csv", ("fpr", "tpr", "threshold"),
                  ({"fpr": float(a), "tpr": float(b),
                    "threshold": float(c) if math.isfinite(float(c)) else ""}
                   for a, b, c in zip(fpr, tpr, thresholds)))
        precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities)
        write_csv(run_folder / "pr.csv", ("precision", "recall", "threshold"),
                  ({"precision": float(precision[i]), "recall": float(recall[i]),
                    "threshold": (float(pr_thresholds[i])
                                  if i < len(pr_thresholds) else "")}
                   for i in range(len(precision))))
        metrics[run_id] = {
            "run_id": run_id, "partition": "test", "n_images": EXPECTED_TEST,
            "n_patients": EXPECTED_PATIENTS,
            "n_positive": EXPECTED_POSITIVE,
            "n_negative": EXPECTED_TEST - EXPECTED_POSITIVE,
            "auroc": float(auc), "average_precision": float(ap),
            "threshold": THRESHOLD,
            "threshold_policy": "fixed_descriptive_only_not_optimized",
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity_at_0p5": tp / (tp + fn),
            "specificity_at_0p5": tn / (tn + fp),
            "checkpoint_sha256": EXPECTED_CHECKPOINTS[run_id],
            "best_epoch": 3,
            "test_pixels_opened": True,
            "no_test_training_or_tuning": True,
        }
        write_json(run_folder / "metrics.json", metrics[run_id])
    return metrics


def patient_bootstrap(test: list[dict], logits: dict) -> dict:
    labels = np.asarray([int(r["target"]) for r in test], dtype=np.int64)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(test):
        groups[item["patient_id"]].append(i)
    ids = sorted(groups)
    group_arrays = [np.asarray(groups[k], dtype=int) for k in ids]
    point = {name: dict(zip(("auroc", "ap"), binary_metrics(labels, z)))
             for name, z in logits.items()}
    rng = np.random.default_rng(SEED)
    values = {name: {metric: [] for metric in ("auroc", "ap")}
              for name in RUN_IDS}
    for iteration in range(N_BOOTSTRAPS):
        sampled = rng.integers(0, len(group_arrays), size=len(group_arrays))
        index = np.concatenate([group_arrays[j] for j in sampled])
        y = labels[index]
        require(np.unique(y).size == 2, "Bootstrap resample has one class")
        for run_id in RUN_IDS:
            auroc, ap = binary_metrics(y, np.asarray(logits[run_id])[index])
            values[run_id]["auroc"].append(auroc)
            values[run_id]["ap"].append(ap)
    intervals = {
        name: {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
               for m, v in model_values.items()}
        for name, model_values in values.items()
    }
    comparisons = {}
    for left, right in (("b1_boxpos_s42", "b0_dev_s42"),
                        ("a3_l100_s42", "b0_dev_s42"),
                        ("a3_l100_s42", "b1_boxpos_s42")):
        metrics = {}
        for metric in ("auroc", "ap"):
            differences = (np.asarray(values[left][metric]) -
                           np.asarray(values[right][metric]))
            metrics[metric] = {
                "delta_left_minus_right": float(point[left][metric] - point[right][metric]),
                "ci95": [float(np.percentile(differences, 2.5)),
                         float(np.percentile(differences, 97.5))],
            }
        comparisons[f"{left}_minus_{right}"] = metrics
    return {
        "partition": "test", "n_images": EXPECTED_TEST,
        "n_patients": len(ids), "n_resamples": N_BOOTSTRAPS,
        "method": "paired_patient_cluster_percentile_bootstrap",
        "seed": SEED, "point_estimates": point,
        "model_ci95": intervals, "comparisons": comparisons,
        "test_pixels_opened": True,
    }


def verify_published() -> None:
    require(OUTPUT.is_dir(), f"Final TEST output missing: {OUTPUT}")
    manifest = read_json(OUTPUT / "manifest.json")
    require(manifest.get("status") == "FINAL_TEST_COMPLETE"
            and manifest.get("scope") == "HELD_OUT_TEST"
            and manifest.get("test_pixels_opened") is True,
            "TEST manifest incomplete")
    for rel, expected in manifest.get("file_sha256", {}).items():
        target = OUTPUT / rel
        require(target.is_file() and sha256(target) == expected,
                f"Published file hash mismatch: {target}")
    require(len(manifest["file_sha256"]) >= 15,
            "Incomplete published output")
    print("FINAL_TEST_OUTPUT_HASH_PASS", "test_images_decoded_for_verification=0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--verify-output", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.verify_output:
        verify_published()
        return
    require(not OUTPUT.exists() and not STAGE.exists(),
            "FINAL TEST output or unfinished staging exists; refuse overwrite/rerun")
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    device = torch.device(args.device)
    val_proof = verify_validation_evidence()
    test = frozen_test_metadata()
    models = verify_models(device)
    protocol = frozen_protocol(val_proof, models)
    if args.preflight:
        print("FINAL_TEST_PREFLIGHT_PASS")
        print(json.dumps({
            "test_images_metadata": len(test),
            "test_patients_metadata": len({r["patient_id"] for r in test}),
            "test_image_pixels_opened": False,
            "runs": list(RUN_IDS),
            "checkpoint_hashes": EXPECTED_CHECKPOINTS,
            "val_evidence": val_proof,
            "staging_or_results_created": False,
        }, indent=2))
        return
    # Persist a protocol snapshot before any TEST image pixels are decoded.
    STAGE.mkdir(parents=True, exist_ok=False)
    write_json(STAGE / "protocol_freeze.json", protocol)
    write_json(STAGE / "test_access_started.json", {
        "status": "TEST_UNSEAL_IN_PROGRESS", "scope": "HELD_OUT_TEST",
        "expected_image_count": EXPECTED_TEST,
        "test_image_pixels_will_be_opened": True,
        "no_tuning_or_training_allowed": True,
    })
    predictions = predict(test, models, device)
    metrics = summaries(test, predictions, STAGE, models)
    bootstrap = patient_bootstrap(test, predictions)
    write_json(STAGE / "paired_bootstrap.json", bootstrap)
    comparisons = {
        "partition": "test", "runs": list(RUN_IDS),
        "n_common_images": EXPECTED_TEST,
        "patient_disjoint_from_train_and_val": True,
        "fixed_threshold": THRESHOLD,
        "metrics": {run_id: {"auroc": m["auroc"],
                              "average_precision": m["average_precision"]}
                    for run_id, m in metrics.items()},
        "paired_comparisons": bootstrap["comparisons"],
        "no_post_test_model_selection": True,
    }
    write_json(STAGE / "comparison.json", comparisons)
    hashes = {
        str(p.relative_to(STAGE)).replace("\\", "/"): sha256(p)
        for p in sorted(STAGE.rglob("*")) if p.is_file()
    }
    write_json(STAGE / "manifest.json", {
        "phase": "P9", "status": "FINAL_TEST_COMPLETE",
        "scope": "HELD_OUT_TEST", "n_images": EXPECTED_TEST,
        "runs": list(RUN_IDS),
        "split_sha256": SPLIT_SHA,
        "file_sha256": hashes,
        "checkpoint_sha256": EXPECTED_CHECKPOINTS,
        "test_pixels_opened": True,
        "source_sha256": sha256(ROOT / "code/evaluate_final_test.py"),
        "no_test_training_or_tuning": True,
    })
    STAGE.rename(OUTPUT)
    print("FINAL_TEST_COMPLETE")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
