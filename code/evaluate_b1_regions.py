"""B1-only P8b regional evaluation against frozen B0/A3 lesion/control groups.

The original evaluate_regions.py and results/evaluation/regions_val are read-only.
Recomputes the same deterministic P8b control selection (no saved control
coordinates exist in regions.csv) and verifies exact group identity/K/tier
parity against existing A3 P8b evidence before publishing B1 results.

    python code/evaluate_b1_regions.py --preflight
    python code/evaluate_b1_regions.py --run

This can be computationally expensive because it decodes 711 positive VAL
radiographs and reruns PSPNet and intervention inference. No TEST access.
"""
from __future__ import annotations
from refactor_helpers import read_json

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import evaluate_regions as p8b
from dataset import decode_dicom, sha256
from sample_controls import rect_from_row, sample_lesion
from evaluate_b1_val import RUN, OUTPUT as B1_P8A, verify_b1

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/evaluation/regions_val"
OUTPUT = ROOT / "results/evaluation/b1_regions_val"
STAGE = ROOT / "results/evaluation/.b1_regions_val_stage"
SEED = 42
BOOTSTRAPS = 2000
METRICS = (
    "rem", "top1", "delta_l_blur", "delta_c_blur", "rie_blur",
    "delta_l_fill", "delta_c_fill", "rie_fill",
)


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)




def old_region_rows() -> tuple[dict, dict]:
    summary = read_json(ORIGINAL / "summary.json")
    require(summary["status"] == "REGIONAL_EVALUATION_COMPLETE", "Original P8b not complete")
    require(summary["inputs"]["regions_csv_sha256"] == sha256(ORIGINAL / "regions.csv"), "Original P8b CSV hash mismatch")
    require(summary["inputs"]["source_sha256"] == sha256(ROOT / "code/evaluate_regions.py"), "Original P8b script changed")
    require(summary.get("test_pixels_opened") is False, "Original P8b TEST seal invalid")
    rows = p8b.read_csv(ORIGINAL / "regions.csv")
    output = {}
    for r in rows:
        require(r["run_id"] in ("b0_dev_s42", "a3_l100_s42"), "Unexpected original P8b model")
        key = (r["image_id"], r["annotation_id"])
        output.setdefault(r["run_id"], {})[key] = r
    require(len(output["b0_dev_s42"]) == 1122 and len(output["a3_l100_s42"]) == 1122, "Original P8b coverage differs")
    return output, summary


def load_pspnet_compat(device: torch.device):
    """Load the frozen PSPNet without modifying the installed library or BG-REL.

    TorchXRayVision's legacy JF Healthcare VGG module imports ``model.utils``
    as a top-level package. BG-REL already imports its own ``code/model.py``
    under the name ``model``. Temporarily redirect ONLY that name while
    importing TorchXRayVision, then restore the exact original BG-REL module.
    The PSPNet constructor, weights, SHA checks and preprocessing still come
    from the unmodified frozen P8b ``load_pspnet`` implementation.
    """
    import importlib.machinery
    import importlib.metadata
    import sys
    import types

    existing = sys.modules.get("model")
    require(existing is not None and getattr(existing, "__file__", None),
            "Expected BG-REL classifier module in sys.modules")
    require(Path(existing.__file__).resolve() == (ROOT / "code/model.py").resolve(),
            "Unexpected module named model; refusing import workaround")

    dist = importlib.metadata.distribution("torchxrayvision")
    jf_dir = Path(dist.locate_file(
        "torchxrayvision/baseline_models/jfhealthcare/model"
    )).resolve()
    require((jf_dir / "utils.py").is_file(),
            "TorchXRayVision JF Healthcare utils.py not found")

    temporary_package = types.ModuleType("model")
    temporary_package.__file__ = str(jf_dir / "__init__.py")
    temporary_package.__package__ = "model"
    temporary_package.__path__ = [str(jf_dir)]
    package_spec = importlib.machinery.ModuleSpec("model", loader=None, is_package=True)
    package_spec.submodule_search_locations = [str(jf_dir)]
    temporary_package.__spec__ = package_spec

    sys.modules["model"] = temporary_package
    try:
        import torchxrayvision as xrv
        require(hasattr(xrv.baseline_models.chestx_det, "PSPNet"),
                "TorchXRayVision PSPNet API unavailable")
    finally:
        sys.modules["model"] = existing

    require(sys.modules.get("model") is existing,
            "BG-REL classifier module was not restored")
    # Unchanged original P8b loader validates PSPNet checkpoint SHA and targets.
    return p8b.load_pspnet(device)


def b1_val_predictions(val: list[dict]) -> dict:
    manifest = read_json(B1_P8A / "manifest.json")
    require(manifest["status"] == "B1_VAL_EVALUATION_COMPLETE", "B1 image evaluation not finished")
    require(manifest.get("test_pixels_opened") is False, "B1 TEST seal changed")
    relative = f"{RUN}/predictions.csv"
    p = B1_P8A / relative
    require(sha256(p) == manifest["file_sha256"][relative], "B1 prediction CSV hash mismatch")
    rows = p8b.read_csv(p)
    mapping = {r["image_id"]: r for r in rows}
    require(len(rows) == len(mapping) == 2969, "B1 predictions incomplete")
    require(set(mapping) == {r["image_id"] for r in val}, "B1 prediction VAL identity mismatch")
    return mapping


def compare_groups(current: list[dict], old: dict) -> None:
    actual = {(r["image_id"], r["annotation_id"]): r for r in current}
    require(len(actual) == len(current) == len(old), "Changed regional cohort")
    for identity, row in actual.items():
        require(identity in old, f"Missing original regional group: {identity}")
        ref = old[identity]
        for key in ("patient_id", "view_position", "k", "tier1", "tier2", "mask_status", "skip_reason"):
            require(str(row[key] if row[key] is not None else "") == str(ref[key]),
                    f"P8b group changed: {identity} {key}")
        require(abs(float(row["lesion_area"]) - float(ref["lesion_area"])) < 1e-6,
                f"P8b lesion area changed: {identity}")


def paired_statistics(new_rows: list[dict], previous: dict) -> dict:
    b1 = {(r["image_id"], r["annotation_id"]): r for r in new_rows if int(r["k"]) > 0}
    result = {}
    for reference in ("b0_dev_s42", "a3_l100_s42"):
        old = {k: r for k, r in previous[reference].items() if int(r["k"]) > 0}
        require(set(b1) == set(old) and len(b1) == 344, "B1 regional cohort differs from B0/A3")
        per_metric = {}
        for metric in METRICS:
            left = []
            right = []
            patient_values = defaultdict(list)
            for key in sorted(b1):
                value_new = float(b1[key][metric])
                value_old = float(old[key][metric])
                require(np.isfinite(value_new) and np.isfinite(value_old), f"Nonfinite regional metric {metric}")
                left.append(value_old)
                right.append(value_new)
                patient_values[b1[key]["patient_id"]].append(value_new - value_old)
            means = np.asarray([np.mean(patient_values[k]) for k in sorted(patient_values)], dtype=float)
            require(len(means) == 168, "Patient regional cohort differs")
            rng = np.random.default_rng(SEED)
            estimates = np.asarray([means[rng.integers(0, len(means), len(means))].mean()
                                    for _ in range(BOOTSTRAPS)])
            per_metric[metric] = {
                "n_lesions": len(b1), "n_patients": len(means),
                "reference_lesion_mean": float(np.mean(left)),
                "b1_lesion_mean": float(np.mean(right)),
                "lesion_mean_difference_b1_minus_reference": float(np.mean(np.asarray(right) - np.asarray(left))),
                "patient_mean_difference_b1_minus_reference": float(means.mean()),
                "patient_ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
            }
        result[f"{RUN}_minus_{reference}"] = per_metric
    return result


def run(device: torch.device) -> None:
    require(not OUTPUT.exists() and not STAGE.exists(), "B1 regional output/stage already exists")
    positive, boxes, _ = p8b.get_inputs()  # Original P8a evidence verified here.
    old, old_summary = old_region_rows()
    _, val = p8b.load_frozen(ROOT)
    predictions = b1_val_predictions(val)
    model = verify_b1(device)["model"]
    pspnet = load_pspnet_compat(device)
    rows = []
    counts = Counter()
    counts["mask_status"] = Counter()
    counts["skip_reason"] = Counter()
    eligible_images = 0

    with torch.inference_mode():
        for image_idx, row in enumerate(positive, start=1):
            sid = row["image_id"]
            raw = decode_dicom(ROOT, row)
            lungs, status = p8b.predict_lungs(pspnet, row, device)
            counts["mask_status"][status] += 1
            image_boxes = sorted(boxes[sid], key=lambda b: (int(b["box_number"]), b["annotation_id"]))
            lesions = [rect_from_row(b) for b in image_boxes]
            groups = []
            for annotation, lesion in zip(image_boxes, lesions):
                sample = sample_lesion(sid, annotation["annotation_id"], lesion, lesions, lungs)
                controls = [c["rect"] for c in sample["accepted"]]
                counts[sample["k"]] += 1
                if sample["k"] == 0:
                    counts["skip_reason"][sample["skip"] or "NO_MATCH"] += 1
                groups.append((annotation, lesion, controls, sample))
            if any(g[3]["k"] for g in groups):
                eligible_images += 1
            blurred = gaussian_filter(raw, sigma=p8b.SIGMA, mode="nearest", truncate=3.0)
            for annotation, lesion, controls, sample in groups:
                record = {
                    "image_id": sid, "patient_id": row["patient_id"],
                    "annotation_id": annotation["annotation_id"],
                    "view_position": row["view_position"], "run_id": RUN,
                    "lesion_area": float(lesion.area), "k": sample["k"],
                    "tier1": int(sample["tier_count"]["1"]),
                    "tier2": int(sample["tier_count"]["2"]),
                    "mask_status": status, "skip_reason": sample["skip"],
                }
                if sample["k"] == 0:
                    record.update({name: "" for name in p8b.FIELDS[11:]})
                    rows.append(record)
                    continue
                p0, scores = p8b.original_region_scores(model, raw, [lesion, *controls], device)
                require(abs(p0 - float(predictions[sid]["probability"])) < 1e-4,
                        f"P8a/B1 probability mismatch: {sid}")
                record.update({
                    "lesion_logit": float(scores[0]),
                    "control_mean_logit": float(scores[1:].mean()),
                    "rem": float(scores[0] - scores[1:].mean()),
                    "top1": int(scores[0] > scores[1:].max()),
                })
                record.update(p8b.intervention_metrics(model, raw, blurred, lesion, controls, p0, device))
                rows.append(record)
            if image_idx % 25 == 0 or image_idx == len(positive):
                print(f"B1 REGIONS {image_idx}/{len(positive)} matched={counts[1]+counts[2]}", flush=True)

    require(len(rows) == 1122 and sum(counts[i] for i in range(3)) == 1122, "B1 regional rows incomplete")
    require(eligible_images == old_summary["eligible_images"], "B1 matched image count differs")
    require({str(k): int(counts[k]) for k in range(3)} == old_summary["k_distribution"], "B1 K distribution differs")
    compare_groups(rows, old["a3_l100_s42"])
    comparisons = paired_statistics(rows, old)
    STAGE.mkdir(parents=True, exist_ok=False)
    path = STAGE / "regions.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=p8b.FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "phase": "P7_B1_P8b", "status": "B1_REGIONAL_EVALUATION_COMPLETE",
        "scope": "FROZEN_VALIDATION_POSITIVE_IMAGES", "n_positive_images": 711,
        "n_lesion_boxes": 1122, "eligible_images": eligible_images,
        "k_distribution": {str(k): int(counts[k]) for k in range(3)},
        "paired_statistics": comparisons,
        "inputs": {
            "b1_checkpoint_sha256": sha256(ROOT / "results/runs" / RUN / "best.pt"),
            "original_regions_csv_sha256": old_summary["inputs"]["regions_csv_sha256"],
            "b1_regions_csv_sha256": sha256(path),
            "source_sha256": sha256(ROOT / "code/evaluate_b1_regions.py"),
        },
        "test_pixels_opened": False,
    }
    with (STAGE / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    STAGE.rename(OUTPUT)
    print("B1_P8B_REGIONAL_COMPLETE")
    print(json.dumps({name: {m: s for m, s in metrics.items() if m in ("rem", "top1", "rie_blur", "rie_fill")}
                      for name, metrics in comparisons.items()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    device = torch.device(args.device)
    positive, _, _ = p8b.get_inputs()
    old, _ = old_region_rows()
    _, val = p8b.load_frozen(ROOT)
    b1_val_predictions(val)
    verify_b1(device)
    require(len(positive) == 711 and len(old["a3_l100_s42"]) == 1122, "Preflight cohort mismatch")
    require(not STAGE.exists(), "Unresolved B1 regional staging directory")
    if args.preflight:
        # Test the import path AND the frozen PSPNet constructor/weights here.
        # The original preflight did not import TorchXRayVision and missed this error.
        segmentation_model = load_pspnet_compat(device)
        del segmentation_model
        print("B1_REGIONAL_PREFLIGHT_PASS original_evaluation_preserved=True "
              "pspnet_import_and_weights=True test_pixels_opened=False")
        return
    run(device)


if __name__ == "__main__":
    main()
