"""BG-REL V3.2 P6 preparation: immutable recipe audit + 2-batch BF16 smoke.

This diagnostic deliberately has NO --train, --resume, checkpoint output, epoch
loop, validation-based model selection, or access to TEST pixels. It is not a
substitute for a full P6 trainer or permission to use P3 control regions.
"""
from __future__ import annotations
from refactor_helpers import verify_current_baseline_layout

import argparse
import copy
import gc
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import RSNAImageDataset, load_frozen
from model import MobileNetV2OS16, imagenet_weight_path
from rel_adapter import FrozenControlIndex, load_frozen_controls, loss_from_frozen_train_batch
from losses import LossParts
from t3_train_integration import (
    IdentifiedTrainDataset, check_b0_recipe, SEED, BATCH, LR, WEIGHT_DECAY,
    MAX_EPOCHS, CLIP_NORM,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "qa"
B0 = ROOT / "results" / "runs" / "b0_dev_s42"
PREFLIGHT_FILENAME = "p6_preflight_exposure_fix.json"
SMOKE_FILENAME = "p6_two_batch_cuda_exposure_fix.json"
SOURCE_NAMES = (
    "dataset.py", "model.py", "metrics.py", "regions.py", "losses.py",
    "rel_adapter.py", "t3_train_integration.py", "p6_preflight.py", "refactor_helpers.py",
)
# Module 05: first two pilot candidates; no candidate is selected here.
PILOT_LAMBDA_MAX_CANDIDATES = (0.10, 0.25)
K_MAX = 2
REVIEWED_SOURCE_SHA = {
    "metrics.py": "b35837f78580926d6f52be3e861227c375f604bc22e6063f58f549df99bdb67b",
    "regions.py": "32166f16739a5abee38461182339d4a15a466068dd93de50d5dd8b4c5883cd0c",
    "losses.py": "e585469170a1591a4deb5d16db9b22b22198932f7200fe3307ad0037df85bfb6",
    "rel_adapter.py": "ed76334f11c67e401eecb53c825d243ff4f2d7d20996c11bf4365cf7d999540f",
    "t3_train_integration.py": "170e41561810d4c54a3ad087890bf67da24c73aa4d2ba7cd9ba19aee108460be",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lambda_at_epoch(epoch: int, lambda_max: float) -> float:
    """Module 05: 20%, 60%, 100% warm-up in epochs 1, 2, >=3."""
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError("Epoch must be a positive integer")
    if not isinstance(lambda_max, (float, int)) or not math.isfinite(lambda_max) or lambda_max < 0:
        raise ValueError("lambda_max must be nonnegative and finite")
    return float(lambda_max) * (0.2 if epoch == 1 else 0.6 if epoch == 2 else 1.0)


def patient_unique_categorized_rows(
    rows: Sequence[Mapping[str, str]], index: FrozenControlIndex
) -> list[Mapping[str, str]]:
    """Select 8 TRAIN examples: two K0, two K1, two K2, two negatives.

    This deliberate enriched diagnostic batch is NOT representative of full
    P3 REL exposure and must never be used as a training/evaluation cohort.
    """
    selected: list[Mapping[str, str]] = []
    used_patients: set[str] = set()
    required = ("matched_K1", "matched_K2", "positive_K0", "negative") * 2
    for kind in required:
        choice = None
        for row in sorted(rows, key=lambda r: r["image_id"]):
            sid = row["image_id"]
            if row["patient_id"] in used_patients or sid not in index.train_targets:
                continue
            if int(row["target"]) != index.train_targets[sid]:
                raise ValueError("Frozen label discrepancy")
            groups = index.for_image(sid)
            has_k2 = any(len(g.control_xyxy) == 2 for g in groups)
            has_k1 = any(len(g.control_xyxy) == 1 for g in groups)
            present = (
                "negative" if row["target"] == "0" and not groups else
                "matched_K2" if has_k2 else
                "matched_K1" if has_k1 else
                "positive_K0" if row["target"] == "1" and not groups else "invalid"
            )
            if present == kind:
                choice = row
                break
        if choice is None:
            raise ValueError(f"Cannot find independent TRAIN patient for {kind}")
        selected.append(choice)
        used_patients.add(choice["patient_id"])
    if len(selected) != 8 or len(used_patients) != 8:
        raise AssertionError("Two-batch cohort must be 8 unique TRAIN patients")
    return selected


def validate_batch_identity(
    index: FrozenControlIndex, image_ids: Sequence[str], targets: Tensor
) -> dict:
    if len(image_ids) != BATCH or len(set(image_ids)) != BATCH or tuple(targets.shape) != (BATCH,):
        raise ValueError("Unexpected batch shape or duplicated image ID")
    kinds = []
    eligible_images = eligible_lesions = 0
    for image_id, target in zip(image_ids, targets.detach().cpu().tolist()):
        if image_id not in index.train_targets or int(target) != index.train_targets[image_id]:
            raise ValueError("Shuffle-induced target/image_id misalignment")
        groups = index.for_image(image_id)
        if int(target) == 0 and groups:
            raise ValueError("Negative with REL annotations")
        k = [len(g.control_xyxy) for g in groups]
        if any(n not in (1, 2) for n in k):
            raise ValueError("Unexpected frozen control count")
        eligible_images += int(bool(k))
        eligible_lesions += len(k)
        kinds.append("negative" if target == 0 else "matched" if k else "positive_K0")
    return {
        "image_ids": list(image_ids), "eligible_images": eligible_images,
        "eligible_lesions": eligible_lesions, "kinds": kinds,
    }


def summarize_exposure(records: Sequence[dict], index: FrozenControlIndex) -> dict:
    """Compute diagnostic batch exposure without extrapolating to all TRAIN."""
    if not records:
        raise ValueError("No batches")
    total = sum(len(r["image_ids"]) for r in records)
    active = sum(r["eligible_images"] > 0 for r in records)
    seen = [sid for r in records for sid in r["image_ids"]]
    if len(set(seen)) != total or any(sid not in index.train_targets for sid in seen):
        raise ValueError("Duplicate or non-TRAIN image across batches")
    eligible_ids = {sid for sid in seen if len(index.for_image(sid)) > 0}
    return {
        "batches": len(records), "images": total, "active_batches": active,
        "active_batch_rate": active / len(records),
        "eligible_unique_images": len(eligible_ids),
        "eligible_lesions_seen": sum(r["eligible_lesions"] for r in records),
        "enriched_not_population_estimate": True,
    }


def save_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preflight() -> tuple[FrozenControlIndex, list[Mapping[str, str]], dict, dict]:
    index = load_frozen_controls(ROOT)
    train, val = load_frozen(ROOT)
    if len(train) != 23748 or len(val) != 2969:
        raise ValueError("Unexpected frozen train/validation dimensions")
    config = json.loads((B0 / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((B0 / "summary.json").read_text(encoding="utf-8"))
    hashes = {name: file_sha256(ROOT / "code" / name) for name in SOURCE_NAMES}
    for name, expected in REVIEWED_SOURCE_SHA.items():
        if hashes[name] != expected:
            raise ValueError(f"Reviewed source changed: {name}")
    current_baselines = verify_current_baseline_layout(ROOT)
    # This remains a historical recipe/source check. R3 dataset drift must fail.
    # Never substitute a current source hash into the archived B0 configuration.
    check_b0_recipe(config, summary, dataset_sha=hashes["dataset.py"], model_sha=hashes["model.py"])
    weights = imagenet_weight_path()
    if not weights.is_file() or file_sha256(weights) != config["pretrained_sha256"]:
        raise ValueError("B0 ImageNet weight provenance mismatch")
    if file_sha256(B0 / "best.pt") != summary["checkpoint_sha256"]:
        raise ValueError("B0 best checkpoint provenance mismatch")
    selected = patient_unique_categorized_rows(train, index)
    if not all(r["image_id"] in index.train_targets for r in selected):
        raise ValueError("Selection includes non-TRAIN images")
    record = {
        "status": "P6_PREPARATION_PREFLIGHT_PASS", "scope": "TRAIN_METADATA_ONLY",
        "purpose": "two_batch_smoke_only_no_P6_training", "test_pixels_opened": False,
        "current_baseline_versions": current_baselines,
        "baseline_run_id": "b0_dev_s42", "baseline_best_sha256": summary["checkpoint_sha256"],
        "recipe": {"seed": SEED, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
                   "max_epochs": MAX_EPOCHS, "augmentation": config["augment_train"],
                   "amp_precision": "bf16", "grad_scaler_enabled": False,
                   "scheduler": config["scheduler"]},
        "rel_schedule": {"lambda_max_candidates": list(PILOT_LAMBDA_MAX_CANDIDATES),
                         "epoch_1_factor": .2, "epoch_2_factor": .6,
                         "epoch_3_onward_factor": 1.0, "k_max": K_MAX},
        "p3_anatomical_quality": "PENDING_INDEPENDENT_REVIEW",
        "p3_frozen": {"matched_lesions": index.eligible_lesions,
                      "controls": index.control_count, "k0_lesions": index.k0_lesions},
        "selected": [{"image_id": r["image_id"], "patient_id": r["patient_id"],
                      "target": int(r["target"])} for r in selected],
        "source_sha256": hashes,
        "b0_config_sha256": file_sha256(B0 / "config.json"),
        "b0_summary_sha256": file_sha256(B0 / "summary.json"),
        "split_sha256": config["split_sha256"],
        "master_sha256": config["master_sha256"],
        "training_started": False, "checkpoint_created": False,
    }
    return index, selected, config, record


def load_two_batches(root: Path, selected: Sequence[Mapping[str, str]]) -> list[tuple[Tensor, Tensor, list[str]]]:
    base = RSNAImageDataset(root, list(selected), train=True, seed=SEED, epoch=1)
    tagged = IdentifiedTrainDataset(base, selected)
    generator = torch.Generator().manual_seed(SEED + 1 * 1009 + 1)
    loader = DataLoader(tagged, batch_size=BATCH, shuffle=True, num_workers=0,
                        pin_memory=False, drop_last=False, generator=generator)
    result = [(x, y, list(ids)) for x, y, ids in loader]
    if len(result) != 2 or any(x.shape != (4, 3, 224, 224) for x, _, _ in result):
        raise RuntimeError("Two-batch smoke loader produced unexpected tensor shapes")
    return result


@dataclass
class DiagnosticResult:
    step_losses: list[dict]
    step_gradients: list[dict[str, Tensor]]
    final_state: dict[str, Tensor]
    head_changed: bool
    eligible_images_seen: int
    exposure: dict
    grad_ratio_head: float | None
    grad_cosine_head: float | None


def head_gradient_diagnostic(bce: Tensor, rel: Tensor, classifier: nn.Linear, lmbda: float) -> tuple[float, float]:
    """Measure ||lambda*g_rel||/||g_bce|| and cosine on shared classifier params."""
    variables = (classifier.weight, classifier.bias)
    g_bce = torch.autograd.grad(bce, variables, retain_graph=True, allow_unused=False)
    g_rel = torch.autograd.grad(rel, variables, retain_graph=True, allow_unused=False)
    v1 = torch.cat([t.detach().float().flatten() for t in g_bce])
    v2 = torch.cat([t.detach().float().flatten() for t in g_rel])
    n1 = v1.norm().item()
    n2 = v2.norm().item()
    return (float(lmbda * n2 / (n1 + 1e-12)),
            float(torch.dot(v1, v2).item() / (n1 * n2 + 1e-12)))



def verify_rel_exposure_contract(
    parts: LossParts, batch_info: Mapping[str, object], *, lambda_value: float
) -> None:
    """Distinguish frozen REL eligibility from loss activation at lambda=0.

    Frozen metadata can include matched lesion/control groups even if lambda=0.
    The adapter intentionally skips ROI evaluation in that branch, so
    LossParts.eligible_* must be zero while batch_info still records eligible
    groups. With a positive lambda, computed and frozen exposure must agree.
    """
    if not math.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("Invalid lambda for REL exposure validation")
    if lambda_value == 0.0:
        if parts.eligible_images != 0 or parts.eligible_lesions != 0:
            raise ValueError("Lambda-zero REL bypass unexpectedly computed ROI terms")
        if not bool(torch.isfinite(parts.rel)) or float(parts.rel.detach().item()) != 0.0:
            raise ValueError("Lambda-zero REL loss must remain exactly zero")
    else:
        if (parts.eligible_images != batch_info["eligible_images"] or
                parts.eligible_lesions != batch_info["eligible_lesions"]):
            raise ValueError("Positive-lambda REL exposure disagrees with frozen metadata")


def two_step_branch(
    model: MobileNetV2OS16,
    batches: Sequence[tuple[Tensor, Tensor, Sequence[str]]],
    index: FrozenControlIndex,
    *,
    mode: str,
) -> DiagnosticResult:
    """Exactly 2 optimizer steps with fresh optimizer, never persist state."""
    if mode not in {"B0", "A3_lambda0", "A3_warmup_lambda010"} or len(batches) != 2:
        raise ValueError("Unexpected diagnostic mode/batch count")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; no fallback permitted")
    model = model.to("cuda")
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=math.ceil(23748 / BATCH) * MAX_EPOCHS
    )
    original_head = model.classifier.weight.detach().clone()
    step_losses: list[dict] = []
    step_gradients: list[dict[str, Tensor]] = []
    batch_reports = []
    ratio = cosine = None
    for step_idx, (images, targets, ids) in enumerate(batches, 1):
        batch_info = validate_batch_identity(index, ids, targets)
        batch_reports.append(batch_info)
        images = images.to("cuda", non_blocking=True)
        targets = targets.to("cuda", non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "B0":
                logits = model(images)
                total = bce = F.binary_cross_entropy_with_logits(logits, targets)
                rel = bce.detach() * 0
            else:
                feature_maps = model.forward_features(images)
                lam = 0.0 if mode == "A3_lambda0" else lambda_at_epoch(1, .10)
                parts = loss_from_frozen_train_batch(
                    index, ids, targets, feature_maps, model.classifier,
                    lambda_value=lam
                )
                total, bce, rel = parts.total, parts.bce, parts.rel
                verify_rel_exposure_contract(parts, batch_info, lambda_value=lam)
        if not all(bool(torch.isfinite(z)) for z in (total, bce, rel)):
            raise FloatingPointError("Nonfinite two-batch objective")
        if mode == "A3_warmup_lambda010" and batch_info["eligible_images"] > 0 and ratio is None:
            ratio, cosine = head_gradient_diagnostic(bce, rel, model.classifier, lambda_at_epoch(1, .10))
            if not (math.isfinite(ratio) and math.isfinite(cosine)):
                raise FloatingPointError("Nonfinite REL head gradient diagnostic")
        total.backward()
        for name, p in model.named_parameters():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                raise FloatingPointError(f"Nonfinite/missing optimizer gradient: {name}")
        grads = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
        norm = nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
        optim.step()
        scheduler.step()
        step_gradients.append(grads)
        step_losses.append({"step": step_idx, "total": float(total.item()),
                            "bce": float(bce.item()), "rel": float(rel.item()),
                            "grad_norm_before_clip": float(norm.item()),
                            "lambda_applied": lambda_at_epoch(1, .10) if mode == "A3_warmup_lambda010" else 0.0,
                            "eligible_images": batch_info["eligible_images"],
                            "eligible_lesions": batch_info["eligible_lesions"],
                            "lr_after_step": float(optim.param_groups[0]["lr"])})
        del images, targets, grads
    return DiagnosticResult(step_losses, step_gradients,
                            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            bool(torch.any(model.classifier.weight.detach() != original_head)),
                            sum(b["eligible_images"] for b in batch_reports),
                            summarize_exposure(batch_reports, index), ratio, cosine)


def max_float_difference(first: Mapping[str, Tensor], second: Mapping[str, Tensor]) -> float:
    if first.keys() != second.keys():
        raise RuntimeError("Different state/gradient keys in parity comparison")
    delta = 0.0
    for key in first:
        a, b = first[key], second[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"Tensor metadata mismatch: {key}")
        if a.is_floating_point():
            delta = max(delta, float((a.float() - b.float()).abs().max()))
        elif not torch.equal(a, b):
            raise RuntimeError(f"Non-floating model state differs: {key}")
    return delta


def verify_two_step_parity(first: DiagnosticResult, zero: DiagnosticResult) -> dict:
    if len(first.step_losses) != 2 or len(zero.step_losses) != 2:
        raise RuntimeError("Not a two-step comparison")
    max_loss = max(abs(a["total"] - b["total"]) for a, b in zip(first.step_losses, zero.step_losses))
    max_grad = max(max_float_difference(a, b) for a, b in zip(first.step_gradients, zero.step_gradients))
    max_state = max_float_difference(first.final_state, zero.final_state)
    if max_loss > 1e-6 or max_grad > 1e-5 or max_state > 1e-6:
        raise RuntimeError(f"Two-step B0/A3(lambda0) parity failed: {max_loss}, {max_grad}, {max_state}")
    if not first.head_changed or not zero.head_changed:
        raise RuntimeError("Classifier was not updated in comparison")
    return {"max_loss_abs_diff": max_loss, "max_gradient_abs_diff": max_grad,
            "max_state_abs_diff": max_state}


def run_two_batch_smoke() -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; no CPU or FP16 substitution")
    path = OUT / PREFLIGHT_FILENAME
    if not path.is_file():
        raise ValueError("Run --preflight before --two-batch-smoke")
    index, selected, config, current = preflight()
    if json.loads(path.read_text(encoding="utf-8")) != current:
        raise ValueError("Stale preflight evidence, source or inputs changed")
    set_seed(SEED)
    batches = load_two_batches(ROOT, selected)
    records = [validate_batch_identity(index, ids, targets) for _, targets, ids in batches]
    exposure = summarize_exposure(records, index)
    if exposure["eligible_unique_images"] != 4 or exposure["eligible_lesions_seen"] < 4:
        raise RuntimeError("Enriched smoke missing K1/K2 region comparisons")
    initial = MobileNetV2OS16(pretrained=True)
    if initial.pretrained_sha256 != config["pretrained_sha256"]:
        raise ValueError("ImageNet weights changed between preflight and smoke")
    initial_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
    del initial

    def branch(mode: str) -> DiagnosticResult:
        m = MobileNetV2OS16(pretrained=False)
        m.load_state_dict(initial_state, strict=True)
        result = two_step_branch(m, batches, index, mode=mode)
        del m
        gc.collect()
        torch.cuda.empty_cache()
        return result

    first = branch("B0")
    zero = branch("A3_lambda0")
    parity = verify_two_step_parity(first, zero)
    active = branch("A3_warmup_lambda010")
    if active.eligible_images_seen <= 0 or not active.head_changed:
        raise RuntimeError("REL was never active or optimizer did not update")
    for step in active.step_losses:
        expected = step["bce"] + step["lambda_applied"] * step["rel"]
        if abs(step["total"] - expected) > 2e-6:
            raise RuntimeError("BCE + lambda*REL arithmetic mismatch")
    result = {
        **current, "status": "P6_TWO_BATCH_CUDA_BF16_SMOKE_PASS",
        "scope": "TRAIN_DICOM_8_PATIENTS_ONLY", "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "two_batch_exposure": exposure,
        "parity": parity, "gradient_ratio_head": active.grad_ratio_head,
        "gradient_cosine_head": active.grad_cosine_head,
        "steps_by_branch": {"B0": first.step_losses, "A3_lambda0": zero.step_losses,
                            "A3_epoch1_lambda002": active.step_losses},
        "optimizer_steps_each_branch": 2, "validation_run": False,
        "training_started": False, "checkpoint_created": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--two-batch-smoke", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        _, _, _, record = preflight()
        save_new_json(OUT / PREFLIGHT_FILENAME, record)
        print("P6 PREPARATION PREFLIGHT PASS: frozen TRAIN/P3/B0 recipe verified")
    else:
        record = run_two_batch_smoke()
        save_new_json(OUT / SMOKE_FILENAME, record)
        print("P6 TWO-BATCH CUDA BF16 SMOKE PASS: B0/A3 lambda0 parity and REL gradients")
    print("P6 MULTI-EPOCH TRAINING NOT STARTED")


if __name__ == "__main__":
    main()
