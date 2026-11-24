"""BG-REL V3.2 P6 candidate trainer: development-only, frozen-P3 A3.

CLI:
    python code/train_bgrel.py --preflight --lambda-max 0.10 --seed 42
    python code/train_bgrel.py --dry-run --lambda-max 0.10 --seed 42
    python code/train_bgrel.py --train --lambda-max 0.10 --seed 42 --approval <JSON>
    python code/train_bgrel.py --train --resume --lambda-max 0.10 --seed 42 --approval <JSON>

Preflight/dry-run never create model checkpoints.  --train requires an explicit,
researcher-supplied pilot approval with P3 limitations acknowledged. TEST DICOM
and TEST metrics are never accessed. This trainer is not an ablation generator.
"""
from __future__ import annotations
from refactor_helpers import close_logger, write_metrics_atomic

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dataset import RSNAImageDataset, load_frozen, sha256
from metrics import binary_metrics
from model import MobileNetV2OS16, imagenet_weight_path
from rel_adapter import FrozenControlIndex, P3_TRACE_SHA, load_frozen_controls, loss_from_frozen_train_batch
from t3_train_integration import IdentifiedTrainDataset, check_b0_recipe
from p6_preflight import lambda_at_epoch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_COUNT = 23748
VAL_COUNT = 2969
BATCH = 4
MAX_EPOCHS = 20
PATIENCE = 4
MIN_IMPROVEMENT = 0.001
LR = 3e-5
WEIGHT_DECAY = 1e-3
CLIP_NORM = 5.0
PILOT_LAMBDAS = (0.10, 0.25)
SEED_ALLOWED = 42  # Initial candidate only; second paired seed requires separate registration.
B0_RUN = "b0_dev_s42"
P6_SAFETY_EVIDENCE = "p6_two_batch_cuda_exposure_fix.json"
P6_PREFLIGHT_EVIDENCE = "p6_preflight_exposure_fix.json"
SRC_FILES = ("dataset.py", "model.py", "metrics.py", "regions.py", "losses.py",
             "rel_adapter.py", "t3_train_integration.py", "p6_preflight.py", "train_bgrel.py", "refactor_helpers.py", "paths.py")
METRIC_COLS = (
    "epoch", "train_bce", "train_rel_active_mean", "train_total", "val_bce",
    "val_auroc", "val_ap", "lambda_epoch", "active_batches", "total_batches",
    "active_batch_rate", "eligible_images_seen", "eligible_unique_patients",
    "eligible_lesions_seen", "k1_lesions_seen", "k2_lesions_seen", "k0_lesions_seen",
    "gradient_ratio_head", "gradient_cosine_head", "optimizer_steps", "lr_last",
    "grad_norm_max", "gpu_peak_mib", "best_epoch", "best_val_auroc",
    "patience_counter", "epoch_seconds"
)


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    return {name: sha256(root / "code" / name) for name in SRC_FILES}


def pilot_run_id(lambda_max: float, seed: int) -> str:
    if lambda_max not in PILOT_LAMBDAS or seed != SEED_ALLOWED:
        raise ValueError("Unregistered pilot candidate/seed; request protocol approval")
    return f"a3_l{int(round(lambda_max * 1000)):03d}_s{seed}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_bf16_cuda() -> torch.device:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required; CPU/FP16 fallback is forbidden")
    return torch.device("cuda")


def atomic_json(path: Path, record: Mapping) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def atomic_replace(path: Path, writer) -> None:
    """Checkpoint/CSV atomic replacement at epoch boundaries, never silent cleanup."""
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Interrupted artifact detected; investigate before rerun: {temporary}")
    try:
        writer(temporary)
        os.replace(temporary, path)
    except Exception:
        # Preserve .tmp for forensics. Do not delete possibly valuable state.
        raise


def write_metrics(path: Path, rows: list[dict]) -> None:
    write_metrics_atomic(path, rows, fields=METRIC_COLS, replace=atomic_replace)


def save_checkpoint(path: Path, state: dict) -> None:
    atomic_replace(path, lambda temp: torch.save(state, temp))


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(record: dict) -> None:
    random.setstate(record["python"])
    np.random.set_state(record["numpy"])
    torch.set_rng_state(record["torch_cpu"])
    if record["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Saved CUDA RNG cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(record["torch_cuda"])


def load_approval(path: Path, *, run_id: str, lambda_max: float, seed: int) -> dict:
    """An explicit protocol decision, not a mathematical/clinical QA endorsement."""
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": "BGREL_V3.2", "scope": "development_pilot_only",
        "run_id": run_id, "seed": seed, "lambda_max": lambda_max,
        "recipe": B0_RUN, "sampler": "frozen_P3_budget128_step8",
        "p3_anatomical_qa": "PENDING_INDEPENDENT_REVIEW",
        "p3_limitations_acknowledged": True,
        "approved_by_researcher": True,
        "test_access_authorized": False,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"P6 pilot approval missing/invalid: {key}")
    if not isinstance(record.get("decision_rationale"), str) or not record["decision_rationale"].strip():
        raise ValueError("Explicit reason for provisional P3 use is required")
    if not isinstance(record.get("decision_date"), str) or not record["decision_date"].strip():
        raise ValueError("Approval decision date required")
    return record


def assert_frozen_recipe(config: Mapping, summary: Mapping, root: Path) -> None:
    check_b0_recipe(config, summary, dataset_sha=sha256(root / "code/dataset.py"),
                    model_sha=sha256(root / "code/model.py"))
    if config.get("batch") != BATCH or config.get("epochs_cap") != MAX_EPOCHS:
        raise ValueError("Unexpected B0 training recipe")
    if config.get("patience") != PATIENCE or config.get("min_improvement") != MIN_IMPROVEMENT:
        raise ValueError("Unexpected B0 early-stop rule")
    if config.get("weight_decay") != WEIGHT_DECAY or config.get("lr") != LR:
        raise ValueError("Unexpected B0 optimizer recipe")
    if summary.get("test_images_opened") is not False:
        raise ValueError("B0 TEST seal provenance invalid")
    checkpoint = root / "results/runs" / B0_RUN / "best.pt"
    if not checkpoint.is_file() or sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("Frozen B0 checkpoint file/hash mismatch")


def preflight(root: Path = ROOT, *, lambda_max: float, seed: int) -> tuple[list[dict], list[dict], FrozenControlIndex, dict]:
    """Read hashed TRAIN/VAL metadata, never decode TEST pixels."""
    run_id = pilot_run_id(lambda_max, seed)
    train_rows, val_rows = load_frozen(root)
    if len(train_rows) != TRAIN_COUNT or len(val_rows) != VAL_COUNT:
        raise RuntimeError("Frozen split counts changed")
    index = load_frozen_controls(root)
    if set(index.train_targets) != {r["image_id"] for r in train_rows}:
        raise ValueError("P3 frozen TRAIN IDs diverge from split")
    config_path = root / "results/runs" / B0_RUN / "config.json"
    summary_path = root / "results/runs" / B0_RUN / "summary.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert_frozen_recipe(config, summary, root)
    safety_path = root / "results/qa" / P6_SAFETY_EVIDENCE
    proof_path = root / "results/qa" / P6_PREFLIGHT_EVIDENCE
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if safety.get("status") != "P6_TWO_BATCH_CUDA_BF16_SMOKE_PASS" or safety.get("parity") != {
        "max_loss_abs_diff": 0.0, "max_gradient_abs_diff": 0.0, "max_state_abs_diff": 0.0
    }:
        raise ValueError("P6 two-batch safety evidence missing or parity changed")
    if proof.get("status") != "P6_PREPARATION_PREFLIGHT_PASS":
        raise ValueError("P6 frozen preflight evidence missing")
    for name in ("source_sha256", "split_sha256", "master_sha256", "baseline_best_sha256",
                 "b0_config_sha256", "b0_summary_sha256"):
        if safety.get(name) != proof.get(name):
            raise ValueError(f"P6 evidence mismatch: {name}")
    if safety.get("baseline_best_sha256") != summary["checkpoint_sha256"]:
        raise ValueError("P6/B0 checkpoint seal changed")
    if safety.get("b0_config_sha256") != sha256(config_path) or safety.get("b0_summary_sha256") != sha256(summary_path):
        raise ValueError("B0 config/summary file contents changed")
    for name, value in safety["source_sha256"].items():
        if sha256(root / "code" / name) != value:
            raise ValueError(f"P6 reviewed source drift: {name}")
    if safety.get("p3_anatomical_quality") != "PENDING_INDEPENDENT_REVIEW":
        raise ValueError("Changed P3 anatomical review status requires new protocol")
    if not imagenet_weight_path().is_file() or sha256(imagenet_weight_path()) != config["pretrained_sha256"]:
        raise ValueError("ImageNet init weight provenance mismatch")
    record = {"status": "P6_TRAINER_PREFLIGHT_READY_NOT_APPROVED",
              "run_id": run_id, "seed": seed, "lambda_max": lambda_max,
              "train_samples": TRAIN_COUNT, "val_samples": VAL_COUNT,
              "p3_eligible_images": index.eligible_images,
              "p3_eligible_lesions": index.eligible_lesions,
              "p3_controls": index.control_count,
              "p3_k0_lesions": index.k0_lesions,
              "source_sha256": source_hashes(root),
              "frozen_split_sha256": config["split_sha256"],
              "p6_smoke_sha256": sha256(safety_path),
              "b0_checkpoint_sha256": summary["checkpoint_sha256"],
              "b0_config_sha256": sha256(config_path),
              "p3_anatomical_qa": "PENDING_INDEPENDENT_REVIEW",
              "test_images_opened": False,
              "multi_epoch_training_started": False}
    return train_rows, val_rows, index, record


def make_loader(root: Path, rows: Sequence[dict], *, train: bool, seed: int, epoch: int) -> DataLoader:
    """Mirror B0 sampler/augmentation; attach IDs only for TRAIN."""
    base = RSNAImageDataset(root, list(rows), train=train, seed=seed, epoch=epoch)
    if train:
        dataset = IdentifiedTrainDataset(base, rows)
        generator = torch.Generator().manual_seed(seed + epoch * 1009 + 1)
    else:
        dataset = base
        generator = torch.Generator().manual_seed(seed + 0)
    return DataLoader(dataset, batch_size=BATCH, shuffle=train, num_workers=0,
                      drop_last=False, pin_memory=torch.cuda.is_available(), generator=generator)


def frozen_k0_counts(root: Path) -> dict[str, int]:
    """Read verified TRAIN P3 trace once, retain K=0 counts per image."""
    path = root / "results/qa/sampler_trace.csv"
    if sha256(path) != P3_TRACE_SHA:
        raise ValueError("P3 frozen trace hash mismatch")
    counts: dict[str, int] = {}
    k0_total = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["k"] == "0":
                counts[row["image_id"]] = counts.get(row["image_id"], 0) + 1
                k0_total += 1
    if k0_total != 6494:
        raise ValueError("P3 K=0 lesion count changed")
    return counts


def batch_exposure(index: FrozenControlIndex, image_ids: Sequence[str], labels: Tensor,
                   patient_ids: Mapping[str, str], k0_by_image: Mapping[str, int]) -> dict:
    if len(set(image_ids)) != len(image_ids) or labels.shape != (len(image_ids),):
        raise ValueError("Duplicate image ID or target mismatch")
    eligible_images, eligible_lesions, k1, k2, k0 = 0, 0, 0, 0, 0
    eligible_patients = []
    for sid, target in zip(image_ids, labels.detach().cpu().tolist()):
        if sid not in index.train_targets or int(target) != index.train_targets[sid]:
            raise ValueError(f"Identity/label mismatch: {sid}")
        groups = index.for_image(sid)
        if groups:
            eligible_images += 1
            if sid not in patient_ids:
                raise ValueError(f"Missing frozen patient mapping: {sid}")
            eligible_patients.append(patient_ids[sid])
        for group in groups:
            k = len(group.control_xyxy)
            if k == 1:
                k1 += 1
            elif k == 2:
                k2 += 1
            else:
                raise ValueError("Only frozen K=1/2 groups may reach REL")
            eligible_lesions += 1
        k0 += k0_by_image.get(sid, 0)
        if target == 0 and k0_by_image.get(sid, 0) > 0:
            raise ValueError("Negative image has positive P3 K=0 annotations")
    return {"eligible_images": eligible_images, "eligible_lesions": eligible_lesions,
            "k1": k1, "k2": k2, "k0": k0, "patients": eligible_patients}


def head_grad_diagnostics(bce: Tensor, rel: Tensor, head: nn.Linear, lam: float) -> tuple[float, float]:
    if lam <= 0:
        raise ValueError("Gradient diagnostics require positive lambda")
    params = tuple(head.parameters())
    gb = torch.autograd.grad(bce, params, retain_graph=True, allow_unused=False)
    gr = torch.autograd.grad(rel, params, retain_graph=True, allow_unused=False)
    a = torch.cat([g.detach().float().flatten() for g in gb])
    b = torch.cat([g.detach().float().flatten() for g in gr])
    ratio = float((lam * torch.linalg.vector_norm(b) / (torch.linalg.vector_norm(a) + 1e-12)).item())
    cosine = float((torch.dot(a, b) / (torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)+1e-12)).item())
    if not math.isfinite(ratio) or not math.isfinite(cosine):
        raise FloatingPointError("Nonfinite REL head gradient diagnostics")
    return ratio, cosine


def train_epoch(model: MobileNetV2OS16, loader: DataLoader, index: FrozenControlIndex,
                optimizer: torch.optim.Optimizer, scheduler, device: torch.device,
                *, epoch: int, lambda_max: float, logger: logging.Logger,
                patient_ids: Mapping[str, str], k0_by_image: Mapping[str, int],
                max_batches: int | None = None) -> dict:
    model.train()
    lam = lambda_at_epoch(epoch, lambda_max)
    if not (0 < lam <= lambda_max + 1e-12):
        raise ValueError("Invalid epoch lambda warmup")
    totals = {"samples": 0, "steps": 0, "bce_sum": 0., "total_sum": 0., "rel_active_sum": 0.,
              "active_batches": 0, "eligible_images": 0, "eligible_lesions": 0, "k1": 0, "k2": 0, "k0": 0,
              "grad_norm_max": 0.}
    patients: set[str] = set()
    diagnostics = (None, None)
    total_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    for batch_no, (x, y, ids) in enumerate(loader, start=1):
        if max_batches is not None and batch_no > max_batches:
            break
        exposure = batch_exposure(index, ids, y, patient_ids, k0_by_image)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feature = model.forward_features(x)
            parts = loss_from_frozen_train_batch(index, ids, y, feature,
                                                 model.classifier, lambda_value=lam)
        if (parts.eligible_images != exposure["eligible_images"] or
                parts.eligible_lesions != exposure["eligible_lesions"]):
            raise ValueError("Computed REL exposure does not match frozen metadata")
        if not all(bool(torch.isfinite(value)) for value in (parts.total, parts.bce, parts.rel)):
            raise FloatingPointError("Nonfinite P6 train objective")
        if exposure["eligible_images"] and diagnostics[0] is None:
            diagnostics = head_grad_diagnostics(parts.bce, parts.rel, model.classifier, lam)
        parts.total.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM, error_if_nonfinite=True)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite gradients")
        optimizer.step()
        scheduler.step()
        n = len(ids)
        totals["samples"] += n
        totals["steps"] += 1
        totals["bce_sum"] += float(parts.bce.item()) * n
        totals["total_sum"] += float(parts.total.item()) * n
        totals["grad_norm_max"] = max(totals["grad_norm_max"], float(norm.item()))
        if exposure["eligible_images"]:
            totals["rel_active_sum"] += float(parts.rel.item()) * exposure["eligible_images"]
            totals["active_batches"] += 1
        totals["eligible_images"] += exposure["eligible_images"]
        totals["eligible_lesions"] += exposure["eligible_lesions"]
        totals["k1"] += exposure["k1"]
        totals["k2"] += exposure["k2"]
        totals["k0"] += exposure["k0"]
        patients.update(exposure["patients"])
        if batch_no % 500 == 0 or batch_no == total_batches:
            logger.info("Epoch %d batch %d/%d BCE=%.5f RELactive=%d LR=%.9g",
                        epoch, batch_no, total_batches,
                        totals["bce_sum"]/totals["samples"], totals["active_batches"],
                        optimizer.param_groups[0]["lr"])
    if totals["samples"] == 0:
        raise RuntimeError("No TRAIN samples processed")
    return {"samples_seen": totals["samples"],
            "train_bce": totals["bce_sum"] / totals["samples"],
            "train_rel_active_mean": totals["rel_active_sum"] / max(totals["eligible_images"], 1),
            "train_total": totals["total_sum"] / totals["samples"],
            "active_batches": totals["active_batches"],
            "total_batches": totals["steps"],
            "active_batch_rate": totals["active_batches"] / totals["steps"],
            "eligible_images_seen": totals["eligible_images"],
            "eligible_unique_patients": len(patients),
            "eligible_lesions_seen": totals["eligible_lesions"],
            "k1_lesions_seen": totals["k1"], "k2_lesions_seen": totals["k2"],
            "k0_lesions_seen": totals["k0"],
            "gradient_ratio_head": diagnostics[0], "gradient_cosine_head": diagnostics[1],
            "optimizer_steps": totals["steps"],
            "lr_last": optimizer.param_groups[0]["lr"],
            "grad_norm_max": totals["grad_norm_max"]}


@torch.inference_mode()
def validate_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    n, sum_loss, targets, logits = 0, 0.0, [], []
    # B0 runs validation FP32 without AMP; do not change prediction recipe.
    for x, y in loader:
        z = model(x.to(device, non_blocking=True)).float().cpu()
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("Nonfinite validation logits")
        sum_loss += float(nn.functional.binary_cross_entropy_with_logits(
            z, y, reduction="sum").item())
        n += len(y)
        targets.extend(y.tolist())
        logits.extend(z.tolist())
    if n != len(loader.dataset):
        raise ValueError("Validation did not cover full frozen VAL partition")
    auroc, ap = binary_metrics(targets, logits)
    return sum_loss/n, auroc, ap


def configured_run(root: Path, record: Mapping, approval: Mapping) -> dict:
    return {"phase": "P6", "experiment": "A3_DEVELOPMENT_PILOT", "run_id": record["run_id"],
            "seed": record["seed"], "lambda_max": record["lambda_max"],
            "lambda_schedule": "epoch1_0.2_epoch2_0.6_epoch3+_1.0",
            "sampler": "frozen_P3_budget128_step8", "k_max": 2,
            "base_recipe": B0_RUN, "batch": BATCH, "epochs_cap": MAX_EPOCHS,
            "patience": PATIENCE, "min_improvement": MIN_IMPROVEMENT,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR_per_optimizer_step",
            "amp_precision": "bf16", "grad_scaler_enabled": False,
            "augment_train": "brightness_contrast_[0.9,1.1]", "validation_augment": False,
            "train_samples": TRAIN_COUNT, "val_samples": VAL_COUNT,
            "source_sha256": record["source_sha256"], "split_sha256": record["frozen_split_sha256"],
            "p6_safety_sha256": record["p6_smoke_sha256"],
            "b0_checkpoint_sha256": record["b0_checkpoint_sha256"],
            "approval_sha256": hashlib.sha256(json.dumps(approval, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "test_images_opened": False, "p3_anatomical_qa": "PENDING_INDEPENDENT_REVIEW"}


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("bgrel_p6_trainer")
    if logger.handlers:
        raise RuntimeError("Logger already initialized; avoid duplicate log file")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(formatter)
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(stdout)
    logger.addHandler(fh)
    return logger




def restore_checkpoint(record: Mapping, config: Mapping, *, epoch_cap: int) -> tuple[int, list[dict]]:
    if record.get("config") != config:
        raise ValueError("Resume config provenance mismatch")
    epoch = record.get("epoch")
    rows = record.get("metrics")
    if not isinstance(epoch, int) or epoch < 1 or epoch >= epoch_cap:
        raise ValueError("Invalid resume epoch")
    if not isinstance(rows, list) or len(rows) != epoch:
        raise ValueError("Missing/corrupt checkpoint metrics")
    if any(int(row["epoch"]) != i for i, row in enumerate(rows, start=1)):
        raise ValueError("Nonconsecutive resume metrics")
    if int(record["patience"]) >= PATIENCE:
        raise RuntimeError("Already early-stopped; do not resume")
    return epoch + 1, rows


def run_pilot(root: Path, *, lambda_max: float, seed: int, approval_path: Path, resume: bool) -> None:
    device = require_bf16_cuda()
    train, val, index, pre = preflight(root, lambda_max=lambda_max, seed=seed)
    approval = load_approval(approval_path, run_id=pre["run_id"], lambda_max=lambda_max, seed=seed)
    if sha256(root / "code/train_bgrel.py") != pre["source_sha256"]["train_bgrel.py"]:
        raise RuntimeError("P6 trainer changed after preflight")
    run = root / "results/runs" / pre["run_id"]
    if resume:
        if not run.is_dir():
            raise ValueError("Requested resume but run folder missing")
        if (run / "summary.json").exists():
            raise RuntimeError("Completed pilot cannot be resumed")
    else:
        if run.exists():
            raise FileExistsError("Existing P6 run; refusing overwrite")
    set_seed(seed)
    model = MobileNetV2OS16(pretrained=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train)/BATCH)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=steps_per_epoch * MAX_EPOCHS)
    conf = configured_run(root, pre, approval)
    # All validations before creating a new run directory.
    if model.pretrained_sha256 != json.loads((root / "results/runs" / B0_RUN / "config.json").read_text())["pretrained_sha256"]:
        raise ValueError("ImageNet model init provenance mismatch")
    if resume:
        if json.loads((run / "config.json").read_text(encoding="utf-8")) != conf:
            raise ValueError("Resume config changed")
        # Only load locally generated, explicitly requested checkpoint. PyTorch optimizer
        # states require trusted deserialization; never load an unknown checkpoint.
        last = torch.load(run / "last.pt", map_location=device, weights_only=False)
        start, rows = restore_checkpoint(last, conf, epoch_cap=MAX_EPOCHS)
        model.load_state_dict(last["model"], strict=True)
        optim.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        restore_rng(last["rng"])
        with (run / "metrics.csv").open(encoding="utf-8", newline="") as f:
            recorded = list(csv.DictReader(f))
        if len(recorded) != len(rows) or int(recorded[-1]["epoch"]) != start-1:
            raise ValueError("Resume metrics checkpoint disagreement")
        # Exact numeric row parity, not only count; CSV stores numeric strings.
        for a, b in zip(recorded, rows):
            if a != {k: "" if v is None else str(v) for k,v in b.items()}:
                raise ValueError("Metrics changed after last checkpoint")
        best, best_epoch, patience, best_patience_auc = (
            float(last["best_auc"]), int(last["best_epoch"]),
            int(last["patience"]), float(last["best_patience_auc"]))
    else:
        start, rows, best, best_epoch, patience, best_patience_auc = (1, [], -math.inf, 0, 0, -math.inf)
        run.mkdir(parents=True, exist_ok=False)
        atomic_json(run / "config.json", conf)
    logger = make_logger(run / "train.log")
    try:
        logger.info("P6 PILOT START run=%s lambda_max=%.3f seed=%d resume=%s P3_QA=PENDING",
                    pre["run_id"],lambda_max,seed,resume)
        patient_ids = {r["image_id"]: r["patient_id"] for r in train}
        k0_by_image = frozen_k0_counts(root)
        for epoch in range(start, MAX_EPOCHS+1):
            t0 = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            loader = make_loader(root, train, train=True, seed=seed, epoch=epoch)
            result = train_epoch(model, loader, index, optim, scheduler, device, epoch=epoch,
                                 lambda_max=lambda_max, logger=logger,
                                 patient_ids=patient_ids, k0_by_image=k0_by_image)
            if result["optimizer_steps"] != steps_per_epoch or result["samples_seen"] != len(train) or result["eligible_images_seen"] != index.eligible_images or result["eligible_lesions_seen"] != index.eligible_lesions or result["k0_lesions_seen"] != index.k0_lesions:
                raise RuntimeError("Unexpected incomplete TRAIN epoch")
            val_bce, auc, ap = validate_epoch(model,
                make_loader(root, val, train=False, seed=seed, epoch=0),device)
            if auc > best + 1e-12:
                best, best_epoch = auc, epoch
                save_checkpoint(run / "best.pt", {
                    "model": model.state_dict(), "epoch": epoch, "val_auc": auc,
                    "val_ap": ap, "config": conf})
            if auc > best_patience_auc + MIN_IMPROVEMENT:
                best_patience_auc, patience = auc, 0
            else:
                patience += 1
            row = {"epoch": epoch, "train_bce": result["train_bce"],
                   "train_rel_active_mean": result["train_rel_active_mean"],
                   "train_total": result["train_total"],
                   "val_bce": val_bce, "val_auroc": auc, "val_ap": ap,
                   "lambda_epoch": lambda_at_epoch(epoch, lambda_max),
                   **{k:v for k,v in result.items() if k not in ("samples_seen", "train_bce", "train_rel_active_mean", "train_total")},
                   "gpu_peak_mib": torch.cuda.max_memory_allocated()/1024**2,
                   "best_epoch": best_epoch, "best_val_auroc": best,
                   "patience_counter": patience, "epoch_seconds": time.monotonic()-t0}
            if set(row) != set(METRIC_COLS):
                raise RuntimeError(f"Unexpected per-epoch metric schema: {set(row)^set(METRIC_COLS)}")
            rows.append(row)
            checkpoint = {"model":model.state_dict(), "optimizer":optim.state_dict(),
                          "scheduler":scheduler.state_dict(), "epoch": epoch,
                          "best_auc": best, "best_epoch": best_epoch,
                          "patience":patience, "best_patience_auc":best_patience_auc,
                          "metrics":rows, "rng":capture_rng(), "config":conf}
            save_checkpoint(run / "last.pt", checkpoint)
            write_metrics(run / "metrics.csv", rows)
            logger.info("EPOCH %d train_BCE=%.5f train_REL=%.5f val_BCE=%.5f AUC=%.6f AP=%.6f active=%.4f best=%d patience=%d",
                        epoch,result["train_bce"],result["train_rel_active_mean"],
                        val_bce,auc,ap,result["active_batch_rate"],best_epoch,patience)
            if patience >= PATIENCE:
                logger.info("EARLY STOP at epoch=%d",epoch)
                break
        atomic_json(run / "summary.json", {"status":"FINISHED_P6_DEVELOPMENT_PILOT",
            "best_epoch":best_epoch,"best_val_auroc":best,"last_epoch":epoch,
            "checkpoint_sha256":sha256(run/"best.pt"),"metrics_sha256":sha256(run/"metrics.csv"),
            "source_sha256":source_hashes(root),"test_images_opened":False,
            "p3_anatomical_qa":"PENDING_INDEPENDENT_REVIEW"})
        logger.info("P6 PILOT COMPLETE best_epoch=%d best_AUROC=%.6f",best_epoch,best)
    finally:
        close_logger(logger)


def dry_run(root: Path, *, lambda_max: float, seed: int) -> dict:
    """Only 2 full-shuffled TRAIN batches plus 1 VAL batch, no checkpoint/scheduler artifact."""
    device = require_bf16_cuda()
    train, val, index, pre = preflight(root, lambda_max=lambda_max, seed=seed)
    output = root / "results/qa" / f"p6_trainer_dryrun_{pre['run_id']}.json"
    if output.exists():
        raise FileExistsError(f"Dry-run evidence exists: {output}")
    set_seed(seed)
    model = MobileNetV2OS16(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=math.ceil(TRAIN_COUNT/BATCH)*MAX_EPOCHS)
    logger = logging.getLogger("p6_dryrun")
    logger.addHandler(logging.NullHandler())
    try:
        result = train_epoch(model, make_loader(root, train, train=True, seed=seed, epoch=1),
                             index,optimizer,scheduler,device,epoch=1,
                             lambda_max=lambda_max,logger=logger,
                             patient_ids={r["image_id"]: r["patient_id"] for r in train},
                             k0_by_image=frozen_k0_counts(root), max_batches=2)
        val_loader = make_loader(root, val[:BATCH], train=False, seed=seed, epoch=0)
        with torch.inference_mode():
            model.eval()
            x,y = next(iter(val_loader))
            z = model(x.to(device)).float().cpu()
            if z.shape != y.shape or not bool(torch.isfinite(z).all()):
                raise ValueError("VAL batch inference failed")
        evidence = {**pre, "status":"P6_TRAINER_DRY_RUN_PASS",
                    "scope":"2_FULL_LOADER_TRAIN_BATCHES_1_VAL_BATCH_ONLY",
                    "optimizer_steps":result["optimizer_steps"],
                    "active_batch_rate_in_enriched_sample":None,
                    "active_batch_rate_random_two_batches":result["active_batch_rate"],
                    "eligible_images_seen":result["eligible_images_seen"],
                    "val_batch_size":len(y), "checkpoint_created":False,
                    "training_run_created":False}
        atomic_json(output,evidence)
        return evidence
    finally:
        del model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--train", action="store_true")
    parser.add_argument("--lambda-max", type=float, required=True, choices=PILOT_LAMBDAS)
    parser.add_argument("--seed", type=int, default=SEED_ALLOWED)
    parser.add_argument("--approval", type=Path, help="Researcher's explicit DEVELOPMENT pilot approval JSON")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.approval or args.resume) and not args.train:
        parser.error("--approval/--resume only valid with --train")
    if args.train and not args.approval:
        parser.error("--train requires researcher pilot approval")
    if args.preflight:
        _,_,_,record=preflight(ROOT,lambda_max=args.lambda_max,seed=args.seed)
        output=ROOT/"results/qa"/f"p6_trainer_preflight_{record['run_id']}.json"
        atomic_json(output,record)
        print("P6 TRAINER PREFLIGHT PASS (NOT TRAINING APPROVAL)")
    elif args.dry_run:
        record=dry_run(ROOT,lambda_max=args.lambda_max,seed=args.seed)
        print("P6 TRAINER DRY-RUN PASS",record["status"])
    else:
        run_pilot(ROOT,lambda_max=args.lambda_max,seed=args.seed,
                  approval_path=args.approval,resume=args.resume)


if __name__ == "__main__":
    main()
