"""P7: B1 box-positive BCE and A2 geometry-only REL ablations.

Development TRAIN/VAL only. TEST images are never decoded.

B1:
    Image-level BCE + positive BCE over every expert lesion box.
    No control regions or lung masks are used.

A2:
    Image-level BCE + listwise REL.
    Expert lesion boxes are retained.
    Controls are selected using image-coordinate geometry only.
    No lung masks or PSPNet inference are used.

Both variants reuse the frozen B0 training recipe:
    MobileNetV2-OS16, seed 42, batch 4, BF16,
    AdamW, cosine scheduler, maximum 20 epochs,
    validation AUROC selection and patience 4.

Commands:
    python code/train_ablations.py --variant b1 --preflight
    python code/train_ablations.py --variant b1 --smoke
    python code/train_ablations.py --variant b1 --train

    python code/train_ablations.py --variant a2 --preflight
    python code/train_ablations.py --variant a2 --smoke
    python code/train_ablations.py --variant a2 --train

    python code/train_ablations.py --variant b1 --resume
    python code/train_ablations.py --variant a2 --resume

Existing B0/A3 checkpoints and P8 outputs are never modified.
"""

from __future__ import annotations
from refactor_helpers import close_logger, require, write_metrics_atomic
from training_objectives import (
    collect_region_scores,
    compute_training_loss,
    candidate_axis,
    geometry_candidates,
    sample_geometry_controls,
    geometry_csv_text,
    text_sha256,
)

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import (
    RSNAImageDataset,
    load_frozen,
    sha256,
)
from losses import bce_rel_loss
from metrics import binary_metrics
from model import MobileNetV2OS16, imagenet_weight_path
from regions import scale_boxes_xyxy, shared_region_scores
from sample_controls import (
    Rect,
    rect_from_row,
    intersect,
    iou,
    stable_jitter,
    EXPECTED_BOXES,
)
from rel_adapter import load_frozen_controls
from t3_train_integration import (
    IdentifiedTrainDataset,
    check_b0_recipe,
)


ROOT = Path(__file__).resolve().parents[1]

SEED = 42
BATCH = 4
TRAIN_COUNT = 23748
VAL_COUNT = 2969
TRAIN_LESIONS = 9161

MAX_EPOCHS = 20
PATIENCE = 4
MIN_IMPROVEMENT = 0.001

LR = 3e-5
WEIGHT_DECAY = 1e-3
CLIP_NORM = 5.0

LAMBDA_MAX = 0.10

SIZE = 1024
STEP = 8
BUDGET = 128
K_MAX = 2
VERTICAL_DISTANCE = 0.10
PAIR_IOU_MAX = 0.20
EPS = 1e-6

RUN_IDS = {
    "b1": "b1_boxpos_s42",
    "a2": "a2_geometry_s42",
}

SOURCE_FILES = (
    "dataset.py",
    "model.py",
    "metrics.py",
    "regions.py",
    "losses.py",
    "sample_controls.py",
    "rel_adapter.py",
    "t3_train_integration.py",
    "train_ablations.py",
    "training_objectives.py",
    "refactor_helpers.py",
    "paths.py",
)

METRIC_FIELDS = (
    "epoch",
    "train_bce",
    "train_aux_active_mean",
    "train_total",
    "val_bce",
    "val_auroc",
    "val_ap",
    "lambda_epoch",
    "active_batches",
    "total_batches",
    "active_batch_rate",
    "eligible_images_seen",
    "eligible_unique_patients",
    "eligible_lesions_seen",
    "optimizer_steps",
    "lr_last",
    "grad_norm_max",
    "gpu_peak_mib",
    "best_epoch",
    "best_val_auroc",
    "patience_counter",
    "epoch_seconds",
)

GEOMETRY_FIELDS = (
    "image_id",
    "patient_id",
    "annotation_id",
    "control_index",
    "tier",
    "lesion_x0",
    "lesion_y0",
    "lesion_x1",
    "lesion_y1",
    "control_x0",
    "control_y0",
    "control_x1",
    "control_y1",
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / "code" / name) for name in SOURCE_FILES}


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_cuda() -> torch.device:
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(
        torch.cuda.is_bf16_supported(),
        "BF16 CUDA support is required",
    )
    return torch.device("cuda")


def lambda_at_epoch(epoch: int) -> float:
    require(epoch >= 1, "Invalid epoch")

    if epoch == 1:
        factor = 0.2
    elif epoch == 2:
        factor = 0.6
    else:
        factor = 1.0

    return LAMBDA_MAX * factor


def atomic_json(path: Path, value: Mapping) -> None:
    require(
        not path.exists(),
        f"Refusing to overwrite existing JSON: {path}",
    )

    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")


def atomic_replace(path: Path, writer) -> None:
    temporary = path.with_name(path.name + ".tmp")

    require(
        not temporary.exists(),
        f"Unresolved temporary artifact: {temporary}",
    )

    try:
        writer(temporary)
        os.replace(temporary, path)
    except Exception:
        # Preserve temporary evidence after an interrupted write.
        raise


def write_metrics(path: Path, rows: list[dict]) -> None:
    write_metrics_atomic(path, rows, fields=METRIC_FIELDS, replace=atomic_replace)


def save_checkpoint(path: Path, record: dict) -> None:
    atomic_replace(
        path,
        lambda temporary: torch.save(record, temporary),
    )


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(record: Mapping) -> None:
    random.setstate(record["python"])
    np.random.set_state(record["numpy"])
    torch.set_rng_state(record["torch_cpu"])
    torch.cuda.set_rng_state_all(record["torch_cuda"])


def check_frozen_baselines() -> dict:
    b0_dir = ROOT / "results/runs/b0_dev_s42"
    a3_dir = ROOT / "results/runs/a3_l100_s42"

    b0_config = read_json(b0_dir / "config.json")
    b0_summary = read_json(b0_dir / "summary.json")
    a3_config = read_json(a3_dir / "config.json")

    check_b0_recipe(
        b0_config,
        b0_summary,
        dataset_sha=sha256(ROOT / "code/dataset.py"),
        model_sha=sha256(ROOT / "code/model.py"),
    )

    require(
        sha256(b0_dir / "best.pt") == b0_summary["checkpoint_sha256"],
        "B0 checkpoint provenance mismatch",
    )

    require(
        a3_config["run_id"] == "a3_l100_s42",
        "Unexpected A3 configuration",
    )

    require(
        a3_config["seed"] == SEED
        and a3_config["batch"] == BATCH
        and a3_config["lambda_max"] == LAMBDA_MAX,
        "A3 seed, batch or lambda mismatch",
    )

    require(
        a3_config["split_sha256"] == b0_config["split_sha256"],
        "A3/B0 split mismatch",
    )

    require(
        a3_config["test_images_opened"] is False and b0_config["test_images_opened"] is False,
        "Baseline TEST seal mismatch",
    )

    weights = imagenet_weight_path()

    require(
        weights.is_file() and sha256(weights) == b0_config["pretrained_sha256"],
        "ImageNet initialization weights mismatch",
    )

    return {
        "b0_config": b0_config,
        "a3_config": a3_config,
        "b0_checkpoint_sha256": b0_summary["checkpoint_sha256"],
        "pretrained_sha256": b0_config["pretrained_sha256"],
    }


def read_train_boxes(
    train_rows: Sequence[Mapping[str, str]],
) -> dict[str, tuple[tuple[str, Rect], ...]]:
    """Read all TRAIN expert boxes without loading any TEST pixels."""
    path = ROOT / "data/processed/boxes.csv"

    require(
        sha256(path) == EXPECTED_BOXES,
        "Frozen P0 bounding-box SHA mismatch",
    )

    train_by_id = {row["image_id"]: row for row in train_rows}

    result = defaultdict(list)
    annotations = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]

            if image_id not in train_by_id:
                continue

            expected = train_by_id[image_id]

            require(
                expected["target"] == "1",
                "Negative TRAIN image has lesion boxes",
            )

            require(
                row["patient_id"] == expected["patient_id"]
                and row["sop_uid"] == expected["sop_uid"],
                "TRAIN box identity mismatch",
            )

            annotation_id = row["annotation_id"]

            require(
                annotation_id not in annotations,
                "Duplicate TRAIN annotation_id",
            )

            annotations.add(annotation_id)

            result[image_id].append((annotation_id, rect_from_row(row)))

    require(
        len(annotations) == TRAIN_LESIONS,
        "Unexpected TRAIN expert lesion count",
    )

    positives = [row for row in train_rows if row["target"] == "1"]

    require(
        len(positives) == 5685,
        "Unexpected positive TRAIN count",
    )

    for row in train_rows:
        image_id = row["image_id"]
        current = result.get(image_id, [])

        require(
            len(current) == int(row["num_boxes"]),
            f"TRAIN box count mismatch: {image_id}",
        )

    return {
        image_id: tuple(sorted(items, key=lambda item: item[0]))
        for image_id, items in result.items()
    }


def build_geometry_index(
    train_rows: Sequence[Mapping[str, str]],
    boxes: Mapping[str, tuple],
) -> tuple[dict, list[dict], dict]:
    """Generate A2 controls using TRAIN geometry only."""
    index = {}
    records = []

    patient_by_id = {row["image_id"]: row["patient_id"] for row in train_rows}

    k_distribution = Counter()

    for image_id in sorted(boxes):
        lesion_items = boxes[image_id]

        all_lesions = [rect for _, rect in lesion_items]

        matched = []

        for annotation_id, lesion in lesion_items:
            controls = sample_geometry_controls(
                image_id,
                annotation_id,
                lesion,
                all_lesions,
            )

            k = len(controls)
            k_distribution[k] += 1

            if not controls:
                continue

            matched.append(
                (
                    annotation_id,
                    lesion,
                    tuple(rect for _, rect in controls),
                )
            )

            for control_index, (
                tier,
                control,
            ) in enumerate(controls, start=1):
                records.append(
                    {
                        "image_id": image_id,
                        "patient_id": patient_by_id[image_id],
                        "annotation_id": annotation_id,
                        "control_index": control_index,
                        "tier": tier,
                        "lesion_x0": lesion.x0,
                        "lesion_y0": lesion.y0,
                        "lesion_x1": lesion.x1,
                        "lesion_y1": lesion.y1,
                        "control_x0": control.x0,
                        "control_y0": control.y0,
                        "control_x1": control.x1,
                        "control_y1": control.y1,
                    }
                )

        if matched:
            index[image_id] = tuple(matched)

    require(
        sum(k_distribution.values()) == TRAIN_LESIONS,
        "A2 geometry sampling did not cover every lesion",
    )

    require(
        sum(k * count for k, count in k_distribution.items()) == len(records),
        "A2 geometry control count inconsistency",
    )

    require(
        len(index) > 0,
        "A2 geometry produced no eligible TRAIN images",
    )

    # Read-only P3 reference: not used to create A2 controls.
    frozen_a3 = load_frozen_controls(ROOT)

    a2_annotations = {annotation_id for groups in index.values() for annotation_id, _, _ in groups}

    a3_annotations = {
        group.annotation_id
        for image_id in frozen_a3.by_image
        for group in frozen_a3.by_image[image_id]
    }

    summary = {
        "algorithm": "geometry_only_image_halves",
        "lung_masks_used": False,
        "pspnet_used": False,
        "seed": SEED,
        "step": STEP,
        "budget_per_tier": BUDGET,
        "k_max": K_MAX,
        "vertical_distance_max_image_fraction": VERTICAL_DISTANCE,
        "pairwise_control_iou_max": PAIR_IOU_MAX,
        "train_lesions": TRAIN_LESIONS,
        "eligible_images": len(index),
        "eligible_lesions": len(a2_annotations),
        "controls": len(records),
        "k_distribution": {str(k): k_distribution[k] for k in range(K_MAX + 1)},
        "a3_train_eligible_lesions": frozen_a3.eligible_lesions,
        "common_a2_a3_eligible_lesions": len(a2_annotations & a3_annotations),
        "comparison_warning": "A2 and A3 may have different eligible sets. "
        "Regional comparisons must report common eligibility.",
    }

    return index, records, summary


def prepare_experiment(variant: str) -> dict:
    require(
        variant in RUN_IDS,
        "Variant must be b1 or a2",
    )

    train, val = load_frozen(ROOT)

    require(
        len(train) == TRAIN_COUNT,
        "Frozen TRAIN count changed",
    )

    require(
        len(val) == VAL_COUNT,
        "Frozen VAL count changed",
    )

    baseline = check_frozen_baselines()
    boxes = read_train_boxes(train)

    geometry_index = {}
    geometry_records = []
    geometry_summary = None
    geometry_text = ""

    if variant == "a2":
        (
            geometry_index,
            geometry_records,
            geometry_summary,
        ) = build_geometry_index(train, boxes)

        geometry_text = geometry_csv_text(geometry_records)

    return {
        "variant": variant,
        "run_id": RUN_IDS[variant],
        "train": train,
        "val": val,
        "boxes": boxes,
        "geometry_index": geometry_index,
        "geometry_summary": geometry_summary,
        "geometry_text": geometry_text,
        "geometry_sha256": (text_sha256(geometry_text) if variant == "a2" else None),
        "baseline": baseline,
        "source_sha256": source_hashes(),
    }


def make_loader(
    rows: Sequence[Mapping[str, str]],
    *,
    train: bool,
    epoch: int,
) -> DataLoader:
    base = RSNAImageDataset(
        ROOT,
        list(rows),
        train=train,
        seed=SEED,
        epoch=epoch,
    )

    if train:
        dataset = IdentifiedTrainDataset(
            base,
            rows,
        )
        generator_seed = SEED + epoch * 1009 + 1
    else:
        dataset = base
        generator_seed = SEED

    generator = torch.Generator().manual_seed(generator_seed)

    return DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=train,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def train_epoch(
    model: MobileNetV2OS16,
    experiment: Mapping,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    *,
    epoch: int,
    logger: logging.Logger,
    max_batches: int | None = None,
) -> dict:
    model.train()

    lam = lambda_at_epoch(epoch)

    totals = {
        "samples": 0,
        "steps": 0,
        "bce_sum": 0.0,
        "total_sum": 0.0,
        "aux_sum": 0.0,
        "active_batches": 0,
        "eligible_images": 0,
        "eligible_lesions": 0,
        "grad_norm_max": 0.0,
    }

    eligible_patients = set()

    patient_by_id = {row["image_id"]: row["patient_id"] for row in experiment["train"]}

    for batch_no, (images, targets, image_ids) in enumerate(
        loader,
        start=1,
    ):
        if max_batches is not None and batch_no > max_batches:
            break

        ids = list(image_ids)

        require(
            len(ids) == len(set(ids)),
            "Duplicate image in TRAIN batch",
        )

        for index, image_id in enumerate(ids):
            require(
                image_id in patient_by_id,
                "Non-TRAIN image encountered",
            )

            # Dataset identity has already been checked by
            # IdentifiedTrainDataset.
            require(
                targets[index].item() in (0.0, 1.0),
                "Unexpected binary target",
            )

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            features = model.forward_features(images)

            (
                total,
                image_bce,
                auxiliary,
                eligible_images,
                eligible_lesions,
            ) = compute_training_loss(
                experiment,
                ids,
                targets,
                features,
                model.classifier,
                lambda_value=lam,
            )

        total.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=CLIP_NORM,
            error_if_nonfinite=True,
        )

        require(
            bool(torch.isfinite(grad_norm)),
            "Non-finite gradient norm",
        )

        optimizer.step()
        scheduler.step()

        n = len(ids)

        totals["samples"] += n
        totals["steps"] += 1
        totals["bce_sum"] += image_bce.item() * n
        totals["total_sum"] += total.item() * n

        totals["grad_norm_max"] = max(
            totals["grad_norm_max"],
            float(grad_norm.item()),
        )

        if eligible_images:
            totals["active_batches"] += 1
            totals["aux_sum"] += auxiliary.item() * eligible_images

        totals["eligible_images"] += eligible_images
        totals["eligible_lesions"] += eligible_lesions

        source_index = (
            experiment["boxes"] if experiment["variant"] == "b1" else experiment["geometry_index"]
        )

        for image_id in ids:
            if source_index.get(image_id):
                eligible_patients.add(patient_by_id[image_id])

        if batch_no % 500 == 0:
            logger.info(
                "Epoch %d batch %d/%d BCE=%.6f active=%d",
                epoch,
                batch_no,
                len(loader),
                totals["bce_sum"] / totals["samples"],
                totals["active_batches"],
            )

    require(
        totals["samples"] > 0 and totals["steps"] > 0,
        "No TRAIN batches processed",
    )

    return {
        "samples_seen": totals["samples"],
        "train_bce": (totals["bce_sum"] / totals["samples"]),
        "train_aux_active_mean": (totals["aux_sum"] / max(totals["eligible_images"], 1)),
        "train_total": (totals["total_sum"] / totals["samples"]),
        "active_batches": totals["active_batches"],
        "total_batches": totals["steps"],
        "active_batch_rate": (totals["active_batches"] / totals["steps"]),
        "eligible_images_seen": totals["eligible_images"],
        "eligible_unique_patients": len(eligible_patients),
        "eligible_lesions_seen": totals["eligible_lesions"],
        "optimizer_steps": totals["steps"],
        "lr_last": optimizer.param_groups[0]["lr"],
        "grad_norm_max": totals["grad_norm_max"],
    }


@torch.inference_mode()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Match B0/A3 image-level FP32 validation."""
    model.eval()

    n = 0
    total_bce = 0.0
    targets = []
    logits = []

    for images, labels in loader:
        output = model(images.to(device, non_blocking=True)).float().cpu()

        require(
            bool(torch.isfinite(output).all()),
            "Non-finite VAL logits",
        )

        total_bce += float(
            F.binary_cross_entropy_with_logits(
                output,
                labels,
                reduction="sum",
            ).item()
        )

        n += len(labels)
        targets.extend(labels.tolist())
        logits.extend(output.tolist())

    require(
        n == VAL_COUNT,
        "Incomplete frozen VAL evaluation",
    )

    auroc, ap = binary_metrics(
        targets,
        logits,
    )

    return total_bce / n, auroc, ap


def create_model(
    experiment: Mapping,
    device: torch.device,
) -> MobileNetV2OS16:
    model = MobileNetV2OS16(pretrained=True).to(device)

    require(
        model.pretrained_sha256 == experiment["baseline"]["pretrained_sha256"],
        "ImageNet initialization provenance mismatch",
    )

    return model


def make_optimizer_and_scheduler(
    model: nn.Module,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=math.ceil(TRAIN_COUNT / BATCH) * MAX_EPOCHS,
    )

    return optimizer, scheduler


def make_config(experiment: Mapping) -> dict:
    return {
        "phase": "P7",
        "protocol": "BGREL_V3.2",
        "experiment": experiment["variant"].upper(),
        "run_id": experiment["run_id"],
        "seed": SEED,
        "batch": BATCH,
        "train_samples": TRAIN_COUNT,
        "val_samples": VAL_COUNT,
        "epochs_cap": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_improvement": MIN_IMPROVEMENT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "scheduler": ("CosineAnnealingLR_per_optimizer_step"),
        "amp_precision": "bf16",
        "grad_scaler_enabled": False,
        "augment_train": ("brightness_contrast_[0.9,1.1]"),
        "validation_augment": False,
        "lambda_max": LAMBDA_MAX,
        "lambda_schedule": ("epoch1_0.2_epoch2_0.6_epoch3+_1.0"),
        "lambda_selection": ("fixed_development_pilot_not_independently_tuned"),
        "backbone": "MobileNetV2_OS16",
        "shared_image_region_head": True,
        "loss": (
            "image_BCE_plus_box_positive_BCE"
            if experiment["variant"] == "b1"
            else "image_BCE_plus_geometry_listwise_REL"
        ),
        "box_source": "frozen_P0_expert_boxes_TRAIN",
        "control_source": (
            "NONE" if experiment["variant"] == "b1" else "geometry_only_image_halves"
        ),
        "lung_masks_used": False,
        "pspnet_inference_used": False,
        "source_sha256": experiment["source_sha256"],
        "boxes_sha256": EXPECTED_BOXES,
        "split_sha256": experiment["baseline"]["b0_config"]["split_sha256"],
        "master_sha256": experiment["baseline"]["b0_config"]["master_sha256"],
        "b0_checkpoint_sha256": experiment["baseline"]["b0_checkpoint_sha256"],
        "pretrained_sha256": experiment["baseline"]["pretrained_sha256"],
        "geometry_sha256": experiment["geometry_sha256"],
        "geometry_summary": experiment["geometry_summary"],
        "test_images_opened": False,
        "status": "DEVELOPMENT_ABLATION",
    }


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("p7_" + path.parent.name)

    require(
        not logger.handlers,
        "Logger already initialized",
    )

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)

    file_handler = logging.FileHandler(
        path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream)
    logger.addHandler(file_handler)

    return logger


def preflight(experiment: Mapping) -> None:
    config = make_config(experiment)

    print("P7 PREFLIGHT")
    print("Variant:", experiment["variant"])
    print("Run ID:", experiment["run_id"])
    print("TRAIN:", len(experiment["train"]))
    print("VAL:", len(experiment["val"]))
    print(
        "Expert TRAIN lesions:",
        sum(len(items) for items in experiment["boxes"].values()),
    )
    print("Lambda max:", LAMBDA_MAX)
    print("Split SHA:", config["split_sha256"])
    print(
        "TEST pixels opened:",
        config["test_images_opened"],
    )

    if experiment["variant"] == "a2":
        print(
            json.dumps(
                experiment["geometry_summary"],
                indent=2,
                ensure_ascii=False,
            )
        )

    print("P7_PREFLIGHT_PASS")


def smoke(experiment: Mapping) -> None:
    """Two optimizer batches and one VAL batch; no file writes."""
    device = require_cuda()
    seed_everything()

    source_index = (
        experiment["boxes"] if experiment["variant"] == "b1" else experiment["geometry_index"]
    )

    selected = []
    used_patients = set()

    positives = [
        row
        for row in experiment["train"]
        if row["target"] == "1" and source_index.get(row["image_id"])
    ]

    negatives = [row for row in experiment["train"] if row["target"] == "0"]

    for group, required_count in (
        (positives, 4),
        (negatives, 4),
    ):
        count = 0

        for row in group:
            if row["patient_id"] in used_patients:
                continue

            selected.append(row)
            used_patients.add(row["patient_id"])
            count += 1

            if count == required_count:
                break

        require(
            count == required_count,
            "Insufficient independent TRAIN smoke samples",
        )

    model = create_model(
        experiment,
        device,
    )

    optimizer, scheduler = make_optimizer_and_scheduler(model)

    logger = logging.getLogger("p7_smoke")
    logger.addHandler(logging.NullHandler())

    result = train_epoch(
        model,
        experiment,
        make_loader(
            selected,
            train=True,
            epoch=1,
        ),
        optimizer,
        scheduler,
        device,
        epoch=1,
        logger=logger,
        max_batches=2,
    )

    require(
        result["optimizer_steps"] == 2,
        "Smoke did not perform two optimizer steps",
    )

    require(
        result["eligible_lesions_seen"] > 0,
        "Smoke did not activate the auxiliary objective",
    )

    # Only four VAL images in smoke mode.
    validation = make_loader(
        experiment["val"][:BATCH],
        train=False,
        epoch=0,
    )

    model.eval()

    with torch.inference_mode():
        images, labels = next(iter(validation))
        output = model(images.to(device)).float().cpu()

    require(
        tuple(output.shape) == tuple(labels.shape) and bool(torch.isfinite(output).all()),
        "Smoke VAL inference failed",
    )

    print(
        "P7_SMOKE_PASS",
        experiment["variant"],
        "optimizer_steps=2",
        "eligible_lesions=",
        result["eligible_lesions_seen"],
    )


def run_experiment(
    experiment: Mapping,
    *,
    resume: bool,
) -> None:
    device = require_cuda()
    seed_everything()

    config = make_config(experiment)

    run_dir = ROOT / "results/runs" / experiment["run_id"]

    if resume:
        require(
            run_dir.is_dir(),
            "Resume requested but run folder does not exist",
        )

        require(
            not (run_dir / "summary.json").exists(),
            "Completed run cannot be resumed",
        )

        require(
            read_json(run_dir / "config.json") == config,
            "Resume configuration mismatch",
        )

    else:
        require(
            not run_dir.exists(),
            f"Run already exists: {run_dir}",
        )

    model = create_model(
        experiment,
        device,
    )

    optimizer, scheduler = make_optimizer_and_scheduler(model)

    start_epoch = 1
    rows = []
    best_auc = -math.inf
    best_epoch = 0
    best_patience_auc = -math.inf
    patience = 0

    if resume:
        checkpoint = torch.load(
            run_dir / "last.pt",
            map_location=device,
            weights_only=False,
        )

        require(
            checkpoint["config"] == config,
            "Checkpoint configuration mismatch",
        )

        require(
            not checkpoint.get("training_complete", False),
            "Checkpoint already finished",
        )

        start_epoch = int(checkpoint["epoch"]) + 1

        require(
            1 < start_epoch <= MAX_EPOCHS,
            "Invalid resume epoch",
        )

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        optimizer.load_state_dict(checkpoint["optimizer"])

        scheduler.load_state_dict(checkpoint["scheduler"])

        rows = checkpoint["metrics"]

        require(
            len(rows) == start_epoch - 1,
            "Checkpoint history incomplete",
        )

        with (run_dir / "metrics.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            saved_rows = list(csv.DictReader(handle))

        require(
            len(saved_rows) == len(rows),
            "Metrics history does not match checkpoint",
        )

        for saved, original in zip(saved_rows, rows):
            expected = {
                key: ("" if value is None else str(value)) for key, value in original.items()
            }

            require(
                saved == expected,
                "Resume metrics row mismatch",
            )

        best_auc = float(checkpoint["best_auc"])
        best_epoch = int(checkpoint["best_epoch"])
        best_patience_auc = float(checkpoint["best_patience_auc"])
        patience = int(checkpoint["patience"])

        require(
            patience < PATIENCE,
            "Already early-stopped; cannot resume",
        )

        restore_rng(checkpoint["rng"])

    else:
        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        atomic_json(
            run_dir / "config.json",
            config,
        )

        if experiment["variant"] == "a2":
            geometry_path = run_dir / "geometry_controls.csv"

            with geometry_path.open(
                "x",
                encoding="utf-8",
                newline="",
            ) as handle:
                handle.write(experiment["geometry_text"])

            require(
                sha256(geometry_path) == experiment["geometry_sha256"],
                "Frozen A2 controls SHA mismatch",
            )

    if experiment["variant"] == "a2":
        geometry_path = run_dir / "geometry_controls.csv"

        require(
            sha256(geometry_path) == experiment["geometry_sha256"],
            "A2 geometry controls changed",
        )

    logger = make_logger(run_dir / "train.log")

    expected_eligible_lesions = (
        TRAIN_LESIONS
        if experiment["variant"] == "b1"
        else experiment["geometry_summary"]["eligible_lesions"]
    )

    expected_eligible_images = (
        len(experiment["boxes"])
        if experiment["variant"] == "b1"
        else experiment["geometry_summary"]["eligible_images"]
    )

    steps_per_epoch = math.ceil(TRAIN_COUNT / BATCH)

    try:
        logger.info(
            "P7 START variant=%s run=%s resume=%s",
            experiment["variant"],
            experiment["run_id"],
            resume,
        )

        for epoch in range(
            start_epoch,
            MAX_EPOCHS + 1,
        ):
            start_time = time.monotonic()

            torch.cuda.reset_peak_memory_stats()

            loader = make_loader(
                experiment["train"],
                train=True,
                epoch=epoch,
            )

            result = train_epoch(
                model,
                experiment,
                loader,
                optimizer,
                scheduler,
                device,
                epoch=epoch,
                logger=logger,
            )

            require(
                result["samples_seen"] == TRAIN_COUNT
                and result["optimizer_steps"] == steps_per_epoch,
                "Incomplete TRAIN epoch",
            )

            require(
                result["eligible_images_seen"] == expected_eligible_images,
                "Eligible TRAIN image coverage mismatch",
            )

            require(
                result["eligible_lesions_seen"] == expected_eligible_lesions,
                "Eligible TRAIN lesion coverage mismatch",
            )

            val_bce, val_auc, val_ap = validate_epoch(
                model,
                make_loader(
                    experiment["val"],
                    train=False,
                    epoch=0,
                ),
                device,
            )

            if val_auc > best_auc + 1e-12:
                best_auc = val_auc
                best_epoch = epoch

                save_checkpoint(
                    run_dir / "best.pt",
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_auc": val_auc,
                        "val_ap": val_ap,
                        "config": config,
                    },
                )

            if val_auc > best_patience_auc + MIN_IMPROVEMENT:
                best_patience_auc = val_auc
                patience = 0
            else:
                patience += 1

            metric = {
                "epoch": epoch,
                "train_bce": result["train_bce"],
                "train_aux_active_mean": result["train_aux_active_mean"],
                "train_total": result["train_total"],
                "val_bce": val_bce,
                "val_auroc": val_auc,
                "val_ap": val_ap,
                "lambda_epoch": lambda_at_epoch(epoch),
                "active_batches": result["active_batches"],
                "total_batches": result["total_batches"],
                "active_batch_rate": result["active_batch_rate"],
                "eligible_images_seen": result["eligible_images_seen"],
                "eligible_unique_patients": result["eligible_unique_patients"],
                "eligible_lesions_seen": result["eligible_lesions_seen"],
                "optimizer_steps": result["optimizer_steps"],
                "lr_last": result["lr_last"],
                "grad_norm_max": result["grad_norm_max"],
                "gpu_peak_mib": (torch.cuda.max_memory_allocated() / 1024**2),
                "best_epoch": best_epoch,
                "best_val_auroc": best_auc,
                "patience_counter": patience,
                "epoch_seconds": (time.monotonic() - start_time),
            }

            require(
                set(metric) == set(METRIC_FIELDS),
                "Unexpected metric schema",
            )

            rows.append(metric)

            last_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
                "best_epoch": best_epoch,
                "best_patience_auc": best_patience_auc,
                "patience": patience,
                "metrics": rows,
                "rng": capture_rng(),
                "config": config,
                "training_complete": (patience >= PATIENCE or epoch == MAX_EPOCHS),
            }

            save_checkpoint(
                run_dir / "last.pt",
                last_state,
            )

            write_metrics(
                run_dir / "metrics.csv",
                rows,
            )

            logger.info(
                "EPOCH %d BCE=%.6f AUX=%.6f " "VAL_AUC=%.6f VAL_AP=%.6f " "BEST=%d PATIENCE=%d",
                epoch,
                result["train_bce"],
                result["train_aux_active_mean"],
                val_auc,
                val_ap,
                best_epoch,
                patience,
            )

            if patience >= PATIENCE:
                logger.info(
                    "EARLY STOP epoch=%d",
                    epoch,
                )
                break

        summary = {
            "phase": "P7",
            "status": "FINISHED",
            "variant": experiment["variant"],
            "run_id": experiment["run_id"],
            "best_epoch": best_epoch,
            "best_val_auroc": best_auc,
            "last_epoch": epoch,
            "checkpoint_sha256": sha256(run_dir / "best.pt"),
            "metrics_sha256": sha256(run_dir / "metrics.csv"),
            "source_sha256": experiment["source_sha256"],
            "geometry_sha256": experiment["geometry_sha256"],
            "geometry_summary": experiment["geometry_summary"],
            "test_images_opened": False,
            "limitations": [
                "One training seed only",
                "Auxiliary coefficient fixed at 0.10 " "rather than independently optimized",
                "A2 controls are geometry matched " "but not certified healthy tissue",
                "A2 and A3 can have different " "eligible lesion populations",
            ],
        }

        atomic_json(
            run_dir / "summary.json",
            summary,
        )

        logger.info(
            "P7_ABLATION_PASS variant=%s best_epoch=%d " "best_val_auroc=%.6f",
            experiment["variant"],
            best_epoch,
            best_auc,
        )

        print(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
        )

    finally:
        close_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--variant",
        required=True,
        choices=("b1", "a2"),
    )

    modes = parser.add_mutually_exclusive_group(required=True)

    modes.add_argument(
        "--preflight",
        action="store_true",
    )

    modes.add_argument(
        "--smoke",
        action="store_true",
    )

    modes.add_argument(
        "--train",
        action="store_true",
    )

    modes.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    experiment = prepare_experiment(args.variant)

    if args.preflight:
        preflight(experiment)

    elif args.smoke:
        smoke(experiment)

    elif args.train:
        run_experiment(
            experiment,
            resume=False,
        )

    else:
        run_experiment(
            experiment,
            resume=True,
        )


if __name__ == "__main__":
    main()
