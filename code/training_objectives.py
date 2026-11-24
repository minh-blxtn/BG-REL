"""B1 positive-box and optional A2 objectives, separate from training I/O.

Functions are extracted unchanged from the uploaded R3 trainer. A2 is an
unreported extension; it is not a completed paper experiment.
"""

from __future__ import annotations
from typing import Mapping, Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from losses import bce_rel_loss
from regions import scale_boxes_xyxy, shared_region_scores
from sample_controls import Rect
from refactor_helpers import require

SIZE = 1024


def collect_region_scores(
    feature_maps: Tensor,
    classifier: nn.Linear,
    boxes: Sequence[Rect],
    owners: Sequence[int],
) -> Tensor:
    if not boxes:
        return feature_maps.new_empty(
            (0,),
            dtype=torch.float32,
        )

    source = torch.tensor(
        [[box.x0, box.y0, box.x1, box.y1] for box in boxes],
        dtype=torch.float32,
        device=feature_maps.device,
    )

    scaled = scale_boxes_xyxy(
        source,
        source_size=(SIZE, SIZE),
        input_size=(224, 224),
    )

    indices = torch.tensor(
        owners,
        dtype=torch.int64,
        device=feature_maps.device,
    )

    return shared_region_scores(
        feature_maps,
        classifier,
        scaled,
        indices,
        input_size=(224, 224),
    )


def compute_training_loss(
    experiment: Mapping,
    image_ids: Sequence[str],
    targets: Tensor,
    feature_maps: Tensor,
    classifier: nn.Linear,
    *,
    lambda_value: float,
) -> tuple[Tensor, Tensor, Tensor, int, int]:
    """Image BCE plus image-balanced B1 or A2 auxiliary objective."""
    image_logits = classifier(feature_maps.mean(dim=(2, 3))).flatten()

    require(
        tuple(image_logits.shape) == tuple(targets.shape),
        "Image logit/target shape mismatch",
    )

    image_bce = F.binary_cross_entropy_with_logits(
        image_logits.float(),
        targets.float(),
    )

    boxes = []
    owners = []
    layout = []

    if experiment["variant"] == "b1":
        source_index = experiment["boxes"]

        for batch_index, image_id in enumerate(image_ids):
            lesions = source_index.get(image_id, ())

            if lesions:
                require(
                    targets[batch_index].item() == 1,
                    "B1 lesion on negative image",
                )

            start = len(boxes)

            for _, lesion in lesions:
                boxes.append(lesion)
                owners.append(batch_index)

            if lesions:
                layout.append((batch_index, start, len(lesions)))

        scores = collect_region_scores(
            feature_maps,
            classifier,
            boxes,
            owners,
        )

        per_image = []

        for _, start, count in layout:
            lesion_scores = scores[start : start + count]

            positive_labels = torch.ones_like(lesion_scores)

            lesion_losses = F.binary_cross_entropy_with_logits(
                lesion_scores.float(),
                positive_labels.float(),
                reduction="none",
            )

            per_image.append(lesion_losses.mean())

        if per_image:
            auxiliary = torch.stack(per_image).mean()
        else:
            auxiliary = image_bce * 0.0

        total = image_bce + lambda_value * auxiliary

        eligible_images = len(per_image)
        eligible_lesions = sum(count for _, _, count in layout)

    else:
        source_index = experiment["geometry_index"]
        pairs_by_image = [[] for _ in image_ids]

        for batch_index, image_id in enumerate(image_ids):
            groups = source_index.get(image_id, ())

            if groups:
                require(
                    targets[batch_index].item() == 1,
                    "A2 controls on negative image",
                )

            for _, lesion, controls in groups:
                start = len(boxes)

                boxes.append(lesion)
                boxes.extend(controls)

                owners.extend([batch_index] * (1 + len(controls)))

                layout.append(
                    (
                        batch_index,
                        start,
                        len(controls),
                    )
                )

        scores = collect_region_scores(
            feature_maps,
            classifier,
            boxes,
            owners,
        )

        for batch_index, start, k in layout:
            pairs_by_image[batch_index].append(
                (
                    scores[start],
                    scores[start + 1 : start + 1 + k],
                )
            )

        parts = bce_rel_loss(
            image_logits,
            targets,
            pairs_by_image,
            lambda_value=lambda_value,
        )

        total = parts.total
        image_bce = parts.bce
        auxiliary = parts.rel
        eligible_images = parts.eligible_images
        eligible_lesions = parts.eligible_lesions

    require(
        bool(torch.isfinite(total))
        and bool(torch.isfinite(image_bce))
        and bool(torch.isfinite(auxiliary)),
        "Non-finite P7 objective",
    )

    return (
        total,
        image_bce,
        auxiliary,
        eligible_images,
        eligible_lesions,
    )


import csv
import hashlib
import io
import math
from sample_controls import stable_jitter, intersect, iou

SEED = 42

STEP = 8

BUDGET = 128

K_MAX = 2

VERTICAL_DISTANCE = 0.10

PAIR_IOU_MAX = 0.20

EPS = 1e-6

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


def candidate_axis(
    low: float,
    high: float,
    jitter: int,
    step: int,
    anchor: float,
) -> list[float]:
    if high < low:
        return []

    start = math.ceil((low - jitter) / step) * step + jitter

    result = []
    current = float(start)

    while current <= high + EPS:
        result.append(current)
        current += step

    result.append(float(min(max(anchor, low), high)))

    return sorted(set(result))


def geometry_candidates(
    lesion: Rect,
    image_id: str,
    annotation_id: str,
    tier: int,
) -> list[Rect]:
    """Build deterministic candidate regions without lung masks.

    Tier 1: same image half as lesion.
    Tier 2: opposite image half.

    Candidates preserve box dimensions, remain in the frame,
    and stay within 10% image-height center distance.

    This is image-coordinate matching, NOT anatomical matching.
    """
    require(tier in (1, 2), "Invalid geometry tier")

    lesion_left = lesion.cx < SIZE / 2

    if tier == 1:
        left_half = lesion_left
    else:
        left_half = not lesion_left

    half_start = 0.0 if left_half else SIZE / 2
    half_end = SIZE / 2 if left_half else float(SIZE)

    if lesion.w > SIZE / 2:
        return []

    low_x = half_start
    high_x = half_end - lesion.w

    low_y = max(
        0.0,
        lesion.cy - VERTICAL_DISTANCE * SIZE - lesion.h / 2,
    )

    high_y = min(
        SIZE - lesion.h,
        lesion.cy + VERTICAL_DISTANCE * SIZE - lesion.h / 2,
    )

    if high_x < low_x or high_y < low_y:
        return []

    if tier == 1:
        # Mirror position within the same image half.
        target_cx = 512 - lesion.cx if lesion_left else 1536 - lesion.cx
    else:
        # Mirror position across the image midline.
        target_cx = SIZE - lesion.cx

    target_x = target_cx - lesion.w / 2
    target_y = lesion.cy - lesion.h / 2

    jx, jy = stable_jitter(
        image_id,
        annotation_id,
        SEED,
    )

    xs = candidate_axis(
        low_x,
        high_x,
        jx % STEP,
        STEP,
        target_x,
    )

    ys = candidate_axis(
        low_y,
        high_y,
        jy % STEP,
        STEP,
        target_y,
    )

    pairs = [(x, y) for y in ys for x in xs]

    pairs.sort(
        key=lambda xy: (
            round(
                abs(xy[1] + lesion.h / 2 - lesion.cy) / STEP * 1.3
                + abs(xy[0] + lesion.w / 2 - target_cx) / STEP * 0.08,
                8,
            ),
            abs(xy[1] + lesion.h / 2 - lesion.cy),
            abs(xy[0] + lesion.w / 2 - target_cx),
            xy[1],
            xy[0],
        )
    )

    return [
        Rect(
            x,
            y,
            x + lesion.w,
            y + lesion.h,
        )
        for x, y in pairs[:BUDGET]
    ]


def sample_geometry_controls(
    image_id: str,
    annotation_id: str,
    lesion: Rect,
    all_lesions: Sequence[Rect],
) -> tuple[tuple[int, Rect], ...]:
    """Select up to two deterministic non-lesion controls."""
    accepted = []

    for tier in (1, 2):
        if len(accepted) >= K_MAX:
            break

        candidates = geometry_candidates(
            lesion,
            image_id,
            annotation_id,
            tier,
        )

        for candidate in candidates:
            if not candidate.valid():
                continue

            if abs(candidate.area - lesion.area) > EPS:
                continue

            if any(intersect(candidate, other) > EPS for other in all_lesions):
                continue

            max_iou = max(
                (iou(candidate, item[1]) for item in accepted),
                default=0.0,
            )

            if max_iou > PAIR_IOU_MAX + EPS:
                continue

            accepted.append((tier, candidate))

            if len(accepted) >= K_MAX:
                break

    return tuple(accepted)


def geometry_csv_text(records: Sequence[dict]) -> str:
    output = io.StringIO(newline="")

    writer = csv.DictWriter(
        output,
        fieldnames=GEOMETRY_FIELDS,
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(records)

    return output.getvalue()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
