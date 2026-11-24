"""Stable listwise BG-REL objective, averaged per lesion then eligible image.

This file does not train models, read DICOMs, or alter control selection.
K=0 is always excluded from REL while the image contributes to BCE.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

# One (lesion score scalar, control scores vector) pair per expert lesion.
RegionPair = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class LossParts:
    total: Tensor
    bce: Tensor
    rel: Tensor
    eligible_images: int
    eligible_lesions: int


def listwise_rel_loss(lesion_score: Tensor, control_scores: Tensor) -> Tensor:
    """logsumexp([lesion, control_1, ...]) - lesion, for K >= 1."""
    if not isinstance(lesion_score, Tensor) or lesion_score.ndim != 0:
        raise ValueError("lesion_score must be a scalar tensor")
    if not isinstance(control_scores, Tensor) or control_scores.ndim != 1 or control_scores.numel() == 0:
        raise ValueError("control_scores must be 1-D with K >= 1")
    if lesion_score.device != control_scores.device or lesion_score.dtype != control_scores.dtype:
        raise ValueError("lesion and control scores must have the same device and dtype")
    if not lesion_score.is_floating_point() or not bool(torch.isfinite(lesion_score)) or not bool(torch.isfinite(control_scores).all()):
        raise ValueError("REL scores must be finite floating point")
    scores = torch.cat((lesion_score.reshape(1), control_scores), dim=0)
    return torch.logsumexp(scores.float(), dim=0) - scores[0].float()


def bce_rel_loss(
    image_logits: Tensor,
    targets: Tensor,
    pairs_by_image: Sequence[Sequence[RegionPair]],
    *,
    lambda_value: float,
) -> LossParts:
    """BCE mean over B + lambda * mean REL over eligible images.

    REL for an eligible image is mean loss over its eligible lesion pairs.
    Empty image entries and K=0 lesions do not contribute to REL. Input
    image/patient eligibility is fixed upstream; no controls are generated here.
    """
    if not isinstance(image_logits, Tensor) or image_logits.ndim != 1 or image_logits.numel() == 0:
        raise ValueError("image_logits must be a nonempty [B] tensor")
    if not isinstance(targets, Tensor) or targets.shape != image_logits.shape:
        raise ValueError("targets must have shape [B]")
    if targets.device != image_logits.device:
        raise ValueError("targets and logits must share a device")
    if len(pairs_by_image) != image_logits.numel():
        raise ValueError("pairs_by_image length must equal batch size")
    if not bool(torch.isfinite(image_logits).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("logits and targets must be finite")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("targets must be 0 or 1")
    if not isinstance(lambda_value, (float, int)) or not isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda_value must be nonnegative and finite")

    bce = F.binary_cross_entropy_with_logits(image_logits.float(), targets.float(), reduction="mean")
    image_rel = []
    eligible_lesions = 0
    for image_index, regions in enumerate(pairs_by_image):
        lesion_terms = []
        if targets[image_index].item() == 0 and any(c.numel() for _, c in regions):
            raise ValueError("negative images cannot contain eligible REL lesions")
        for lesion_score, control_scores in regions:
            if not isinstance(control_scores, Tensor) or control_scores.ndim != 1:
                raise ValueError("control_scores must be a 1-D tensor, including for K=0")
            if control_scores.numel() == 0:
                # K=0: do not create a fake negative or include a zero REL term.
                continue
            lesion_terms.append(listwise_rel_loss(lesion_score, control_scores))
        if lesion_terms:
            eligible_lesions += len(lesion_terms)
            image_rel.append(torch.stack(lesion_terms).mean())
    if image_rel:
        rel = torch.stack(image_rel).mean()
    else:
        # Exact zero, with a valid backward path when all images are inactive.
        rel = image_logits.float().sum() * 0.0
    total = bce + float(lambda_value) * rel
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("non-finite total loss")
    return LossParts(total=total, bce=bce, rel=rel,
                     eligible_images=len(image_rel), eligible_lesions=eligible_lesions)
