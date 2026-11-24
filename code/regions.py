"""Fractional area-weighted ROI pooling for BG-REL V3.2.

Geometry uses continuous, half-open XYXY boxes in classifier input pixels.
PSPNet masks are NOT consumed by this module; P3 fixes control coordinates.
There is no separate regional classifier head.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _size(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must be (width, height)")
    w, h = value
    if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
        raise ValueError(f"{name} must contain positive integer dimensions")
    return (w, h)


def _check_boxes(boxes: Tensor, size: tuple[int, int]) -> None:
    """Reject rather than silently clip malformed continuous XYXY boxes."""
    if not isinstance(boxes, Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be a tensor of shape [R, 4]")
    if boxes.dtype not in (torch.float32, torch.float64):
        raise ValueError("boxes must be float32 or float64 continuous coordinates")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("non-finite box coordinate")
    x0, y0, x1, y1 = boxes.unbind(-1)
    w, h = size
    valid = (x0 >= 0) & (y0 >= 0) & (x1 <= w) & (y1 <= h) & (x1 > x0) & (y1 > y0)
    if not bool(valid.all()):
        raise ValueError("invalid, zero-area, or out-of-bounds XYXY box")


def scale_boxes_xyxy(
    boxes_xyxy: Tensor,
    *,
    source_size: tuple[int, int] = (1024, 1024),
    input_size: tuple[int, int] = (224, 224),
) -> Tensor:
    """Map P3 XYXY boxes into input coordinates without rounding or clipping.

    Size pairs mean (width, height), not (height, width). The mapping is
    continuous; a later region pooling call handles fractional cell coverage.
    """
    src = _size(source_size, "source_size")
    dst = _size(input_size, "input_size")
    _check_boxes(boxes_xyxy, src)
    factors = boxes_xyxy.new_tensor([dst[0] / src[0], dst[1] / src[1]]).repeat(2)
    scaled = boxes_xyxy * factors
    _check_boxes(scaled, dst)
    return scaled


def fractional_roi_pool(
    feature_maps: Tensor,
    boxes_xyxy: Tensor,
    batch_indices: Tensor,
    *,
    input_size: tuple[int, int] = (224, 224),
) -> Tensor:
    """Pool each ROI with exact fractional intersections on the feature grid.

    Args:
      feature_maps: [B, C, Hf, Wf] feature activations, on CPU or CUDA.
      boxes_xyxy: [R, 4] continuous half-open INPUT-pixel coordinates.
      batch_indices: [R] zero-based image index for each ROI.
      input_size: (input_width, input_height), normally (224, 224).
    Returns:
      [R, C] descriptors in float32 for AMP-stable subsequent linear scores.

    No nearest-neighbor sampling, mask interpolation, or area rounding occurs.
    R=0 is allowed and yields a correctly shaped empty descriptor tensor.
    """
    size = _size(input_size, "input_size")
    if not isinstance(feature_maps, Tensor) or feature_maps.ndim != 4:
        raise ValueError("feature_maps must be [B, C, Hf, Wf]")
    b, c, hf, wf = feature_maps.shape
    if min(b, c, hf, wf) <= 0 or not feature_maps.is_floating_point():
        raise ValueError("feature_maps dimensions/dtype invalid")
    if not bool(torch.isfinite(feature_maps).all()):
        raise ValueError("feature_maps contain non-finite values")
    _check_boxes(boxes_xyxy, size)
    r = boxes_xyxy.shape[0]
    if (not isinstance(batch_indices, Tensor) or batch_indices.ndim != 1
            or batch_indices.shape[0] != r or batch_indices.dtype != torch.int64):
        raise ValueError("batch_indices must be int64 shape [R]")
    if batch_indices.device != feature_maps.device or boxes_xyxy.device != feature_maps.device:
        raise ValueError("feature_maps, boxes and indices must share a device")
    if r == 0:
        return feature_maps.new_empty((0, c), dtype=torch.float32)
    if not bool(((batch_indices >= 0) & (batch_indices < b)).all()):
        raise ValueError("batch index out of range")

    # Compute intersection areas in float64: tiny boxes and subcell boundaries
    # retain geometric precision, even during BF16 autocast of the backbone.
    coords = boxes_xyxy.to(torch.float64)
    x0, y0, x1, y1 = coords.unbind(-1)
    x0, x1 = x0 * (wf / size[0]), x1 * (wf / size[0])
    y0, y1 = y0 * (hf / size[1]), y1 * (hf / size[1])
    grid_x = torch.arange(wf, dtype=torch.float64, device=coords.device)[None, :]
    grid_y = torch.arange(hf, dtype=torch.float64, device=coords.device)[None, :]
    overlap_x = (torch.minimum(x1[:, None], grid_x + 1) - torch.maximum(x0[:, None], grid_x)).clamp_min(0)
    overlap_y = (torch.minimum(y1[:, None], grid_y + 1) - torch.maximum(y0[:, None], grid_y)).clamp_min(0)
    areas = overlap_y[:, :, None] * overlap_x[:, None, :]
    totals = areas.sum(dim=(1, 2), keepdim=True)
    if not bool(torch.isfinite(totals).all()) or not bool((totals > 0).all()):
        raise ValueError("non-positive fractional ROI intersection area")
    weights = (areas / totals).to(torch.float32)
    # Feature activations remain differentiable; geometry is constant.
    pooled = torch.einsum("rchw,rhw->rc", feature_maps[batch_indices].float(), weights)
    return pooled


def shared_region_scores(
    feature_maps: Tensor,
    classifier: nn.Linear,
    boxes_xyxy: Tensor,
    batch_indices: Tensor,
    *,
    input_size: tuple[int, int] = (224, 224),
) -> Tensor:
    """Score pooled regions with THE SAME affine head as image logits."""
    if not isinstance(classifier, nn.Linear) or classifier.out_features != 1:
        raise ValueError("classifier must be the shared single-logit nn.Linear head")
    if classifier.in_features != feature_maps.shape[1]:
        raise ValueError("classifier dimension incompatible with feature maps")
    descriptors = fractional_roi_pool(feature_maps, boxes_xyxy, batch_indices, input_size=input_size)
    return classifier(descriptors).flatten()
