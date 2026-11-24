"""Read-only bridge from frozen P3 TRAIN control coordinates to P4 REL.

No resampling, PSPNet inference, DICOM decoding, data modification, or training.
This module accepts GEOMETRY-VERIFIED controls; anatomical normality is not certified.
"""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from losses import LossParts, bce_rel_loss
from regions import scale_boxes_xyxy, shared_region_scores

P3_CONTROLS_SHA = "59ac71c8c2268d55e448bc9b674999060bb2f828d0a3c1cf541bb856367f2e55"
P3_TRACE_SHA = "2932a24672f4c5db2fe8daf13fd2a54474732f4ab72f0ecc3b13d1d56b91654f"
FROZEN_SPLIT_SHA = "ccabcf432b0d266264e4ea96ca018b05a67d944af65fca4ad159f5130da7f9a7"
FROZEN_MASTER_SHA = "4c690ddf7918d0890b6715dba290be85a16959adc0dcaa70b5b4c5bd387d6c9b"
P0_BOXES_SHA = "6372a1facc3e1b78fbcd97a5bd103d2fc33ed257c92f8076b1d36e9168078e45"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_verified_csv(path: Path, expected_sha: str) -> list[dict[str, str]]:
    if not path.is_file() or _sha256(path) != expected_sha:
        raise ValueError(f"Frozen input missing or SHA-256 mismatch: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _box(row: Mapping[str, str], *, lesion: bool) -> tuple[float, float, float, float]:
    prefix = "lesion_" if lesion else ""
    vals = tuple(float(row[prefix + key]) for key in ("x0", "y0", "x1", "y1"))
    x0, y0, x1, y1 = vals
    if not all(math.isfinite(x) for x in vals) or not (0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 1024):
        raise ValueError(f"Malformed {prefix}box: {vals}")
    return vals


@dataclass(frozen=True)
class LesionControls:
    annotation_id: str
    lesion_xyxy: tuple[float, float, float, float]
    control_xyxy: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class FrozenControlIndex:
    by_image: Mapping[str, tuple[LesionControls, ...]]
    train_targets: Mapping[str, int]
    eligible_lesions: int
    eligible_images: int
    control_count: int
    k0_lesions: int
    # These controls are unannotated geometry matches, NOT verified healthy tissue.

    def for_image(self, image_id: str) -> tuple[LesionControls, ...]:
        if image_id not in self.train_targets:
            raise ValueError(f"Non-TRAIN or unknown image_id: {image_id}")
        return self.by_image.get(image_id, ())


def load_frozen_controls(root: Path) -> FrozenControlIndex:
    """Fail-closed full TRAIN metadata integrity checks; never opens TEST images."""
    root = Path(root)
    qa = root / "results" / "qa"
    dp = root / "data" / "processed"
    controls = _read_verified_csv(qa / "qa_controls.csv", P3_CONTROLS_SHA)
    traces = _read_verified_csv(qa / "sampler_trace.csv", P3_TRACE_SHA)
    split = _read_verified_csv(dp / "split.csv", FROZEN_SPLIT_SHA)
    master = _read_verified_csv(dp / "master_manifest.csv", FROZEN_MASTER_SHA)
    boxes = _read_verified_csv(dp / "boxes.csv", P0_BOXES_SHA)
    if (len(controls), len(traces), len(split), len(master), len(boxes)) != (3488, 9161, 29684, 29684, 11390):
        raise ValueError("Frozen input row count mismatch")
    split_by_id = {r["image_id"]: r for r in split}
    master_by_id = {r["image_id"]: r for r in master}
    box_by_ann = {r["annotation_id"]: r for r in boxes}
    if len(split_by_id) != len(split) or len(master_by_id) != len(master) or len(box_by_ann) != len(boxes):
        raise ValueError("Duplicated frozen identifiers")
    if {r["patient_id"] for r in split if r["split"] == "train"} & {r["patient_id"] for r in split if r["split"] != "train"}:
        raise ValueError("TRAIN patient leakage")
    train_targets = {}
    for sid, sp in split_by_id.items():
        if sid not in master_by_id or master_by_id[sid]["patient_id"] != sp["patient_id"]:
            raise ValueError("Split/master patient mismatch")
        if sp["split"] == "train":
            target = master_by_id[sid]["target"]
            if target not in ("0", "1"):
                raise ValueError("Invalid TRAIN target")
            train_targets[sid] = int(target)
    if len(train_targets) != 23748:
        raise ValueError("TRAIN size mismatch")
    trace_by_ann = {}
    for trace in traces:
        ann, sid = trace["annotation_id"], trace["image_id"]
        if ann in trace_by_ann or sid not in train_targets or train_targets[sid] != 1:
            raise ValueError("Duplicate annotation or non-positive/non-TRAIN trace")
        if (trace["patient_id"] != split_by_id[sid]["patient_id"] or
                ann not in box_by_ann or box_by_ann[ann]["image_id"] != sid):
            raise ValueError("Trace/P0 annotation mismatch")
        trace_by_ann[ann] = trace
    if len(trace_by_ann) != 9161:
        raise ValueError("Trace size mismatch")
    groups = defaultdict(list)
    control_ids = set()
    for r in controls:
        sid, ann, cid = r["image_id"], r["annotation_id"], r["control_id"]
        if cid in control_ids or ann not in trace_by_ann or sid != trace_by_ann[ann]["image_id"]:
            raise ValueError("Duplicate/orphan P3 control")
        control_ids.add(cid)
        if (r["patient_id"] != trace_by_ann[ann]["patient_id"] or
                r["geometry_valid"].upper() != "TRUE"):
            raise ValueError("P3 control patient/geometry status invalid")
        if r["tier"] not in ("1", "2") or r["side"] not in ("left", "right"):
            raise ValueError("P3 control tier/side invalid")
        occupancy = float(r["occupancy"])
        if not math.isfinite(occupancy) or occupancy < 0.80 - 1e-6 or occupancy > 1.0 + 1e-6:
            raise ValueError("P3 occupancy invalid")
        lesion = _box(r, lesion=True)
        bx = box_by_ann[ann]
        actual = (float(bx["x"]), float(bx["y"]), float(bx["x2"]), float(bx["y2"]))
        if any(abs(a - b) > 2e-4 for a, b in zip(lesion, actual)):
            raise ValueError("Control lesion coordinates differ from P0")
        groups[ann].append((r["control_id"], lesion, _box(r, lesion=False)))
    result = defaultdict(list)
    n_control = 0
    n_matched = 0
    for ann, trace in trace_by_ann.items():
        got = sorted(groups.get(ann, ()), key=lambda item: item[0])
        k = int(trace["k"])
        if k != len(got) or k not in (0, 1, 2):
            raise ValueError("Trace K does not match frozen P3 controls")
        if not got:
            continue
        lesion_box = _box(trace, lesion=True)
        if any(any(abs(a - b) > 2e-4 for a, b in zip(lesion_box, item[1])) for item in got):
            raise ValueError("Lesion boxes disagree across records")
        n_matched += 1
        n_control += len(got)
        result[trace["image_id"]].append(LesionControls(ann, lesion_box, tuple(item[2] for item in got)))
    if n_control != 3488 or n_matched != 2667 or len(result) != 2421:
        raise ValueError("Unexpected P3 eligible count")
    sorted_result = {k: tuple(sorted(v, key=lambda group: group.annotation_id)) for k, v in result.items()}
    return FrozenControlIndex(sorted_result, train_targets, n_matched, len(result), n_control, len(traces) - n_matched)


def pair_scores_from_frozen(
    index: FrozenControlIndex,
    image_ids: Sequence[str],
    targets: Tensor,
    feature_maps: Tensor,
    classifier: nn.Linear,
) -> list[list[tuple[Tensor, Tensor]]]:
    """Vectorized shared-head region scores, indexed by batch image, no resampling."""
    batch_size = len(image_ids)
    if feature_maps.ndim != 4 or feature_maps.shape[0] != batch_size:
        raise ValueError("Batch size / feature map mismatch")
    if targets.shape != (batch_size,) or len(set(image_ids)) != batch_size:
        raise ValueError("Batch targets invalid or duplicate image_ids")
    boxes = []
    owners = []
    layout = []
    for i, image_id in enumerate(image_ids):
        if image_id not in index.train_targets:
            raise ValueError("Cannot use unknown or non-TRAIN sample for REL")
        if targets[i].item() != index.train_targets[image_id]:
            raise ValueError("Batch label disagrees with frozen TRAIN")
        for group in index.for_image(image_id):
            start = len(boxes)
            boxes.append(group.lesion_xyxy)
            boxes.extend(group.control_xyxy)
            owners.extend([i] * (1 + len(group.control_xyxy)))
            layout.append((i, start, len(group.control_xyxy)))
    pairs: list[list[tuple[Tensor, Tensor]]] = [[] for _ in image_ids]
    if not boxes:
        return pairs
    # Use FP32 boxes for the P4 API; preserve continuous XYXY, never round to pixels.
    source_boxes = torch.tensor(boxes, dtype=torch.float32, device=feature_maps.device)
    scaled = scale_boxes_xyxy(source_boxes, source_size=(1024, 1024), input_size=(224, 224))
    owners_tensor = torch.tensor(owners, dtype=torch.int64, device=feature_maps.device)
    region_scores = shared_region_scores(feature_maps, classifier, scaled, owners_tensor, input_size=(224, 224))
    for i, start, k in layout:
        pairs[i].append((region_scores[start], region_scores[start + 1:start + 1 + k]))
    return pairs


def loss_from_frozen_train_batch(
    index: FrozenControlIndex,
    image_ids: Sequence[str],
    targets: Tensor,
    feature_maps: Tensor,
    classifier: nn.Linear,
    *,
    lambda_value: float,
) -> LossParts:
    """BCE on every TRAIN image, REL only for frozen P3 K>=1 lesions."""
    if not isinstance(lambda_value, (float, int)) or not math.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("Invalid lambda")
    # Match MobileNetV2OS16.forward() image logits with the same classifier.
    image_logits = classifier(feature_maps.mean(dim=(2, 3))).flatten()
    if lambda_value == 0:
        # Avoid even evaluating REL ROIs: exactly the BCE-only computation graph.
        return bce_rel_loss(image_logits, targets, [[] for _ in image_ids], lambda_value=0.0)
    pairs = pair_scores_from_frozen(index, image_ids, targets, feature_maps, classifier)
    return bce_rel_loss(image_logits, targets, pairs, lambda_value=lambda_value)
