"""BG-REL V3.2 P3: frozen-TRAIN, anatomy-aware, deterministic control sampling.

Run --preflight, then --run. CPU only; consumes the P2 mask cache without
modifying it. A control is an unannotated comparison region, not a clinically
verified healthy region. No validation/test rows are sampled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from paths import dicom_from_manifest

import numpy as np

EXPECTED_SPLIT = "ccabcf432b0d266264e4ea96ca018b05a67d944af65fca4ad159f5130da7f9a7"
EXPECTED_MASTER = "4c690ddf7918d0890b6715dba290be85a16959adc0dcaa70b5b4c5bd387d6c9b"
EXPECTED_BOXES = "6372a1facc3e1b78fbcd97a5bd103d2fc33ed257c92f8076b1d36e9168078e45"
SIZE = 1024
SEED = 42
KMAX = 2
STEP = 8
BUDGET = 128
OCC_MIN = 0.80
DOMINANT_SHARE = 0.80
VERT_MAX = 0.10
PAIR_IOU_MAX = 0.20
EPS = 1e-6

TRACE_FIELDS = ["image_id", "patient_id", "annotation_id", "view_position", "num_boxes",
                "lesion_x0", "lesion_y0", "lesion_x1", "lesion_y1", "lesion_area",
                "mask_status", "mask_geometry_status", "lesion_side", "lesion_lung_coverage",
                "dominant_side_share", "lesion_u_y", "candidate_count", "tier1_examined",
                "tier2_examined", "reject_bounds", "reject_lesion_overlap", "reject_occupancy",
                "reject_vertical", "reject_duplicate", "k", "tier1_count", "tier2_count", "skip_reason"]
CONTROL_FIELDS = ["image_id", "patient_id", "annotation_id", "control_id", "tier", "side",
                  "x0", "y0", "x1", "y1", "lesion_x0", "lesion_y0", "lesion_x1", "lesion_y1",
                  "occupancy", "u_y", "lesion_u_y", "ocr", "max_pairwise_iou",
                  "geometry_valid", "visual_status", "visual_reason"]
FEAS_FIELDS = ["analysis", "stratum", "budget", "step", "images", "lesions", "matched_lesions",
              "k0", "k1", "k2", "tier1_controls", "tier2_controls", "elapsed_seconds"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for part in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, fields: list[str], records: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def f(v: float) -> str:
    return format(float(v), ".6f")


def verify_split_partitions(rows: list[dict[str, str]]) -> None:
    partitions = {sp: {r["patient_id"] for r in rows if r["split"] == sp}
                  for sp in ("train", "val", "test")}
    if set(r["split"] for r in rows) != set(partitions):
        raise ValueError("Unexpected frozen split partition labels")
    if (partitions["train"] & partitions["val"] or partitions["train"] & partitions["test"]
            or partitions["val"] & partitions["test"]):
        raise ValueError("Patient leakage detected")


def verified_inputs(root: Path, *, require_masks: bool = False):
    processed = root / "data/processed"
    qa = root / "results/qa"
    split_path = processed / "split.csv"
    master_path = processed / "master_manifest.csv"
    boxes_path = processed / "boxes.csv"
    audit_path = qa / "train_mask_audit.csv"
    manifest_path = qa / "train_mask_manifest.json"
    for path, expected in ((split_path, EXPECTED_SPLIT), (master_path, EXPECTED_MASTER),
                           (boxes_path, EXPECTED_BOXES)):
        if sha256(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    seal = (processed / "split.sha256").read_text(encoding="utf-8").split()[0]
    if seal != EXPECTED_SPLIT:
        raise ValueError("Split seal mismatch")
    p2 = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (p2.get("status") != "GENERATED_PROVISIONAL_NOT_REL_APPROVED"
            or p2.get("split_sha256") != EXPECTED_SPLIT
            or p2.get("processed") != 23748
            or sha256(audit_path) != p2.get("csv_sha256")):
        raise ValueError("P2 final manifest/provenance mismatch")
    split = read_csv(split_path)
    train_ids = {r["image_id"] for r in split if r["split"] == "train"}
    if len(train_ids) != 23748 or len(split) != 29684 or len({r["image_id"] for r in split}) != len(split):
        raise ValueError("Frozen split membership/count inconsistent")
    verify_split_partitions(split)
    master = {r["image_id"]: r for r in read_csv(master_path)}
    if len(master) != len(split):
        raise ValueError("Master cohort size mismatch")
    audit = read_csv(audit_path)
    if len(audit) != len(train_ids) or {r["image_id"] for r in audit} != train_ids:
        raise ValueError("P2 audit membership mismatch")
    by_img = {r["image_id"]: r for r in audit}
    boxes_by_img = defaultdict(list)
    annotation_ids = set()
    for r in read_csv(boxes_path):
        if r["annotation_id"] in annotation_ids:
            raise ValueError("Duplicate annotation ID")
        annotation_ids.add(r["annotation_id"])
        if r["image_id"] in train_ids:
            boxes_by_img[r["image_id"]].append(r)
    positives = sorted(i for i in train_ids if master[i]["target"] == "1")
    if len(positives) != 5685 or sum(len(boxes_by_img[i]) for i in positives) != 9161:
        raise ValueError("Unexpected positive/lesion count: investigate frozen inputs")
    for image_id in train_ids:
        a, m = by_img[image_id], master[image_id]
        if (a["final_label"] != m["final_label"] or a["view_position"] != m["view_position"]
                or int(a["num_boxes"]) != int(m["num_boxes"])):
            raise ValueError(f"P2 metadata mismatch: {image_id}")
    for i in positives:
        if not boxes_by_img[i] or len(boxes_by_img[i]) != int(master[i]["num_boxes"]):
            raise ValueError(f"Missing positive boxes: {i}")
        for b in boxes_by_img[i]:
            if b["patient_id"] != master[i]["patient_id"] or b["sop_uid"] != master[i]["sop_uid"]:
                raise ValueError(f"Box identity mismatch: {i}")
    if require_masks:
        for i in positives:
            a = by_img[i]
            if a["rel_policy"] == "REL_CANDIDATE_ONLY":
                expected = f"data/lung_masks/train/{Path(i).stem}.npz"
                if a["mask_path"] != expected or not len(a["mask_sha256"]) == 64:
                    raise ValueError(f"Mask audit path/hash invalid: {i}")
                if not (root / expected).is_file():
                    raise ValueError(f"Missing P2 mask: {i}")
            elif a["rel_policy"] != "SKIP_REL" or a["mask_path"] or a["mask_sha256"]:
                raise ValueError(f"Mask SKIP_REL audit inconsistent: {i}")
    return positives, master, boxes_by_img, by_img, p2


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    def valid(self) -> bool:
        return all(math.isfinite(v) for v in asdict(self).values()) and (0 <= self.x0 < self.x1 <= SIZE and
                                                                         0 <= self.y0 < self.y1 <= SIZE)


def rect_from_row(r: dict[str, str]) -> Rect:
    box = Rect(float(r["x"]), float(r["y"]), float(r["x2"]), float(r["y2"]))
    if not box.valid() or abs(box.w - float(r["width"])) > 1e-3 or abs(box.h - float(r["height"])) > 1e-3:
        raise ValueError(f"Invalid source bbox: {r['annotation_id']}")
    return box


def intersect(a: Rect, b: Rect) -> float:
    return max(0., min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0., min(a.y1, b.y1) - max(a.y0, b.y0))


def iou(a: Rect, b: Rect) -> float:
    v = intersect(a, b)
    return v / (a.area + b.area - v) if v > 0 else 0.0


class Lung:
    """Area-integral of a boolean mask, exact for axis-aligned fractional ROIs."""
    def __init__(self, mask: np.ndarray):
        if mask.shape != (SIZE, SIZE) or mask.dtype != np.bool_:
            raise ValueError("Expected bool lung mask at source 1024x1024")
        self.mask = mask
        self.prefix = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        ys, xs = np.nonzero(mask)
        self.bounds = ((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                       if len(xs) else None)

    def _integral(self, x: float, y: float) -> float:
        x = float(np.clip(x, 0, SIZE))
        y = float(np.clip(y, 0, SIZE))
        ix, iy = int(x), int(y)
        fx, fy = x - ix, y - iy
        p = self.prefix
        v = float(p[iy, ix])
        if fx > 0 and ix < SIZE:
            v += fx * float(p[iy, ix + 1] - p[iy, ix])
        if fy > 0 and iy < SIZE:
            v += fy * float(p[iy + 1, ix] - p[iy, ix])
        if fx > 0 and fy > 0 and ix < SIZE and iy < SIZE:
            v += fx * fy * float(self.mask[iy, ix])
        return v

    def area(self, rect: Rect) -> float:
        if not rect.valid():
            raise ValueError("ROI outside source frame")
        return max(0., self._integral(rect.x1, rect.y1) - self._integral(rect.x0, rect.y1)
                   - self._integral(rect.x1, rect.y0) + self._integral(rect.x0, rect.y0))

    def u_y(self, rect: Rect) -> float:
        if self.bounds is None:
            raise ValueError("Empty lung mask")
        _, ymin, _, ymax = self.bounds
        return (rect.cy - ymin) / (ymax - ymin)


def load_lungs(path: Path, digest: str) -> dict[str, Lung]:
    if sha256(path) != digest:
        raise ValueError(f"P2 mask SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as p:
        if set(p.files) != {"left", "right", "schema_version"} or int(p["schema_version"]) != 1:
            raise ValueError(f"Unexpected P2 mask schema: {path}")
        left, right = p["left"], p["right"]
        if left.dtype != np.bool_ or right.dtype != np.bool_ or left.shape != (SIZE, SIZE) or right.shape != (SIZE, SIZE):
            raise ValueError(f"Invalid P2 mask shape/type: {path}")
        return {"left": Lung(left), "right": Lung(right)}


def side_of_lesion(rect: Rect, lungs: dict[str, Lung]):
    a = {s: lung.area(rect) for s, lung in lungs.items()}
    total = sum(a.values())
    if total <= EPS:
        return None, 0., 0., None, "NO_LUNG_OVERLAP"
    side = max(("left", "right"), key=lambda s: a[s])
    share = a[side] / total
    coverage = min(1., total / rect.area)
    if share + EPS < DOMINANT_SHARE:
        return None, coverage, share, None, "SIDE_AMBIGUOUS"
    if not lungs[side].bounds:
        return None, coverage, share, None, "EMPTY_LUNG_SIDE"
    return side, coverage, share, lungs[side].u_y(rect), ""


def stable_jitter(image_id: str, annotation_id: str, seed: int):
    b = hashlib.sha256(f"{seed}|{image_id}|{annotation_id}".encode()).digest()
    return b[0] % STEP, b[1] % STEP


def make_candidates(lesion: Rect, lung: Lung, target_u: float, target_x: float,
                    image_id: str, annotation_id: str, seed: int, step: int) -> list[Rect]:
    if lung.bounds is None:
        return []
    xmin, ymin, xmax, ymax = lung.bounds
    w, h = lesion.w, lesion.h
    if w > SIZE or h > SIZE:
        return []
    # Candidate origin can extend slightly beyond mask support when occupancy >=80%.
    low_x = max(0., xmin - 0.20 * w)
    high_x = min(SIZE - w, xmax - 0.80 * w)
    y_center = ymin + target_u * (ymax - ymin)
    target_y = y_center - h / 2.0
    low_y = max(0., target_y - 0.13 * (ymax - ymin))
    high_y = min(SIZE - h, target_y + 0.13 * (ymax - ymin))
    if high_x < low_x or high_y < low_y:
        return []
    jx, jy = stable_jitter(image_id, annotation_id, seed)
    # A single deterministic lattice for each resolution; stable budget prefixes.
    def axis_values(lo: float, hi: float, jitter: int):
        start = math.ceil((lo - jitter) / step) * step + jitter
        return [float(v) for v in np.arange(start, hi + EPS, step)]
    xs = axis_values(low_x, high_x, jx % step)
    ys = axis_values(low_y, high_y, jy % step)
    # Include the exact target-origin anchor to mitigate missed very narrow strips.
    ax, ay = min(max(target_x - w / 2, low_x), high_x), min(max(target_y, low_y), high_y)
    xs = sorted(set(xs + [float(ax)]))
    ys = sorted(set(ys + [float(ay)]))
    pairs = [(x, y) for y in ys for x in xs]
    pairs.sort(key=lambda xy: (round(abs(xy[1] + h / 2 - y_center) / step * 1.3 +
                                     abs(xy[0] + w / 2 - target_x) / step * 0.08, 8),
                               abs(xy[1] + h / 2 - y_center), abs(xy[0] + w / 2 - target_x), xy[1], xy[0]))
    return [Rect(x, y, x + w, y + h) for x, y in pairs]


def sample_lesion(image_id: str, annotation_id: str, lesion: Rect, all_boxes: list[Rect],
                  lungs: dict[str, Lung] | None, *, budget: int = BUDGET, step: int = STEP,
                  seed: int = SEED):
    rejects = Counter()
    accepted = []
    tier_count = Counter()
    examined = 0
    side, coverage, share, uy, skip = ((None, 0., 0., None, "IMAGE_SKIP_REL") if lungs is None
                                        else side_of_lesion(lesion, lungs))
    if skip:
        return {"side": side or "", "coverage": coverage, "share": share, "uy": uy,
                "k": 0, "skip": skip, "rejects": rejects, "examined": examined,
                "tier_count": tier_count, "accepted": accepted}
    for tier, candidate_side in ((1, side), (2, "right" if side == "left" else "left")):
        if len(accepted) == KMAX:
            break
        lung = lungs[candidate_side]
        if lung.bounds is None:
            continue
        target_x = lesion.cx if tier == 1 else (lung.bounds[0] + lung.bounds[2]) / 2.0
        for c in make_candidates(lesion, lung, uy, target_x, image_id, annotation_id, seed, step)[:budget]:
            examined += 1
            tier_count[f"examined_{tier}"] += 1
            if not c.valid() or abs(c.area - lesion.area) > EPS:
                rejects["bounds"] += 1
                continue
            if any(intersect(c, b) > EPS for b in all_boxes):
                rejects["lesion_overlap"] += 1
                continue
            occ = lung.area(c) / c.area
            if occ + EPS < OCC_MIN:
                rejects["occupancy"] += 1
                continue
            cuy = lung.u_y(c)
            if abs(cuy - uy) > VERT_MAX + EPS:
                rejects["vertical"] += 1
                continue
            max_iou = max((iou(c, row["rect"]) for row in accepted), default=0.)
            if max_iou > PAIR_IOU_MAX + EPS:
                rejects["duplicate"] += 1
                continue
            accepted.append({"rect": c, "tier": tier, "side": candidate_side,
                             "occupancy": occ, "uy": cuy, "max_iou": max_iou})
            tier_count[str(tier)] += 1
            if len(accepted) == KMAX:
                break
    if not accepted:
        skip = "SEARCH_EXHAUSTED"  # Not proof that a control cannot exist.
    return {"side": side, "coverage": coverage, "share": share, "uy": uy,
            "k": len(accepted), "skip": skip, "rejects": rejects,
            "examined": examined, "tier_count": tier_count, "accepted": accepted}


def make_trace(image_id, patient_id, a, lesion, view, num_boxes, mask_status, geo_status, sample):
    z = sample["rejects"]
    t = sample["tier_count"]
    return dict(zip(TRACE_FIELDS, (
        image_id, patient_id, a, view, num_boxes, f(lesion.x0), f(lesion.y0), f(lesion.x1),
        f(lesion.y1), f(lesion.area), mask_status, geo_status, sample["side"],
        f(sample["coverage"]), f(sample["share"]), "" if sample["uy"] is None else f(sample["uy"]),
        sample["examined"], t["examined_1"], t["examined_2"], z["bounds"],
        z["lesion_overlap"], z["occupancy"], z["vertical"], z["duplicate"], sample["k"],
        t["1"], t["2"], sample["skip"])))


def make_controls(image_id, patient_id, a, lesion, sample):
    records = []
    for k, c in enumerate(sample["accepted"], 1):
        r = c["rect"]
        records.append(dict(zip(CONTROL_FIELDS, (image_id, patient_id, a, f"{a}:C{k}", c["tier"], c["side"],
                         f(r.x0), f(r.y0), f(r.x1), f(r.y1), f(lesion.x0), f(lesion.y0),
                         f(lesion.x1), f(lesion.y1), f(c["occupancy"]), f(c["uy"]),
                         f(sample["uy"]), "0.000000", f(c["max_iou"]), "TRUE", "UNKNOWN",
                         "Geometry verified only; no clinical normality claim"))))
    return records


def choose_hard_images(positives, boxes_by_img, audit, n=80):
    # Freeze algorithm before evaluating benchmark. It emphasizes difficult boxes,
    # multi-box images, and P2 mask-geometry issues, without using model scores.
    scored = []
    for i in positives:
        a = audit[i]
        boxes = boxes_by_img[i]
        areas = [float(b["width"]) * float(b["height"]) for b in boxes]
        max_frac = max(areas) / (SIZE * SIZE)
        small_frac = min(areas) / (SIZE * SIZE)
        score = (10 if a["rel_policy"] == "SKIP_REL" else 0,
                 5 if len(boxes) >= 3 else 3 if len(boxes) > 1 else 0,
                 2 if max_frac >= 0.08 or small_frac <= 0.005 else 0,
                 max_frac, len(boxes))
        scored.append((score, i))
    scored.sort(key=lambda e: (e[0], e[1]), reverse=True)
    # Select the benchmark from images with masks so candidate budgets are testable.
    usable = [i for _, i in scored if audit[i]["rel_policy"] == "REL_CANDIDATE_ONLY"]
    selected = usable[:n]
    if len(selected) < n:
        raise ValueError("Not enough positive TRAIN masks for feasibility comparison")
    return selected


def feasibility(positives, master, boxes_by_img, audit, root, logger):
    hard = choose_hard_images(positives, boxes_by_img, audit)
    results = []
    for budget, step, tag in ((128, 8, "primary_128"), (512, 8, "expanded_512"),
                               (512, 4, "dense_step4_512")):
        started = time.monotonic()
        kcounts = Counter()
        tiers = Counter()
        for ix, image_id in enumerate(hard):
            info = audit[image_id]
            lung = load_lungs(root / info["mask_path"], info["mask_sha256"])
            boxes = sorted(boxes_by_img[image_id], key=lambda b: (int(b["box_number"]), b["annotation_id"]))
            rects = [rect_from_row(b) for b in boxes]
            for b, rect in zip(boxes, rects):
                s = sample_lesion(image_id, b["annotation_id"], rect, rects, lung, budget=budget, step=step)
                kcounts[s["k"]] += 1
                tiers[1] += s["tier_count"]["1"]
                tiers[2] += s["tier_count"]["2"]
            if (ix + 1) % 20 == 0:
                logger.info("Feasibility %s %d/%d", tag, ix + 1, len(hard))
        results.append(dict(zip(FEAS_FIELDS, ("budget_benchmark", tag, budget, step, len(hard),
                   sum(kcounts.values()), kcounts[1] + kcounts[2], kcounts[0], kcounts[1],
                   kcounts[2], tiers[1], tiers[2], f(time.monotonic() - started)))))
    return results, hard


def statistics(traces, controls):
    n_lesions = len(traces)
    by_img = defaultdict(list)
    for r in traces:
        by_img[r["image_id"]].append(r)
    positive_match_images = sum(any(int(r["k"]) > 0 for r in rows) for rows in by_img.values())
    counts = Counter(int(r["k"]) for r in traces)
    by_group = defaultdict(list)
    areas = sorted(float(r["lesion_area"]) for r in traces)
    quartile = [areas[int((len(areas) - 1) * q)] for q in (0.25, 0.5, 0.75)]
    for r in traces:
        size = float(r["lesion_area"])
        size_bin = "Q1" if size <= quartile[0] else "Q2" if size <= quartile[1] else "Q3" if size <= quartile[2] else "Q4"
        for dimension, val in (("view", r["view_position"]), ("box_count", "multi" if int(r["num_boxes"]) > 1 else "single"),
                               ("mask_quality", r["mask_geometry_status"]), ("lesion_area", size_bin),
                               ("lesion_side", r["lesion_side"] or "UNKNOWN")):
            by_group[(dimension, val)].append(r)
    group_summary = [dict(dimension=d, value=v, lesions=len(items), eligible=sum(int(r["k"]) > 0 for r in items),
                          k0=sum(int(r["k"]) == 0 for r in items), k1=sum(int(r["k"]) == 1 for r in items),
                          k2=sum(int(r["k"]) == 2 for r in items))
                     for (d, v), items in sorted(by_group.items())]
    return {"positive_images": len(by_img), "lesions": n_lesions,
            "matched_positive_images": positive_match_images, "matched_lesions": counts[1] + counts[2],
            "k_distribution": {str(k): counts[k] for k in range(KMAX + 1)},
            "tier_distribution": {str(t): sum(int(c["tier"]) == t for c in controls) for t in (1, 2)},
            "skip_reasons": dict(sorted(Counter(r["skip_reason"] for r in traces if r["skip_reason"]).items())),
            "lesion_area_quartile_thresholds": quartile, "subgroups": group_summary}


def verify_outputs(traces, controls, boxes_by_img) -> None:
    """Recheck serialized coordinates against ALL lesion boxes and K constraints."""
    trace_by_annotation = {}
    for t in traces:
        key = (t["image_id"], t["annotation_id"])
        if key in trace_by_annotation:
            raise ValueError(f"Duplicate lesion trace: {key}")
        trace_by_annotation[key] = t
    control_by_annotation = defaultdict(list)
    seen_ids = set()
    for c in controls:
        key = (c["image_id"], c["annotation_id"])
        if key not in trace_by_annotation or c["control_id"] in seen_ids:
            raise ValueError(f"Orphan/duplicate control: {c['control_id']}")
        seen_ids.add(c["control_id"])
        r = Rect(*(float(c[p]) for p in ("x0", "y0", "x1", "y1")))
        lesion = Rect(*(float(c[p]) for p in ("lesion_x0", "lesion_y0", "lesion_x1", "lesion_y1")))
        if not r.valid() or abs(r.w - lesion.w) > 2e-5 or abs(r.h - lesion.h) > 2e-5:
            raise ValueError(f"Invalid accepted control geometry: {c['control_id']}")
        for box in boxes_by_img[c["image_id"]]:
            if intersect(r, rect_from_row(box)) > EPS:
                raise ValueError(f"Accepted control overlaps lesion union: {c['control_id']}")
        if (float(c["occupancy"]) < OCC_MIN - 2e-6 or float(c["ocr"]) != 0.
                or float(c["max_pairwise_iou"]) > PAIR_IOU_MAX + 2e-6
                or abs(float(c["u_y"]) - float(c["lesion_u_y"])) > VERT_MAX + 2e-6):
            raise ValueError(f"Accepted control violates threshold: {c['control_id']}")
        if c["visual_status"] != "UNKNOWN" or c["geometry_valid"] != "TRUE":
            raise ValueError("P3 may not claim manual clinical review")
        control_by_annotation[key].append(r)
    for key, t in trace_by_annotation.items():
        items = control_by_annotation.get(key, [])
        if int(t["k"]) != len(items) or len(items) > KMAX:
            raise ValueError(f"K/controls mismatch: {key}")
        if len(items) != int(t["tier1_count"]) + int(t["tier2_count"]):
            raise ValueError(f"Tier count mismatch: {key}")
        if (len(items) == 0) != bool(t["skip_reason"]):
            raise ValueError(f"Invalid skip reason: {key}")
        for n in range(len(items)):
            for j in range(n):
                if iou(items[n], items[j]) > PAIR_IOU_MAX + 2e-6:
                    raise ValueError(f"Accepted controls overlap each other: {key}")


def save_overlays(root, selected, master, boxes_by_img, audit, trace_by_img, control_by_img, staging):
    # Display-only QA. Do not use rendered images as training masks.
    from PIL import Image, ImageDraw
    import pydicom
    out = staging / "overlays"
    out.mkdir()
    for image_id in selected:
        meta = master[image_id]
        source = dicom_from_manifest(meta["dicom_path"], root=root)
        if not source.is_file():
            raise FileNotFoundError(f"DICOM missing for overlay: {image_id}")
        data = pydicom.dcmread(str(source)).pixel_array.astype(np.float32)
        if data.shape != (SIZE, SIZE):
            raise ValueError(f"Overlay DICOM has unexpected geometry: {image_id}")
        lo, hi = float(data.min()), float(data.max())
        gray = (np.clip((data - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8) if hi > lo else np.zeros(data.shape, dtype=np.uint8)
        if meta["photometric_interpretation"] == "MONOCHROME1":
            gray = 255 - gray
        im = Image.fromarray(gray, mode="L").convert("RGB")
        # Subtle mask overlay helps identify left/right misalignment.
        a = audit[image_id]
        if a["rel_policy"] == "REL_CANDIDATE_ONLY":
            with np.load(root / a["mask_path"], allow_pickle=False) as p:
                pix = np.asarray(im, dtype=np.uint8).copy()
                lm, rm = p["left"], p["right"]
                pix[lm, 1] = np.maximum(pix[lm, 1], 128)
                pix[rm, 2] = np.maximum(pix[rm, 2], 145)
                im = Image.fromarray(pix)
        draw = ImageDraw.Draw(im)
        for b in boxes_by_img[image_id]:
            r = rect_from_row(b)
            draw.rectangle((r.x0, r.y0, r.x1, r.y1), outline=(255, 70, 70), width=4)
        for c in control_by_img.get(image_id, []):
            draw.rectangle(tuple(float(c[k]) for k in ("x0", "y0", "x1", "y1")),
                           outline=(255, 230, 30) if c["tier"] == 1 else (255, 145, 35), width=3)
        im.save(out / f"{Path(image_id).stem}.png")
    return out


def run(root: Path, logger) -> None:
    qa = root / "results/qa"
    files = [qa / f for f in ("sampler_trace.csv", "qa_controls.csv", "feasibility_summary.csv", "sampler_manifest.json")]
    lock = qa / ".sample_controls.lock"
    stage = qa / ".sample_controls_stage"
    if any(p.exists() for p in files) or lock.exists() or stage.exists() or (qa / "control_overlays").exists():
        raise RuntimeError("P3 output/lock/staging already exists; no overwrite allowed")
    positives, master, boxes_by_img, audit, p2 = verified_inputs(root, require_masks=True)
    # Exclusive lock prevents concurrent P3 writers. Left in place on failure.
    with lock.open("x", encoding="utf-8") as fp:
        fp.write(f"pid={os.getpid()}\n")
    stage.mkdir()
    logger.info("P3 START positive_images=%d lesion_boxes=%d", len(positives), sum(map(len, boxes_by_img.values())))
    feas, hard = feasibility(positives, master, boxes_by_img, audit, root, logger)
    logger.info("P3 feasibility benchmark complete: %s", feas)
    started = time.monotonic()
    traces, controls = [], []
    for index, image_id in enumerate(positives, 1):
        a, meta = audit[image_id], master[image_id]
        boxes = sorted(boxes_by_img[image_id], key=lambda b: (int(b["box_number"]), b["annotation_id"]))
        rects = [rect_from_row(b) for b in boxes]
        lungs = load_lungs(root / a["mask_path"], a["mask_sha256"]) if a["rel_policy"] == "REL_CANDIDATE_ONLY" else None
        for b, lesion in zip(boxes, rects):
            s = sample_lesion(image_id, b["annotation_id"], lesion, rects, lungs)
            traces.append(make_trace(image_id, meta["patient_id"], b["annotation_id"], lesion,
                                    meta["view_position"], len(boxes), a["rel_policy"], a["geometry_status"], s))
            controls.extend(make_controls(image_id, meta["patient_id"], b["annotation_id"], lesion, s))
        if index % 250 == 0 or index == len(positives):
            logger.info("P3 %d/%d positive images, %d controls", index, len(positives), len(controls))
            print(f"P3: {index}/{len(positives)} positive images")
    summary = statistics(traces, controls)
    if len(traces) != 9161 or summary["positive_images"] != 5685:
        raise ValueError("P3 result not complete")
    verify_outputs(traces, controls, boxes_by_img)
    trace_by_img, controls_by_img = defaultdict(list), defaultdict(list)
    for t in traces:
        trace_by_img[t["image_id"]].append(t)
    for c in controls:
        controls_by_img[c["image_id"]].append(c)
    # Include different K outcomes and views, then deterministic sorted coverage.
    overlay_ids = []
    for view in ("AP", "PA"):
        for kval in (0, 1, 2):
            for img in positives:
                if (img not in overlay_ids and master[img]["view_position"] == view
                        and any(int(t["k"]) == kval for t in trace_by_img[img])):
                    overlay_ids.append(img)
                    break
    overlay_ids.extend(i for i in hard if i not in overlay_ids)
    overlay_ids = overlay_ids[:24]
    overlay_dir = save_overlays(root, overlay_ids, master, boxes_by_img, audit, trace_by_img, controls_by_img, stage)
    write_csv(stage / "sampler_trace.csv", TRACE_FIELDS, traces)
    write_csv(stage / "qa_controls.csv", CONTROL_FIELDS, controls)
    quartile = summary["lesion_area_quartile_thresholds"]
    def area_bin(row):
        area = float(row["lesion_area"])
        return ("Q1" if area <= quartile[0] else "Q2" if area <= quartile[1]
                else "Q3" if area <= quartile[2] else "Q4")
    for g in summary["subgroups"]:
        group_rows = [r for r in traces if (
            (g["dimension"] == "view" and r["view_position"] == g["value"]) or
            (g["dimension"] == "mask_quality" and r["mask_geometry_status"] == g["value"]) or
            (g["dimension"] == "lesion_side" and (r["lesion_side"] or "UNKNOWN") == g["value"]) or
            (g["dimension"] == "box_count" and ("multi" if int(r["num_boxes"]) > 1 else "single") == g["value"]) or
            (g["dimension"] == "lesion_area" and area_bin(r) == g["value"]))]
        feas.append(dict(zip(FEAS_FIELDS, ("subgroup", g["dimension"] + "/" + g["value"], BUDGET,
                      STEP, len({r["image_id"] for r in group_rows}), g["lesions"], g["eligible"],
                      g["k0"], g["k1"], g["k2"], sum(int(r["tier1_count"]) for r in group_rows),
                      sum(int(r["tier2_count"]) for r in group_rows), ""))))
    write_csv(stage / "feasibility_summary.csv", FEAS_FIELDS, feas)
    report = {
        "phase": "P3", "status": "GENERATED_PENDING_INDEPENDENT_REVIEW", "protocol": "BG-REL_V3.2_tier12_k2",
        "scope": "FROZEN_TRAIN_ONLY", "candidate_is_clinically_normal": False,
        "parameters": {"seed": SEED, "k_max": KMAX, "step": STEP, "budget_per_tier": BUDGET,
                       "control_occupancy_min": OCC_MIN, "dominant_side_share_min": DOMINANT_SHARE,
                       "normalized_y_distance_max": VERT_MAX, "pairwise_iou_max": PAIR_IOU_MAX,
                       "lesion_control_overlap_area_max_pixels_squared": EPS},
        "inputs": {"split_sha256": EXPECTED_SPLIT, "master_sha256": EXPECTED_MASTER,
                   "boxes_sha256": EXPECTED_BOXES, "p2_audit_sha256": p2["csv_sha256"],
                   "p2_manifest_sha256": sha256(qa / "train_mask_manifest.json")},
        "source_sha256": sha256(root / "code/sample_controls.py"),
        "summary": summary,
        "benchmark": {"selected_images": hard, "comparisons": feas},
        "sampling_elapsed_seconds": round(time.monotonic() - started, 3),
        "overlay_images": overlay_ids,
        "outputs_sha256": {name: sha256(stage / name) for name in
                           ("sampler_trace.csv", "qa_controls.csv", "feasibility_summary.csv")},
        "overlay_sha256": {p.name: sha256(p) for p in sorted(overlay_dir.glob("*.png"))},
    }
    (stage / "sampler_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for fp in files:
        os.replace(stage / fp.name, fp)
    os.replace(overlay_dir, qa / "control_overlays")
    stage.rmdir()
    lock.unlink()
    logger.info("P3 COMPLETE %s", json.dumps({k: summary[k] for k in ("positive_images", "lesions", "matched_lesions", "k_distribution", "tier_distribution")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="BG-REL V3.2 P3 sampler")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.root.resolve()
    log_path = root / "results/logs/sample_controls.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger("p3")
    try:
        if args.preflight:
            positives, master, boxmap, audit, _ = verified_inputs(root, require_masks=True)
            for image_id in positives:
                for row in boxmap[image_id]:
                    rect_from_row(row)
            sample_id = next(i for i in positives if audit[i]["rel_policy"] == "REL_CANDIDATE_ONLY")
            example = audit[sample_id]
            load_lungs(root / example["mask_path"], example["mask_sha256"])
            from PIL import Image  # noqa: F401 - overlay runtime dependency
            import pydicom
            dicom = dicom_from_manifest(master[sample_id]["dicom_path"], root=root)
            if pydicom.dcmread(str(dicom)).pixel_array.shape != (SIZE, SIZE):
                raise ValueError("Preflight DICOM geometry invalid")
            print(f"P3 PREFLIGHT PASS: {len(positives)} positive TRAIN images / {sum(len(boxmap[i]) for i in positives)} lesion boxes")
            logger.info("P3 PREFLIGHT PASS, sampled %s mask/DICOM decoded", sample_id)
        else:
            run(root, logger)
    except Exception:
        logger.exception("P3 FAILED")
        raise


if __name__ == "__main__":
    main()
