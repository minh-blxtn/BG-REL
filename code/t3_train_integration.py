"""T3: identity-preserving TRAIN loader and guarded one-step utilities.

T3 is a diagnostic only. It does not replace the frozen P5 baseline trainer,
modify the P3 sampler, or implement multi-epoch P6 training.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import RSNAImageDataset
from rel_adapter import FrozenControlIndex, loss_from_frozen_train_batch

SEED = 42
BATCH = 4
EPOCH = 1
TRAIN_COUNT = 23748
MAX_EPOCHS = 20
LR = 3e-5
WEIGHT_DECAY = 1e-3
CLIP_NORM = 5.0
LAMBDAS = (0.0, 0.1)


class IdentifiedTrainDataset(Dataset):
    """Add image IDs without changing RSNAImageDataset's pixel or label path."""
    def __init__(self, base: Dataset, rows: Sequence[Mapping[str, str]]):
        if len(base) != len(rows) or not rows:
            raise ValueError("Dataset/rows length mismatch")
        image_ids = [r['image_id'] for r in rows]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("Duplicate image_id in selected rows")
        if any(r['target'] not in ('0', '1') for r in rows):
            raise ValueError("Invalid binary target")
        self.base = base
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        image, target = self.base[index]
        row = self.rows[index]
        if tuple(image.shape) != (3, 224, 224):
            raise ValueError("Unexpected B0 image tensor shape")
        if target.ndim != 0 or float(target.item()) != float(row['target']):
            raise ValueError("Identity/target alignment failure")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("Non-finite image tensor")
        return image, target, row['image_id']


def make_identified_loader(root: Path, rows: Sequence[Mapping[str, str]], *, epoch: int = EPOCH) -> DataLoader:
    """Reproduce B0's augmentation and shuffle recipe for a selected TRAIN subset.

    B0's DataLoader generator seed is 42 + epoch*1009 + 1 for TRAIN.
    For this smoke, the selected 4-item subset is deliberate, not an epoch of
    the full 23,748-image training population.
    """
    if epoch < 1:
        raise ValueError("TRAIN epoch must be >= 1")
    base = RSNAImageDataset(root, list(rows), train=True, seed=SEED, epoch=epoch)
    dataset = IdentifiedTrainDataset(base, rows)
    generator = torch.Generator().manual_seed(SEED + epoch * 1009 + 1)
    return DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0,
                      pin_memory=torch.cuda.is_available(), drop_last=False, generator=generator)


def select_four_train_rows(rows: Sequence[Mapping[str, str]], index: FrozenControlIndex) -> list[Mapping[str, str]]:
    """Select K=1, K=2, K=0 positive and negative, all distinct TRAIN patients."""
    selected: list[Mapping[str, str]] = []
    patients: set[str] = set()

    def take(predicate, label: str):
        for row in sorted(rows, key=lambda r: r['image_id']):
            image_id = row['image_id']
            if row['patient_id'] in patients or image_id not in index.train_targets:
                continue
            if int(row['target']) != index.train_targets[image_id]:
                raise ValueError("TRAIN target mismatch")
            if predicate(row):
                selected.append(row)
                patients.add(row['patient_id'])
                return
        raise ValueError(f"Insufficient distinct TRAIN patients for {label}")

    take(lambda r: r['target']=='1' and any(len(g.control_xyxy)==1 for g in index.for_image(r['image_id']))
         and not any(len(g.control_xyxy)==2 for g in index.for_image(r['image_id'])), 'K1')
    take(lambda r: r['target']=='1' and any(len(g.control_xyxy)==2 for g in index.for_image(r['image_id'])), 'K2')
    take(lambda r: r['target']=='1' and len(index.for_image(r['image_id']))==0, 'positive K0')
    take(lambda r: r['target']=='0' and len(index.for_image(r['image_id']))==0, 'negative')
    if len(selected) != BATCH or len({r['image_id'] for r in selected}) != BATCH:
        raise ValueError("Unable to select independent TRAIN smoke batch")
    return selected


def batch_metadata_check(index: FrozenControlIndex, image_ids: Sequence[str], targets: Tensor) -> dict:
    """Detect identity corruption even if torch DataLoader shuffles the batch."""
    if len(image_ids) != BATCH or targets.shape != (BATCH,) or len(set(image_ids)) != BATCH:
        raise ValueError("Bad TRAIN batch dimensions or duplicate image IDs")
    kinds: list[str] = []
    for image_id, label in zip(image_ids, targets.tolist()):
        if image_id not in index.train_targets or int(label) != index.train_targets[image_id]:
            raise ValueError("Non-TRAIN image or wrong target after shuffle")
        groups = index.for_image(image_id)
        if label == 0:
            if groups: raise ValueError("Negative image has REL annotations")
            kind = 'negative'
        elif any(len(g.control_xyxy)==2 for g in groups):
            kind = 'matched_K2'
        elif any(len(g.control_xyxy)==1 for g in groups):
            kind = 'matched_K1'
        else:
            kind = 'positive_K0'
        kinds.append(kind)
    if sorted(kinds) != sorted(['matched_K1','matched_K2','positive_K0','negative']):
        raise ValueError(f"Smoke batch composition changed: {kinds}")
    return {'image_ids':list(image_ids), 'targets':[int(t) for t in targets.tolist()], 'kinds':kinds}


def check_b0_recipe(config: dict, summary: dict, *, dataset_sha: str, model_sha: str) -> None:
    """Bind T3 smoke to completed B0, NOT interrupted no-augmentation T2-A."""
    expected = {
        'run_id':'b0_dev_s42', 'seed':SEED, 'batch':BATCH, 'epochs_cap':MAX_EPOCHS,
        'lr':LR, 'weight_decay':WEIGHT_DECAY, 'optimizer':'AdamW',
        'scheduler':'CosineAnnealingLR_per_optimizer_step', 'amp_precision':'bf16',
        'grad_scaler_enabled':False,
        'augment_train':'brightness_contrast_[0.9,1.1]', 'validation_augment':False,
        'train_samples':TRAIN_COUNT, 'val_samples':2969, 'test_images_opened':False,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"B0 recipe mismatch: {name}")
    if config.get('source_sha256',{}).get('dataset.py') != dataset_sha:
        raise ValueError("B0 dataset.py SHA mismatch")
    if config.get('source_sha256',{}).get('model.py') != model_sha:
        raise ValueError("B0 model.py SHA mismatch")
    if summary.get('status') != 'FINISHED' or summary.get('best_epoch') != 3:
        raise ValueError("B0 completion proof mismatch")
    if not math.isclose(float(summary.get('best_val_auroc',-1)),.8809477538216985,abs_tol=1e-12):
        raise ValueError("B0 validation metric proof mismatch")
    if summary.get('checkpoint_sha256') != 'f6bda8a74c5d72b948a9ee4ba35cf53c1c0176e7d15dbb457ecc78a4d79f2929':
        raise ValueError("B0 best checkpoint provenance mismatch")


@dataclass
class StepEvidence:
    total: float
    bce: float
    rel: float
    grad_norm_before_clip: float
    eligible_images: int
    eligible_lesions: int
    gradients: dict[str, Tensor]
    final_state: dict[str, Tensor]
    head_changed: bool
    optimizer_steps: int
    scheduler_lr: float


def train_one_diagnostic_step(model: nn.Module, x: Tensor, y: Tensor, ids: Sequence[str],
                              index: FrozenControlIndex, *, mode: str) -> StepEvidence:
    """Run one in-memory B0 or A3 step, with B0's AdamW/cosine/clip/BF16 path.

    All buffers and weights are mutated in-memory only. There is deliberately
    no optimizer or model checkpoint persistence and no multi-batch loop.
    """
    if mode not in ('B0', 'A3_lambda0', 'A3_lambda010'):
        raise ValueError("Unknown T3 step mode")
    batch_metadata_check(index, ids, y)
    device=x.device
    if device != y.device or device.type != 'cuda':
        raise ValueError("T3 optimizer smoke requires CUDA tensors")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 GPU unavailable")
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=math.ceil(TRAIN_COUNT/BATCH)*MAX_EPOCHS)
    initial_head=model.classifier.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if mode == 'B0':
            logits=model(x)
            total=F.binary_cross_entropy_with_logits(logits,y)
            bce=total
            rel=total.detach()*0.0
            eligible_images=eligible_lesions=0
        else:
            features=model.forward_features(x)
            lambda_value = 0.0 if mode == 'A3_lambda0' else 0.1
            parts=loss_from_frozen_train_batch(index,ids,y,features,model.classifier,lambda_value=lambda_value)
            total,bce,rel=parts.total,parts.bce,parts.rel
            eligible_images,eligible_lesions=parts.eligible_images,parts.eligible_lesions
    if not bool(torch.isfinite(total)) or not bool(torch.isfinite(bce)) or not bool(torch.isfinite(rel)):
        raise FloatingPointError("Non-finite T3 loss")
    total.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"Missing/non-finite gradient: {name}")
        gradients[name]=parameter.grad.detach().cpu().clone()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP_NORM,error_if_nonfinite=True)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("Non-finite T3 grad norm")
    optimizer.step()
    scheduler.step()
    head_changed=bool(torch.any(model.classifier.weight.detach()!=initial_head))
    if not head_changed:
        raise RuntimeError("Optimizer did not update shared head")
    return StepEvidence(total=float(total.item()),bce=float(bce.item()),rel=float(rel.item()),
                        grad_norm_before_clip=float(norm.item()),eligible_images=eligible_images,
                        eligible_lesions=eligible_lesions,gradients=gradients,
                        final_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        head_changed=head_changed,optimizer_steps=1,
                        scheduler_lr=float(optimizer.param_groups[0]['lr']))


def compare_parity(b0: StepEvidence, a3zero: StepEvidence, *,
                   loss_tol: float=1e-6, grad_tol: float=1e-5, state_tol: float=1e-6) -> dict:
    """Fail closed if zero-lambda changes B0's loss, gradient or optimizer update."""
    if set(b0.gradients) != set(a3zero.gradients) or set(b0.final_state) != set(a3zero.final_state):
        raise ValueError("Parity parameter/buffer names differ")
    grad_diff=max(float((b0.gradients[k].float()-a3zero.gradients[k].float()).abs().max()) for k in b0.gradients)
    state_diff=0.0
    for key in b0.final_state:
        x,y=b0.final_state[key],a3zero.final_state[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"State metadata mismatch: {key}")
        if not x.is_floating_point():
            if not torch.equal(x,y):
                raise RuntimeError(f"Integer model state mismatch: {key}")
        else:
            state_diff=max(state_diff,float((x.float()-y.float()).abs().max()))
    loss_diff=abs(b0.total-a3zero.total)
    output={'max_grad_abs_difference':grad_diff,'max_model_state_abs_difference':state_diff,
            'loss_abs_difference':loss_diff,'loss_tol':loss_tol,'grad_tol':grad_tol,'state_tol':state_tol}
    if loss_diff>loss_tol or grad_diff>grad_tol or state_diff>state_tol:
        raise RuntimeError(f"B0/A3(lambda=0) numerical parity failed: {output}")
    if b0.optimizer_steps != 1 or a3zero.optimizer_steps != 1 or abs(b0.scheduler_lr-a3zero.scheduler_lr)>1e-12:
        raise RuntimeError("Optimizer/scheduler parity failed")
    return output
