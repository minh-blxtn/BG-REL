"""T2-A no-augmentation baseline CLI, using the shared B0 training engine.

Distinct archived experiment: never misidentify as the paper B0 run.
Run `--help` for supported diagnostic modes. Archived --train/--resume are blocked.
"""
from __future__ import annotations
from pathlib import Path

# Import public helpers to retain diagnostic/API compatibility, rather than
# carrying another 350 lines of near-duplicate trainer implementation.
from train_baseline import (
    BATCH, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, MIN_IMPROVEMENT, SEED,
    METRIC_COLUMNS, PROJECT_ROOT, T2A_RECIPE,
    BaselineRecipe, baseline_main,
    init_logger, set_seed, ensure_precision_supported, validate_metadata,
    validate_sample_pixels, save_json_exclusive, atomic_pt, atomic_metrics,
    train_epoch, validate_epoch,
    make_loader as _shared_make_loader,
    smoke as _shared_smoke, run_training as _shared_run_training,
    code_hashes as _shared_code_hashes,
)
from dataset import RSNAImageDataset
from torch.utils.data import DataLoader
import torch

RUN_ID='b0_t2_noaug_s42'
SMOKE_FILENAME='bce_smoke_t2_noaug_s42.json'
EXPERIMENT='T2-A'
CONTROL_RUN_ID='b0_dev_s42'
AUGMENT_TRAIN=False
CODE_FILES=['dataset.py','model.py','metrics.py','train_baseline.py','train_t2a_noaug.py']


def code_hashes():
    return _shared_code_hashes(T2A_RECIPE)


def make_loader(root,rows,*,train:bool,epoch:int,augment_train:bool=AUGMENT_TRAIN):
    return _shared_make_loader(root,rows,train=train,epoch=epoch,augment_train=augment_train)


def smoke(root:Path,device:torch.device):
    return _shared_smoke(root,device,recipe=T2A_RECIPE)


def run_training(root:Path,device:torch.device,*,resume:bool):
    return _shared_run_training(root,device,resume=resume,recipe=T2A_RECIPE)


def main():
    baseline_main(T2A_RECIPE)


if __name__=='__main__':
    main()
