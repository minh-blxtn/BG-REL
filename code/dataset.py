"""BG-REL frozen RSNA dataset, image-only preprocessing shared by B0 and A3.

No validation/test images used for model training; augmentation is photometric only.
"""
from __future__ import annotations
import csv
import hashlib
import math
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from paths import dicom_from_manifest

SPLIT_SHA = 'ccabcf432b0d266264e4ea96ca018b05a67d944af65fca4ad159f5130da7f9a7'
MASTER_SHA = '4c690ddf7918d0890b6715dba290be85a16959adc0dcaa70b5b4c5bd387d6c9b'
IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open('rb') as fp:
        for b in iter(lambda:fp.read(2**20),b''):
            digest.update(b)
    return digest.hexdigest()


def read_csv(path:Path):
    with path.open('r',encoding='utf-8-sig',newline='') as fp:
        return list(csv.DictReader(fp))


def load_frozen(root:Path):
    """Read metadata for all splits; construct ONLY train/val datasets."""
    base=root/'data/processed'
    for file,expected in [('split.csv',SPLIT_SHA),('master_manifest.csv',MASTER_SHA)]:
        if sha256(base/file)!=expected:
            raise ValueError(f'Frozen data changed: {file}')
    if (base/'split.sha256').read_text(encoding='utf-8').split()[0]!=SPLIT_SHA:
        raise ValueError('Split seal changed')
    split=read_csv(base/'split.csv')
    manifest=read_csv(base/'master_manifest.csv')
    if len(split)!=29684 or len(manifest)!=29684:
        raise ValueError('Frozen row counts changed')
    by_id={r['image_id']:r for r in manifest}
    if len(by_id)!=29684 or len({r['image_id'] for r in split})!=29684:
        raise ValueError('Duplicated image IDs')
    patients={k:set() for k in ('train','val','test')}
    groups={k:[] for k in patients}
    for row in split:
        sid=row['image_id'];part=row['split']
        if part not in groups:
            raise ValueError(f'Unexpected split: {part}')
        m=by_id[sid]
        if m['patient_id']!=row['patient_id']:
            raise ValueError('Patient mismatch')
        if (m['final_label']=='Lung Opacity') != (m['target']=='1'):
            raise ValueError('Target/final label mismatch')
        if m['target'] not in ('0','1'):
            raise ValueError('Invalid binary target')
        patients[part].add(row['patient_id'])
        if part!='test':
            groups[part].append(m)
    if any(patients[a]&patients[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Patient leakage')
    if len(groups['train'])!=23748 or len(groups['val'])!=2969:
        raise ValueError('Unexpected train/validation size')
    return sorted(groups['train'],key=lambda x:x['image_id']),sorted(groups['val'],key=lambda x:x['image_id'])


def decode_dicom(root:Path,row:dict[str,str])->np.ndarray:
    import pydicom
    dicom_path=dicom_from_manifest(row['dicom_path'], root=root)
    if not dicom_path.is_file():
        raise FileNotFoundError(f'DICOM file missing: {dicom_path}')
    ds=pydicom.dcmread(str(dicom_path))
    if str(ds.SOPInstanceUID)!=row['sop_uid'] or str(ds.PatientID)!=row['dicom_patient_id']:
        raise ValueError(f'DICOM UID/patient mismatch: {row["image_id"]}')
    if str(ds.PhotometricInterpretation)!=row['photometric_interpretation']:
        raise ValueError(f'DICOM photometric mismatch: {row["image_id"]}')
    if (int(ds.Rows),int(ds.Columns))!=(1024,1024) or int(ds.SamplesPerPixel)!=1:
        raise ValueError(f'DICOM shape mismatch: {row["image_id"]}')
    a=np.asarray(ds.pixel_array,dtype=np.float32)
    if a.shape!=(1024,1024) or not np.isfinite(a).all():
        raise ValueError(f'DICOM invalid pixels: {row["image_id"]}')
    lo=float(a.min());hi=float(a.max())
    if hi<=lo:
        raise ValueError(f'DICOM constant pixels: {row["image_id"]}')
    a=(a-lo)/(hi-lo)
    if row['photometric_interpretation']=='MONOCHROME1':
        a=1.0-a
    return a


def prepare_image(pixel01:np.ndarray,*,train:bool=False,brightness=1.0,contrast=1.0):
    if pixel01.shape!=(1024,1024) or not np.isfinite(pixel01).all():
        raise ValueError('Unexpected source image shape or invalid values')
    x=torch.from_numpy(np.array(pixel01,dtype=np.float32,copy=True))[None,None]
    x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False).squeeze(0)
    if train:
        mean=x.mean()
        x=((x-mean)*float(contrast)+mean)*float(brightness)
        x=x.clamp(0,1)
    x=x.expand(3,-1,-1)
    mean=torch.tensor(IMAGENET_MEAN,dtype=x.dtype)[:,None,None]
    std=torch.tensor(IMAGENET_STD,dtype=x.dtype)[:,None,None]
    return (x-mean)/std


def augmentation_factors(seed:int,epoch:int,image_id:str):
    raw=hashlib.sha256(f'{seed}|{epoch}|{image_id}|photo'.encode()).digest()
    a=int.from_bytes(raw[:8],'little')/float(2**64)
    b=int.from_bytes(raw[8:16],'little')/float(2**64)
    return 0.9+0.2*a,0.9+0.2*b


class RSNAImageDataset(Dataset):
    def __init__(self,root:Path,rows:list[dict[str,str]],*,train:bool,seed:int,epoch:int=0):
        self.root=root;self.rows=rows;self.train=train;self.seed=seed;self.epoch=epoch
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        img=decode_dicom(self.root,row)
        bright,contrast=augmentation_factors(self.seed,self.epoch,row['image_id']) if self.train else (1.,1.)
        return prepare_image(img,train=self.train,brightness=bright,contrast=contrast),torch.tensor(float(row['target']),dtype=torch.float32)
