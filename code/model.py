"""MobileNetV2-OS16 with GAP and one shared affine classification head.

The same image graph will be reused for A3; PSPNet is not invoked here.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import torch
from torch import nn
from torchvision.models import mobilenet_v2,MobileNet_V2_Weights


def imagenet_weight_path()->Path:
    url=MobileNet_V2_Weights.IMAGENET1K_V1.url
    return Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(url).path).name


def load_imagenet_weights(base:nn.Module)->str:
    path=imagenet_weight_path()
    if not path.is_file():
        raise FileNotFoundError(f'ImageNet MobileNetV2 weights not cached: {path}. Download once with torchvision before training.')
    state=torch.load(path,map_location='cpu',weights_only=True)
    base.load_state_dict(state,strict=True)
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    return h


class MobileNetV2OS16(nn.Module):
    def __init__(self,*,pretrained:bool):
        super().__init__()
        base=mobilenet_v2(weights=None)
        self.pretrained_sha256=load_imagenet_weights(base) if pretrained else None
        # Original stride reduction to 14x14 (OS16): block 14 is the final /2.
        # All downstream depthwise convolutions use dilation 2.
        for idx in range(14,len(base.features)):
            block=base.features[idx]
            for conv in block.modules():
                if isinstance(conv,nn.Conv2d) and conv.groups==conv.in_channels and conv.kernel_size==(3,3):
                    if idx==14 and conv.stride==(2,2):
                        conv.stride=(1,1)
                    conv.dilation=(2,2)
                    conv.padding=(2,2)
        self.features=base.features
        self.classifier=nn.Linear(1280,1)
        nn.init.normal_(self.classifier.weight,mean=0.,std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward_features(self,x:torch.Tensor)->torch.Tensor:
        f=self.features(x)
        if tuple(f.shape[1:])!=(1280,14,14):
            raise ValueError(f'MobileNetV2-OS16 feature shape mismatch: {tuple(f.shape)}')
        return f

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.classifier(self.forward_features(x).mean(dim=(2,3))).flatten()
