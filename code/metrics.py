"""Validation metrics (no threshold selection from TEST)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score


def binary_metrics(y,logits):
    y=np.asarray(y,dtype=np.int64);z=np.asarray(logits,dtype=np.float64)
    if y.ndim!=1 or z.ndim!=1 or len(y)!=len(z) or len(y)<2:
        raise ValueError('Invalid score shape')
    if set(np.unique(y))!={0,1} or not np.isfinite(z).all():
        raise ValueError('Invalid labels or non-finite predictions')
    z=np.clip(z,-80,80)
    scores=1.0/(1.0+np.exp(-z))
    return float(roc_auc_score(y,scores)),float(average_precision_score(y,scores))
