"""Utilities for deterministic CV, preprocessing, and metrics."""
import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def zscore(signal, axis=-1, eps=1e-8):
    signal = np.asarray(signal, dtype=np.float32)
    mu = np.mean(signal, axis=axis, keepdims=True)
    sigma = np.std(signal, axis=axis, keepdims=True)
    return (signal - mu) / (sigma + eps)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
