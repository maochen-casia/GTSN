"""Seed training and DataLoader workers independently."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, CPU and CUDA random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Use PyTorch's per-worker seed for Python and NumPy randomness."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def require_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; run Docker with --gpus all")
    return device

