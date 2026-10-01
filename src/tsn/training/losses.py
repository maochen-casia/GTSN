"""Masked, horizon-weighted joint residual imitation loss."""

import torch
from torch.nn import functional as F


def imitation_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                   beta: float, decay_steps: float, floor: float) -> torch.Tensor:
    """Smooth-L1 over (B,L,7); exclude padded horizons and normalize by valid weight."""
    if beta <= 0 or decay_steps <= 0 or not 0 <= floor <= 1:
        raise ValueError("Invalid Huber beta or horizon weighting")
    horizon = torch.arange(target.shape[1], device=target.device)
    weights = (floor + (1 - floor) * torch.exp(-horizon / decay_steps))[None] * valid
    errors = F.smooth_l1_loss(prediction.float(), target.float(), beta=beta, reduction="none")
    return (errors * weights[..., None]).sum() / (weights.sum().clamp_min(1) * target.shape[2])

