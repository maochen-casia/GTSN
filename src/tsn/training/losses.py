"""Supervision for the main model; depth and expert futures never enter control."""
import torch
from torch.nn import functional as F


def masked_mean(values, valid):
    return (values*valid).sum()/valid.sum().clamp_min(1)


def geometry_nll(auxiliary, target, valid):
    error = (auxiliary['mean'].float()-target.float()).square()
    nll = .5*(error*(-auxiliary['logvar'].float()).exp()+auxiliary['logvar'].float())
    return masked_mean(nll.mean(-1), valid)


def point_quantile_loss(prediction, target, valid, quantile=.9):
    residual = target-prediction
    values = torch.maximum(quantile*residual, (quantile-1)*residual)
    per_row = (values*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return masked_mean(per_row, valid.any(-1))


def map_loss(prediction, target, valid):
    point = F.huber_loss(prediction[:, :3].float(), target[:, :3], delta=.05, reduction='none').mean(1)
    task = F.binary_cross_entropy(prediction[:, 3:].float().clamp(1e-5, 1-1e-5), target[:, 3:])
    return masked_mean(point, valid)+.1*task
