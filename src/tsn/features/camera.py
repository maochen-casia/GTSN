"""Calibrated OpenCV rays with align_corners=False pixel-center resizing."""
import torch
from torch.nn import functional as F


def resize_intrinsics(K, source_hw, target_hw):
    result = K.float().clone()
    sy, sx = target_hw[0]/source_hw[0], target_hw[1]/source_hw[1]
    result[..., 0, :] *= sx
    result[..., 1, :] *= sy
    result[..., 0, 2] += (sx-1)/2
    result[..., 1, 2] += (sy-1)/2
    return result


def camera_rays(K, source_hw, target_hw):
    calibration = resize_intrinsics(K, source_hw, target_hw)
    v, u = torch.meshgrid(torch.arange(target_hw[0], device=K.device),
                         torch.arange(target_hw[1], device=K.device), indexing='ij')
    pixels = torch.stack((u, v, torch.ones_like(u)), -1).float()
    return torch.einsum('bij,hwj->bhwi', torch.linalg.inv(calibration), pixels)


def depth_to_base(depth, K, source_hw, pose):
    """Predict camera Z, enforce each pixel's ray, then apply measured T_B_C."""
    rays = camera_rays(K, source_hw, depth.shape[-2:])
    camera = rays*depth.float()[..., None]
    return torch.einsum('bij,bhwj->bhwi', pose[:, :3, :3].float(), camera)+pose[:, None, None, :3, 3].float()


def resize_observation(sample, target_hw):
    """Keep RGB, depth and K registered while admitting mixed-resolution batches."""
    if target_hw is None or tuple(sample['depth'].shape) == tuple(target_hw):
        return sample
    source_hw = sample['depth'].shape
    sample['K'] = resize_intrinsics(sample['K'], source_hw, target_hw)
    sample['depth'] = F.interpolate(sample['depth'][None, None], target_hw, mode='nearest-exact')[0, 0]
    if 'rgb' in sample:
        image = sample['rgb'].permute(2, 0, 1)[None].float()
        sample['rgb'] = F.interpolate(image, target_hw, mode='bilinear', align_corners=False,
                                     antialias=True)[0].round().clamp(0, 255).byte().permute(1, 2, 0)
    return sample
