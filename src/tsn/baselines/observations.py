"""Identical calibrated preprocessing for cached demonstrations and live frames."""
import torch
from torch.nn import functional as F

from tsn.features.camera import depth_to_base, resize_intrinsics
from tsn.features.maps import MapSettings
from tsn.features.state import policy_state


def observation_state(qpos, goal, K, pose, source_hw, map_settings):
    state = policy_state(qpos, goal, MapSettings(**map_settings))
    calibration = resize_intrinsics(K, source_hw, (84, 112))[:, :2].flatten(1)
    calibration = calibration / calibration.new_tensor([112, 112, 112, 84, 84, 84])
    return torch.cat((state, calibration, pose[:, :3].flatten(1)), -1)  # 34 values


def rgb_image(rgb):
    image = rgb.permute(0, 3, 1, 2).float()
    image = F.interpolate(image, (84, 112), mode='bilinear', align_corners=False, antialias=True)
    return image.round().clamp(0, 255).byte().permute(0, 2, 3, 1)


def point_cloud(depth, K, pose, count=512):
    """Visible base-frame XYZ, cropped to the workspace, sampled without labels.

    Deterministic uniform sampling in raster order replaces upstream FPS. Empty
    views get a zero cloud; sparse views repeat visible points. No expert futures,
    ground-truth object geometry, route label or learned GTSN component enters.
    """
    source_hw = depth.shape[-2:]
    z = F.interpolate(depth[:, None].float(), (48, 64), mode='nearest-exact')[:, 0]
    points = depth_to_base(z, K, source_hw, pose).flatten(1, 2)
    valid = torch.isfinite(points).all(-1) & torch.isfinite(z.flatten(1))
    valid &= (z.flatten(1) > .02) & (z.flatten(1) < 2.)
    valid &= (points[..., 0] > .05) & (points[..., 0] < 1.2)
    valid &= (points[..., 1].abs() < .8) & (points[..., 2] > -.02) & (points[..., 2] < 1.2)
    result = points.new_zeros(len(depth), count, 3)
    for row in range(len(depth)):
        visible = points[row, valid[row]]
        if len(visible):
            indices = torch.linspace(0, len(visible)-1, count, device=points.device).round().long()
            result[row] = visible[indices]
    return result
