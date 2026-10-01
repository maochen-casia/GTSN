"""Generate six geometry channels on the device, without preprocessing files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MapSettings:
    height: int = 80
    width: int = 80
    goal_sigma_m: float = 0.04
    action_sigma_m: float = 0.05
    action_decay_steps: float = 10.0
    cutoff_sigma: float = 3.0
    occlusion_tolerance_m: float = 0.015
    near_m: float = 0.02
    far_m: float = 2.0
    point_center_m: tuple[float, ...] = (0.65, 0.0, 0.22)
    point_scale_m: tuple[float, ...] = (0.55, 0.55, 0.50)
    trajectory_block_size: int = 8

    def __post_init__(self) -> None:
        if min(self.height, self.width, self.trajectory_block_size) <= 0:
            raise ValueError("Map dimensions and trajectory block size must be positive")
        if min(self.goal_sigma_m, self.action_sigma_m, self.action_decay_steps, self.cutoff_sigma) <= 0:
            raise ValueError("Map Gaussian scales, cutoff and decay must be positive")
        if not 0 < self.near_m < self.far_m or self.occlusion_tolerance_m < 0:
            raise ValueError("Invalid map depth range or occlusion tolerance")
        if len(self.point_center_m) != 3 or len(self.point_scale_m) != 3 or min(self.point_scale_m) <= 0:
            raise ValueError("Point normalization requires three positive scales")


class GeometryMaps(nn.Module):
    """Backproject depth and score metric Gaussians along OpenCV camera rays.

    Input: depth (B,H,W) in metres, K (B,3,3), T_B_C (B,4,4), goal (B,3),
    future TCP positions (B,L,3), valid future mask (B,L). Output: (B,6,h,w)
    in point XYZ / projected goal / visible goal / future action order.
    Rays are computed per sample so episodes may have different calibration.
    Future trajectories are processed in blocks to bound GPU working memory.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.settings = MapSettings(**config)
        cfg = self.settings
        v, u = torch.meshgrid(torch.arange(cfg.height), torch.arange(cfg.width), indexing="ij")
        self.register_buffer("pixels", torch.stack((u, v, torch.ones_like(u)), -1).float(), persistent=False)
        self.register_buffer("center", torch.tensor(cfg.point_center_m).float(), persistent=False)
        self.register_buffer("scale", torch.tensor(cfg.point_scale_m).float(), persistent=False)

    @torch.no_grad()
    def forward(self, depth: torch.Tensor, K: torch.Tensor, T_B_C: torch.Tensor,
                goal: torch.Tensor, future_ee: torch.Tensor,
                valid_future: torch.Tensor) -> torch.Tensor:
        cfg = self.settings
        batch, source_h, source_w = depth.shape
        # Match align_corners=False pixel centers in nearest-exact downsampling.
        calibration = K.float().clone()
        sx, sy = cfg.width / source_w, cfg.height / source_h
        calibration[:, 0, :] *= sx
        calibration[:, 1, :] *= sy
        calibration[:, 0, 2] += (sx - 1) / 2
        calibration[:, 1, 2] += (sy - 1) / 2
        rays = torch.einsum("bij,hwj->bhwi", torch.linalg.inv(calibration), self.pixels)
        z = F.interpolate(depth[:, None].float(), size=(cfg.height, cfg.width), mode="nearest-exact")[:, 0]
        observed = torch.isfinite(z) & (z >= cfg.near_m) & (z <= cfg.far_m)
        z = torch.where(observed, z, torch.zeros_like(z))
        rotation, origin = T_B_C[:, :3, :3].float(), T_B_C[:, :3, 3].float()
        xyz_base = torch.einsum("bij,bhwj->bhwi", rotation, rays * z[..., None]) + origin[:, None, None]
        points = ((xyz_base - self.center) / self.scale).clamp(-2, 2)
        points = torch.where(observed[..., None], points, 0).permute(0, 3, 1, 2)

        def camera_positions(positions: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bji,blj->bli", rotation, positions.float() - origin[:, None])

        goal_camera = camera_positions(goal[:, None])
        goal_mask = torch.ones((batch, 1), dtype=torch.bool, device=depth.device)
        projected = self._scores(rays, z, observed, goal_camera, goal_mask, cfg.goal_sigma_m, False)
        visible = self._scores(rays, z, observed, goal_camera, goal_mask, cfg.goal_sigma_m, True)
        future_camera = camera_positions(future_ee)
        action = torch.zeros_like(z)
        for start in range(0, future_ee.shape[1], cfg.trajectory_block_size):
            stop = min(start + cfg.trajectory_block_size, future_ee.shape[1])
            horizon = torch.arange(start, stop, device=z.device, dtype=torch.float32)
            scores = self._scores(rays, z, observed, future_camera[:, start:stop],
                                  valid_future[:, start:stop], cfg.action_sigma_m, True,
                                  torch.exp(-horizon / cfg.action_decay_steps))
            action = torch.maximum(action, scores)
        return torch.cat((points, projected[:, None], visible[:, None], action[:, None]), dim=1).contiguous()

    def _scores(self, rays: torch.Tensor, z: torch.Tensor, observed: torch.Tensor,
                positions: torch.Tensor, mask: torch.Tensor, sigma: float, occlusion: bool,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """Max-reduce Gaussian distance to finite rays; projected goals ignore missing depth."""
        cfg = self.settings
        dot = torch.einsum("bhwc,blc->blhw", rays, positions)
        extent = z[:, None] + cfg.occlusion_tolerance_m if occlusion else cfg.far_m
        along = (dot / rays.square().sum(-1)[:, None]).clamp_min(cfg.near_m)
        along = torch.minimum(along, torch.as_tensor(extent, device=z.device))
        delta = rays[:, None] * along[..., None] - positions[:, :, None, None]
        distance_squared = delta.square().sum(-1)
        in_front = (positions[:, :, 2] >= cfg.near_m) & (positions[:, :, 2] <= cfg.far_m)
        allowed = mask[:, :, None, None] & in_front[:, :, None, None]
        if occlusion:
            allowed = allowed & observed[:, None] & (positions[:, :, 2, None, None] <= extent)
        score = torch.exp(-0.5 * distance_squared / (sigma**2))
        score = torch.where(allowed & (distance_squared <= (sigma * cfg.cutoff_sigma)**2), score, 0)
        if weights is not None:
            score = score * weights[None, :, None, None]
        return score.amax(1)

