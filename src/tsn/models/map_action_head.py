"""Pi3 residual CNN and state MLP for joint action-chunk regression."""

from __future__ import annotations

import torch
from torch import nn


class SpatialStage(nn.Module):
    """Downsample once, then refine with a two-convolution residual branch."""

    def __init__(self, input_channels: int, output_channels: int, first: bool) -> None:
        super().__init__()
        kernel = 5 if first else 3
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel, stride=2, padding=kernel // 2, bias=False),
            nn.GroupNorm(8, output_channels), nn.SiLU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        encoded = self.downsample(maps)
        return torch.nn.functional.silu(encoded + self.refine(encoded))


class MapActionHead(nn.Module):
    """Fuse maps (B,6,H,W) and state (B,16) into residual arm targets (B,L,7).

    Each horizon target is a displacement from the SAME observed qpos, rather
    than an increment from the preceding action. Bounded tanh output is in radians.
    RGB and route labels are not inputs to this policy.
    """

    def __init__(self, *, state_dim: int, action_dim: int, chunk_size: int,
                 action_scale_rad: float, encoder_channels: list[int], spatial_pool_size: int,
                 map_feature_dim: int, state_feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if state_dim != 16 or action_dim != 7 or chunk_size <= 0 or action_scale_rad <= 0:
            raise ValueError("Expected Panda state=16, arm actions=7 and positive horizon/scale")
        if (not encoder_channels or any(c <= 0 or c % 8 for c in encoder_channels)
                or min(spatial_pool_size, map_feature_dim, state_feature_dim, hidden_dim) <= 0
                or not 0 <= dropout < 1):
            raise ValueError("Invalid policy encoder or head dimensions")
        self.chunk_size, self.action_dim = chunk_size, action_dim
        self.action_scale_rad = action_scale_rad
        stages = []
        previous = 6
        for index, channels in enumerate(encoder_channels):
            stages.append(SpatialStage(previous, channels, first=index == 0))
            previous = channels
        self.spatial = nn.Sequential(*stages)
        self.map_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(spatial_pool_size), nn.Flatten(),
            nn.Linear(previous * spatial_pool_size**2, map_feature_dim),
            nn.LayerNorm(map_feature_dim), nn.SiLU(),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, state_feature_dim), nn.LayerNorm(state_feature_dim), nn.SiLU(),
            nn.Linear(state_feature_dim, state_feature_dim), nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(map_feature_dim + state_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def forward(self, maps: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = torch.cat((self.map_projection(self.spatial(maps)), self.state_projection(state)), dim=1)
        output = self.output(features).reshape(-1, self.chunk_size, self.action_dim)
        return output.tanh() * self.action_scale_rad

