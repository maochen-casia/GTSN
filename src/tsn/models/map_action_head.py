"""Pi3 residual CNN and state MLP for joint action-chunk regression."""

from __future__ import annotations

import torch
from torch import nn


class SpatialStage(nn.Module):
    """Downsample once, then refine with a two-convolution residual branch."""

    def __init__(self, input_channels: int, output_channels: int, first: bool) -> None:
        """Build a stride-two convolution and shape-preserving residual block.

        Args:
            input_channels (int): Number of channels in the input feature map.
            output_channels (int): Number of output channels, divisible by 8
                for GroupNorm.
            first (bool): Use a 5 x 5 downsampling kernel if True, else 3 x 3.

        Returns:
            None. Initializes convolution and normalization parameters.
        """
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
        """Downsample and refine a batch of spatial feature maps.

        Args:
            maps (torch.Tensor): Floating maps (B, input_channels, H, W).

        Returns:
            torch.Tensor: Floating features
            (B, output_channels, ceil(H/2), ceil(W/2)). The residual branch
            preserves the downsampled resolution and channel count.
        """
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
        """Build spatial/state encoders and the bounded joint regression head.

        Args:
            state_dim (int): State width; must be 16.
            action_dim (int): Number of arm joints; must be 7.
            chunk_size (int): Positive number L of future joint targets.
            action_scale_rad (float): Positive per-coordinate offset bound, rad.
            encoder_channels (list[int]): Nonempty sequence of positive stage
                widths, each divisible by 8; each stage halves spatial size.
            spatial_pool_size (int): Positive pooled grid side length.
            map_feature_dim (int): Positive width of the pooled map embedding.
            state_feature_dim (int): Positive width of the state embedding.
            hidden_dim (int): Positive width of the fused regression layer.
            dropout (float): Training dropout probability in [0, 1).

        Returns:
            None. Initializes trainable layers and stores action horizon/scale.

        Raises:
            ValueError: State/action dimensions or layer settings are invalid.
        """
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
        """Fuse predicted geometry with proprioception and the requested goal.

        Args:
            maps (torch.Tensor): Floating maps (B, 6, H, W): normalized
                base-frame point XYZ, projected goal, visible goal, and future
                action score. The first three channels use the training map
                centre/scale; the last three are scores in [0, 1].
            state (torch.Tensor): Floating state (B, 16): seven arm angles / pi,
                two finger positions / .04 m, normalized goal XYZ, and a unit
                goal quaternion (wxyz). Batch size/device must match maps.

        Returns:
            torch.Tensor: Floating joint offsets (B, chunk_size, 7), radians,
            bounded by +/-action_scale_rad. Every target is relative to the
            same observed qpos; targets are not accumulated along the horizon.
            Output dtype follows the active autocast context.
        """
        features = torch.cat((self.map_projection(self.spatial(maps)), self.state_projection(state)), dim=1)
        output = self.output(features).reshape(-1, self.chunk_size, self.action_dim)
        return output.tanh() * self.action_scale_rad
