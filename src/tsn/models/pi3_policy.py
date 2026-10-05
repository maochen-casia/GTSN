"""Pi3 small decoder with learned geometry maps and action-chunk regression.

Only RGB, proprioception, the requested goal, and measured camera calibration
enter this module. Depth and expert futures are auxiliary supervision elsewhere.
The upstream ViT-L image encoder is retained so published Pi3 weights load
exactly; the official 384-wide, 24-block small decoder is newly initialized.
"""
from functools import partial
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tsn.models.map_action_head import MapActionHead


class DevicePositionGetter:
    """Keep Pi3's patch positions separate for each DataParallel device."""
    def __init__(self):
        """Initialize an empty per-device grid cache.

        Args:
            None.

        Returns:
            None. Cached tensors are created lazily on the requested device.
        """
        self.cache = {}

    def __call__(self, batch, height, width, device):
        """Return row/column indices for every patch in a rectangular grid.

        Args:
            batch (int): Number of copies of the grid to return.
            height (int): Number of patch rows, not image pixels.
            width (int): Number of patch columns.
            device (torch.device | str): Device on which to construct positions.

        Returns:
            torch.Tensor: Int64 positions (batch, height * width, 2) in
            row-major order, each pair (row, column). The returned clone can
            be modified without changing the cached grid.
        """
        key = (height, width, device)
        if key not in self.cache:
            self.cache[key] = torch.cartesian_prod(torch.arange(height, device=device),
                                                   torch.arange(width, device=device))
        return self.cache[key].view(1, height * width, 2).expand(batch, -1, -1).clone()


class Pi3MapPolicy(nn.Module):
    """Predict six geometry/task maps and joint offsets from one RGB view.

    The small decoder concatenates two 384-wide layer outputs into 768-wide
    patch features. State and camera calibration condition each patch before
    dense map prediction; a CNN/MLP converts those maps into joint targets.
    """
    uses_predicted_maps = True

    def __init__(self, config, initialize_backbone=True):
        """Construct the image encoder, small decoder, map heads, and policy.

        Args:
            config (dict[str, object]): Model configuration. 'pi3' contains
                decoder_size='small', input_hw=(height, width) in pixels
                (positive multiples of 14), and pretrained_weights path.
                'maps' contains output height/width; the remaining action-head
                settings match MapActionHead.__init__ and include chunk_size.
            initialize_backbone (bool): Load only the pretrained image encoder
                from safetensors when True. False skips file loading so a full
                project state dict can subsequently restore the model.

        Returns:
            None. Initializes the modules and normalization buffers. Decoder,
            conditioning, and output heads are newly initialized in either case.

        Raises:
            ValueError: Decoder size or input image dimensions are unsupported.
        """
        super().__init__()
        from pi3.models.pi3 import Pi3
        from pi3.models.dinov2.hub.backbones import dinov2_vitl14_reg
        from pi3.models.dinov2.layers import Mlp
        from pi3.models.layers.attention import FlashAttentionRope
        from pi3.models.layers.block import BlockRope
        from pi3.models.layers.pos_embed import RoPE2D

        cfg = config['pi3']
        if cfg['decoder_size'] != 'small':
            raise ValueError('This experiment requires the official Pi3 small decoder')
        self.input_hw = tuple(cfg['input_hw'])
        if len(self.input_hw) != 2 or any(s <= 0 or s % 14 for s in self.input_hw):
            raise ValueError('Pi3 image dimensions must be positive multiples of 14')
        self.output_hw = (config['maps']['height'], config['maps']['width'])
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        del self.encoder.mask_token
        self.initialization = None
        if initialize_backbone:
            from safetensors import safe_open
            path = Path(cfg['pretrained_weights'])
            with safe_open(path, framework='pt', device='cpu') as weights:
                encoder_weights = {key[len('encoder.'):]: weights.get_tensor(key)
                                   for key in weights.keys() if key.startswith('encoder.')}
            self.encoder.load_state_dict(encoder_weights, strict=True)
            self.initialization = {'source': str(path), 'loaded_encoder_tensors': len(encoder_weights),
                                   'decoder': 'random small', 'policy_heads': 'random'}
        self.dec_embed_dim = 384
        self.patch_size = 14
        self.patch_start_idx = 5
        self.pos_type = 'rope100'
        self.rope = RoPE2D(freq=100.0)
        self.position_getter = DevicePositionGetter()
        # Upstream Pi3.small lacks the required 1024 -> 384 projection.
        self.encoder_projection = nn.Linear(1024, self.dec_embed_dim)
        self.register_token = nn.Parameter(torch.empty(1, 1, 5, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)
        self.decoder = nn.ModuleList([
            BlockRope(dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True,
                      proj_bias=True, ffn_bias=True, drop_path=0.0,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                      ffn_layer=Mlp, init_values=0.01, qk_norm=True,
                      attn_class=FlashAttentionRope, rope=self.rope)
            for _ in range(24)
        ])
        self._decode = Pi3.decode
        self.condition = nn.Sequential(nn.Linear(16 + 16 + 9, 384), nn.SiLU(), nn.Linear(384, 768))
        self.point_head = nn.Linear(768, 14 * 14 * 3)
        self.goal_head = nn.Linear(768, 14 * 14 * 2)
        self.action_map_head = nn.Linear(768, 14 * 14)
        nn.init.constant_(self.goal_head.bias, -3.0)
        nn.init.constant_(self.action_map_head.bias, -3.0)
        self.action_policy = MapActionHead(**{k: config[k] for k in inspect.signature(MapActionHead).parameters})
        self.chunk_size = self.action_policy.chunk_size
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def _unpatchify(self, values, channels):
        """Reassemble per-patch pixel predictions into a channel-first image.

        Args:
            values (torch.Tensor): Floating patch predictions
                (B, P, 14 * 14 * channels), where P=(h/14)*(w/14) and
                (h, w)=self.input_hw; patches are in row-major order.
            channels (int): Number of scalar channels at each output pixel.

        Returns:
            torch.Tensor: Dense predictions (B, channels, h, w), preserving
            input dtype and device; no interpolation or activation is applied.
        """
        h, w = self.input_hw
        values = values.reshape(-1, h // 14, w // 14, 14, 14, channels)
        return values.permute(0, 5, 1, 3, 2, 4).reshape(-1, channels, h, w)

    def forward_with_maps(self, rgb, state, calibration, camera_transform, return_features=False,
                          return_maps=False):
        """Run perception and expose maps or pooled features alongside actions.

        Args:
            rgb (torch.Tensor): Uint8 sensor RGB (B, H, W, 3), values 0..255;
                resized internally to input_hw and ImageNet-normalized.
            state (torch.Tensor): Floating normalized state (B, 16): seven
                arm angles / pi, two finger positions / .04 m, normalized goal
                XYZ, and unit goal quaternion (wxyz).
            calibration (torch.Tensor): Floating intrinsics (B, 3, 3) in
                original sensor pixels; row 0/1 normalized by W/H internally.
            camera_transform (torch.Tensor): Floating camera-to-robot-base
                transforms (B, 4, 4), with translation in metres.
            return_features (bool): Include pooled spatial tokens and geometry.
            return_maps (bool): Include dense maps when returning features.
                With return_features=False this method always returns maps.

        Returns:
            tuple[torch.Tensor, ...]: If return_features=False, (action, maps).
            Otherwise (action, tokens, geometry), with maps appended if
            return_maps=True. Floating tensor shapes and meanings:
            action (B, L, 7) is radians relative to current arm joints, where
            L=self.chunk_size; maps (B, 6, h_m, w_m) uses configured output_hw;
            tokens (B, 16, 768) and geometry (B, 16, 6) are 4 x 4 pooled cells.
            Map channels are normalized base XYZ (bounded by +/-2), projected
            goal, visible goal, and future-action score (each in [0, 1]). XYZ
            uses the training map centre/scale, normally (.65, 0, .22) and
            (.55, .55, .50) m. Dtypes follow the active autocast context.

        Raises:
            ValueError: RGB is not a batched uint8 channel-last image tensor.
        """
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError('Expected sensor RGB as batched uint8 HWC images')
        h, w = self.input_hw
        image = F.interpolate(rgb.permute(0, 3, 1, 2).float() / 255,
                              size=(h, w), mode='bilinear', align_corners=False)
        hidden = self.encoder((image - self.image_mean) / self.image_std,
                              is_training=True)['x_norm_patchtokens']
        hidden = self.encoder_projection(hidden)
        hidden, _ = self._decode(self, hidden, 1, h, w)
        # Remove five decoder register tokens, leaving P spatial tokens of
        # width 768 (the concatenated outputs of the last two decoder blocks).
        hidden = hidden[:, self.patch_start_idx:]
        # Intrinsics are normalized by the sensor image dimensions, not target-map size.
        K = calibration.float().clone()
        K[:, 0] /= rgb.shape[2]
        K[:, 1] /= rgb.shape[1]
        condition = self.condition(torch.cat((state, camera_transform.flatten(1), K.flatten(1)), -1))
        hidden = hidden + condition[:, None]
        point = 2 * self._unpatchify(self.point_head(hidden), 3).tanh()
        goal = self._unpatchify(self.goal_head(hidden), 2).sigmoid()
        action_map = self._unpatchify(self.action_map_head(hidden), 1).sigmoid()
        predicted_maps = F.interpolate(torch.cat((point, goal, action_map), 1),
                                       size=self.output_hw, mode='bilinear', align_corners=False)
        action = self.action_policy(predicted_maps, state)
        if return_features:
            grid = hidden.transpose(1, 2).reshape(-1, 768, h // 14, w // 14)
            tokens = F.adaptive_avg_pool2d(grid, (4, 4)).flatten(2).transpose(1, 2)
            geometry = F.adaptive_avg_pool2d(predicted_maps, (4, 4)).flatten(2).transpose(1, 2)
            if return_maps:
                return action, tokens, geometry, predicted_maps
            return action, tokens, geometry
        return action, predicted_maps

    def forward(self, rgb, state, calibration, camera_transform, return_maps=False, return_features=False):
        """Predict joint offsets, optionally returning perception intermediates.

        Args:
            rgb (torch.Tensor): Uint8 RGB images (B, H, W, 3), values 0..255.
            state (torch.Tensor): Floating normalized policy state (B, 16),
                with the entry meanings described in forward_with_maps.
            calibration (torch.Tensor): Floating sensor intrinsics (B, 3, 3), px.
            camera_transform (torch.Tensor): Floating camera-to-base poses
                (B, 4, 4), translations in metres.
            return_maps (bool): Include six dense predicted maps.
            return_features (bool): Include pooled tokens and geometry.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: With neither flag, action
            (B, L, 7), joint offsets in radians. Maps only: (action, maps).
            Features only: (action, tokens, geometry). Both: (action, tokens,
            geometry, maps). Tokens are (B, 16, 768), geometry (B, 16, 6), and
            maps (B, 6, h_m, w_m); channel meanings are in forward_with_maps.
            Outputs are floating tensors; CUDA uses bfloat16 autocast, while
            CPU runs without this method enabling autocast. Inputs and module
            must be on the same device.
        """
        # DataParallel creates worker autocast contexts; select bfloat16 explicitly.
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16,
                            enabled=rgb.device.type == 'cuda'):
            result = self.forward_with_maps(rgb, state, calibration, camera_transform,
                                            return_features, return_maps)
        return result if return_maps or return_features else result[0]
