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

from tsn.models.geometry_policy import GeometryPolicy


class DevicePositionGetter:
    """Keep Pi3's patch positions separate for each DataParallel device."""
    def __init__(self):
        self.cache = {}

    def __call__(self, batch, height, width, device):
        key = (height, width, device)
        if key not in self.cache:
            self.cache[key] = torch.cartesian_prod(torch.arange(height, device=device),
                                                   torch.arange(width, device=device))
        return self.cache[key].view(1, height * width, 2).expand(batch, -1, -1).clone()


class Pi3MapPolicy(nn.Module):
    uses_predicted_maps = True

    def __init__(self, config, initialize_backbone=True):
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
        self.action_policy = GeometryPolicy(**{k: config[k] for k in inspect.signature(GeometryPolicy).parameters})
        self.chunk_size = self.action_policy.chunk_size
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def _unpatchify(self, values, channels):
        h, w = self.input_hw
        values = values.reshape(-1, h // 14, w // 14, 14, 14, channels)
        return values.permute(0, 5, 1, 3, 2, 4).reshape(-1, channels, h, w)

    def forward_with_maps(self, rgb, state, calibration, camera_transform):
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError('Expected sensor RGB as batched uint8 HWC images')
        h, w = self.input_hw
        image = F.interpolate(rgb.permute(0, 3, 1, 2).float() / 255,
                              size=(h, w), mode='bilinear', align_corners=False)
        hidden = self.encoder((image - self.image_mean) / self.image_std,
                              is_training=True)['x_norm_patchtokens']
        hidden = self.encoder_projection(hidden)
        hidden, _ = self._decode(self, hidden, 1, h, w)
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
        return self.action_policy(predicted_maps, state), predicted_maps

    def forward(self, rgb, state, calibration, camera_transform, return_maps=False):
        # DataParallel creates worker autocast contexts; select bfloat16 explicitly.
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16,
                            enabled=rgb.device.type == 'cuda'):
            result = self.forward_with_maps(rgb, state, calibration, camera_transform)
        return result if return_maps else result[0]
