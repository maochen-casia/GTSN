"""RGB-only metric geometry, spatial tokens and a 30-step joint proposal."""
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tsn.models.joint_head import JointProposalHead


class PatchPositions:
    def __init__(self):
        self.cache = {}

    def __call__(self, batch, height, width, device):
        key = height, width, device
        if key not in self.cache:
            self.cache[key] = torch.cartesian_prod(torch.arange(height, device=device), torch.arange(width, device=device))
        return self.cache[key][None].expand(batch, -1, -1).clone()


class RGBPerception(nn.Module):
    """Pi3 image encoder + freshly initialized small decoder and navigation heads.

    Optional published weights initialize only the image encoder. No previous
    navigation checkpoint is needed. Outputs use base-frame normalized XYZ and
    predicted task scores; depth and expert futures are training targets only.
    """
    def __init__(self, config, initialize_encoder=True):
        super().__init__()
        from pi3.models.pi3 import Pi3
        from pi3.models.dinov2.hub.backbones import dinov2_vitl14_reg
        from pi3.models.dinov2.layers import Mlp
        from pi3.models.layers.attention import FlashAttentionRope
        from pi3.models.layers.block import BlockRope
        from pi3.models.layers.pos_embed import RoPE2D
        cfg = config['perception']
        self.input_hw = tuple(cfg['input_hw'])
        if len(self.input_hw) != 2 or any(v <= 0 or v % 14 for v in self.input_hw):
            raise ValueError('RGB input dimensions must be positive multiples of 14')
        self.output_hw = config['maps']['height'], config['maps']['width']
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        del self.encoder.mask_token
        if initialize_encoder and cfg['pretrained_weights']:
            from safetensors import safe_open
            with safe_open(Path(cfg['pretrained_weights']), framework='pt', device='cpu') as weights:
                state = {key[len('encoder.'):]: weights.get_tensor(key) for key in weights.keys() if key.startswith('encoder.')}
            self.encoder.load_state_dict(state, strict=True)
        self.dec_embed_dim, self.patch_size, self.patch_start_idx = 384, 14, 5
        self.pos_type = 'rope100'
        self.rope, self.position_getter = RoPE2D(freq=100.), PatchPositions()
        self.encoder_projection = nn.Linear(1024, 384)
        self.register_token = nn.Parameter(torch.empty(1, 1, 5, 384))
        nn.init.normal_(self.register_token, std=1e-6)
        self.decoder = nn.ModuleList([BlockRope(dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True,
            proj_bias=True, ffn_bias=True, drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU, ffn_layer=Mlp, init_values=.01, qk_norm=True,
            attn_class=FlashAttentionRope, rope=self.rope) for _ in range(24)])
        self._decode = Pi3.decode
        self.condition = nn.Sequential(nn.Linear(41, 384), nn.SiLU(), nn.Linear(384, 768))
        self.point_head = nn.Linear(768, 14*14*3)
        self.goal_head = nn.Linear(768, 14*14*2)
        self.action_map_head = nn.Linear(768, 14*14)
        nn.init.constant_(self.goal_head.bias, -3.)
        nn.init.constant_(self.action_map_head.bias, -3.)
        self.joint_head = JointProposalHead(**config['joint_head'])
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def unpatchify(self, values, channels):
        h, w = self.input_hw
        return values.reshape(-1, h//14, w//14, 14, 14, channels).permute(0, 5, 1, 3, 2, 4).reshape(-1, channels, h, w)

    def forward(self, rgb, state, calibration, pose):
        """Return joint offsets, (B,16,768) tokens, pooled and dense six-channel maps."""
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError('Expected batched uint8 RGB images (B,H,W,3)')
        h, w = self.input_hw
        image = F.interpolate(rgb.permute(0, 3, 1, 2).float()/255, (h, w), mode='bilinear', align_corners=False)
        hidden = self.encoder((image-self.image_mean)/self.image_std, is_training=True)['x_norm_patchtokens']
        hidden, _ = self._decode(self, self.encoder_projection(hidden), 1, h, w)
        hidden = hidden[:, self.patch_start_idx:]
        K = calibration.float().clone()
        K[:, 0] /= rgb.shape[2]; K[:, 1] /= rgb.shape[1]
        hidden = hidden+self.condition(torch.cat((state, pose.flatten(1), K.flatten(1)), -1))[:, None]
        points = 2*self.unpatchify(self.point_head(hidden), 3).tanh()
        goals = self.unpatchify(self.goal_head(hidden), 2).sigmoid()
        action_map = self.unpatchify(self.action_map_head(hidden), 1).sigmoid()
        dense = F.interpolate(torch.cat((points, goals, action_map), 1), self.output_hw, mode='bilinear', align_corners=False)
        tokens = F.adaptive_avg_pool2d(hidden.transpose(1, 2).reshape(-1, 768, h//14, w//14), (4, 4)).flatten(2).transpose(1, 2)
        geometry = F.adaptive_avg_pool2d(dense, (4, 4)).flatten(2).transpose(1, 2)
        return self.joint_head(dense, state), tokens, geometry, dense
