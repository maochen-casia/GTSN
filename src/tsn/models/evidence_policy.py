"""Route-conditioned retrieval of uncertain, causally observed scene evidence.

The existing controller proposes a route. Each knot retrieves geometry in metric
space using the expected Gaussian proximity under the predicted point variance.
A null evidence token prevents distant observations from being treated as local
support. No obstacle mesh, measured depth, or future observation enters forward.
"""
import math

import torch
from torch import nn

from tsn.models.compact_policy import CompactRouteHead, goal_xyz


class EvidenceRouteHead(nn.Module):
    variants = ('full', 'deterministic', 'current', 'global', 'residual')

    def __init__(self, base=None, variant='full', radius=.12, correction=.06):
        super().__init__()
        if variant not in self.variants:
            raise ValueError(f'Unknown evidence variant: {variant}')
        self.base = base if base is not None else CompactRouteHead()
        self.base.requires_grad_(False)
        self.variant, self.radius, self.correction = variant, radius, correction
        self.register_buffer('variance_scale', torch.ones(3))
        self.visual = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 64), nn.SiLU())
        # Knot-relative point, goal-relative point, camera-relative point, age,
        # predictive standard deviation and the backbone's three task maps.
        self.value = nn.Sequential(nn.Linear(64+3*3+1+3+3, 96), nn.SiLU(), nn.Linear(96, 64))
        self.query = nn.Sequential(nn.Linear(16+16+18+3+1, 128), nn.SiLU(), nn.LayerNorm(128))
        self.key = nn.Linear(64, 128)
        self.output = nn.Sequential(nn.Linear(128+64+2, 128), nn.SiLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def expected_proximity(offset, variance, radius):
        """E[exp(-||X-query||²/(2 r²))] for a diagonal Gaussian X."""
        total = radius**2 + variance
        return -.5 * (offset.square()/total + torch.log(total/radius**2)).sum(-1)

    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        with torch.no_grad():
            proposal, auxiliary = self.base(tokens, geometry, poses, ages, mask, state, tcp)
            raw = self.base.distribution(self.base.visual(tokens.float())).float()
            means = geometry[..., :3].float() + .25 * raw[..., :3].tanh()
            variance = (-5 + 4*raw[..., 3:].tanh()).exp() * self.variance_scale
        scale = tokens.new_tensor([.55, .55, .50]).float()
        center = tokens.new_tensor([.65, 0, .22]).float()
        points = means * scale + center
        variance = variance * scale.square()
        if self.variant == 'deterministic':
            variance = torch.zeros_like(variance)
        batch, history, cells, _ = tokens.shape
        valid = mask[..., None].expand(-1, -1, cells).reshape(batch, -1).clone()
        if self.variant == 'current':
            valid[:, :-cells] = False
        xyz = points.flatten(1, 2)
        var = variance.flatten(1, 2)
        targets = tcp[:, None, :3, 3] + proposal
        offset = xyz[:, None] - targets[:, :, None]
        goal_offset = (xyz - goal_xyz(state)[:, None])[:, None].expand(-1, 6, -1, -1)
        camera_offset = (points - poses[:, :, None, :3, 3]).flatten(1, 2)[:, None].expand(-1, 6, -1, -1)
        age = ages[..., None, None].expand(-1, -1, cells, -1).flatten(1, 2)[:, None].expand(-1, 6, -1, -1)/60
        visual = self.visual(tokens.float()).flatten(1, 2)[:, None].expand(-1, 6, -1, -1)
        maps = geometry[..., 3:].flatten(1, 2)[:, None].expand(-1, 6, -1, -1)
        sigma = var.clamp_min(0).sqrt()[:, None].expand(-1, 6, -1, -1)/.3
        spatial = (offset/.3, goal_offset/.3, camera_offset/.3)
        if self.variant == 'global':
            spatial = tuple(torch.zeros_like(x) for x in spatial)
        values = self.value(torch.cat((visual, *spatial, age, sigma, maps), -1))
        phase = torch.linspace(1/6, 1, 6, device=tokens.device)[None, :, None].expand(batch, -1, -1)
        context = torch.cat((state.float(), tcp.flatten(1).float(), proposal.flatten(1)/.3), -1)
        queries = self.query(torch.cat((context[:, None].expand(-1, 6, -1),
                                       (goal_xyz(state)[:, None]-targets)/.3, phase), -1))
        scores = (self.key(values) * queries[:, :, None]).sum(-1)/math.sqrt(128)
        proximity = self.expected_proximity(offset, var[:, None], self.radius)
        if self.variant == 'global':
            proximity = torch.zeros_like(proximity)
        scores = scores.float() + proximity
        # Normalize by the number of valid observations: repeat views must not
        # fabricate additional evidence solely by increasing the token count.
        scores = scores - valid.sum(-1).clamp_min(1).log()[:, None, None]
        scores = scores.masked_fill(~valid[:, None], -torch.inf)
        null = torch.full_like(scores[..., :1], -2.)
        attention = torch.cat((scores, null), -1).softmax(-1)
        retrieved = (attention[..., :-1, None] * values.float()).sum(-2)
        support = 1-attention[..., -1:]
        past_mask = torch.arange(history*cells, device=tokens.device) < (history-1)*cells
        past_mass = (attention[..., :-1]*past_mask).sum(-1, keepdim=True)
        if self.variant == 'residual':
            retrieved = torch.zeros_like(retrieved)
            support, past_mass = torch.zeros_like(support), torch.zeros_like(past_mass)
        correction = self.correction*self.output(torch.cat((queries, retrieved, support, past_mass), -1)).tanh()
        return proposal + correction.float(), {**auxiliary, 'support': support.detach(),
                'past_mass': past_mass.detach(), 'correction': correction.detach()}
