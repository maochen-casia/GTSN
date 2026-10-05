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
    """Add a learned bounded residual to a frozen six-knot route proposal."""
    variants = ('full', 'deterministic', 'current', 'global', 'residual')

    def __init__(self, base=None, variant='full', radius=.12, correction=.06):
        """Create the retrieval head and freeze its underlying route proposer.

        Args:
            base (CompactRouteHead | None): Existing route head; None creates
                one. Its parameters are frozen, including when supplied here.
            variant (str): 'full' uses uncertain local retrieval; 'deterministic'
                zeros variance; 'current' retrieves only the last frame;
                'global' removes spatial offsets/proximity; 'residual' removes
                retrieved features and support signals from the residual head.
            radius (float): Positive Gaussian retrieval radius in metres.
            correction (float): Per-coordinate residual bound in metres.

        Returns:
            None. Initializes the residual output to zero so initial routes
            match the base exactly. variance_scale is a three-coordinate
            dimensionless buffer multiplying predicted normalized variance.

        Raises:
            ValueError: The variant is unknown.
        """
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
        """Set residual-head training mode while keeping the proposal in eval.

        Args:
            mode (bool): True enables training for the trainable head layers.

        Returns:
            EvidenceRouteHead: This instance, for normal nn.Module chaining.
            The base is always set to evaluation mode by this call.
        """
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def expected_proximity(offset, variance, radius):
        """Return log E[exp(-||X-query||²/(2 r²))] for a diagonal Gaussian X.

        Args:
            offset (torch.Tensor): Floating mean-minus-query offsets (..., 3), m.
            variance (torch.Tensor): Nonnegative floating diagonal variances
                (..., 3), m²; must broadcast with offset.
            radius (float): Positive Gaussian kernel radius r, metres.

        Returns:
            torch.Tensor: Floating log expected proximity (...,), reducing the
            final XYZ dimension after broadcasting. Exponentiating gives a
            kernel value in (0, 1], subject to floating-point underflow.
        """
        total = radius**2 + variance
        return -.5 * (offset.square()/total + torch.log(total/radius**2)).sum(-1)

    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        """Retrieve scene evidence near each proposal knot and predict a residual.

        Args:
            tokens (torch.Tensor): Floating Pi3 features (B, T, 16, 768),
                ordered oldest to newest over T observed/padded frames.
            geometry (torch.Tensor): Floating maps (B, T, 16, 6): normalized
                base XYZ, projected goal, visible goal, future-action score.
            poses (torch.Tensor): Floating camera-to-base poses (B, T, 4, 4),
                with translations in metres.
            ages (torch.Tensor): Numeric observation ages (B, T), control steps.
            mask (torch.Tensor): Boolean frame validity (B, T), with at least
                one valid frame per sample for the base head's attention.
            state (torch.Tensor): Floating normalized policy state (B, 16),
                following CompactRouteHead's state convention.
            tcp (torch.Tensor): Floating current TCP-to-base poses (B, 4, 4).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Float32 corrected
            offsets (B, 6, 3), base-frame metres from the current TCP at steps
            5, 10, ..., 30. Auxiliary 'mean'/'logvar' are (B, 16, 3) normalized
            point means/log variances from the last frame of the frozen base;
            'support' and 'past_mass' are detached floating (B, 6, 1) attention
            masses on all real evidence and on past frames, respectively;
            'correction' is detached floating (B, 6, 3), metres. In 'residual'
            mode both returned attention masses are zeroed ablation signals.

        Notes:
            The base proposal and point distributions run without gradients.
            A null token absorbs unsupported attention; residual magnitude is
            bounded independently per coordinate by self.correction.
        """
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
        # Flatten time/cell into E=T*16 evidence entries. Each of six knots
        # queries all entries: offsets/values carry axes (B, 6, E, channels).
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
