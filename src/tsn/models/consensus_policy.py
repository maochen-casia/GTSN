"""Task-space consensus with causal visual memory and disagreement sensing."""
import torch
from torch import nn

from tsn.models.cartesian_policy import CartesianPolicy


class ConsensusHead(nn.Module):
    """Combine complementary route estimates in common metric coordinates.

    Disagreement is an action-space diagnostic, not calibrated collision risk.
    Geometry distributions remain trained by the original proper scoring rule.
    """
    def __init__(self, heads, weights=None):
        super().__init__()
        self.heads = nn.ModuleList(heads)
        weights = torch.ones(len(heads)) if weights is None else torch.tensor(weights)
        if len(heads) == 0 or len(weights) != len(heads) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Expected one nonnegative weight per head and positive total')
        self.register_buffer('weights', weights.float() / weights.sum())

    def forward(self, *args):
        predictions, auxiliaries = zip(*(head(*args) for head in self.heads))
        values = torch.stack(predictions)
        mean = (values * self.weights[:, None, None, None]).sum(0)
        variance = ((values - mean).square().sum(-1) * self.weights[:, None, None]).sum(0)
        # Disagreement along the actually executed first fifteen control steps.
        risk = variance[:, :3].mean(-1).sqrt()
        return mean, {**auxiliaries[-1], 'risk': risk}


class ConsensusPolicy(CartesianPolicy):
    def __init__(self, *args, disagreement_threshold=float('inf'), temporal_weight=0., **kwargs):
        self.disagreement_threshold = disagreement_threshold
        self.temporal_weight = temporal_weight
        super().__init__(*args, **kwargs)
        self.schedule = f'consensus_e{self.execute}_d{disagreement_threshold}_t{temporal_weight}'

    def reset_episode(self):
        super().reset_episode()
        self.previous = None

    def execution_horizon(self, default):
        return 5 if self.last_risk > self.disagreement_threshold else self.execute

    def forward(self, rgb, state, K, pose):
        chunk = super().forward(rgb, state, K, pose)
        anchor = state[:, :7].float() * torch.pi
        absolute = chunk + anchor[:, None]
        if self.previous is not None and self.temporal_weight > 0:
            old_step, old = self.previous
            elapsed = self.step - old_step
            overlap = min(30, 30 - elapsed)
            if elapsed > 0 and overlap > 0:
                absolute[:, :overlap] = (1-self.temporal_weight)*absolute[:, :overlap] + self.temporal_weight*old[:, elapsed:elapsed+overlap]
        self.previous = (self.step, absolute.detach().clone())
        return absolute-anchor[:, None]
