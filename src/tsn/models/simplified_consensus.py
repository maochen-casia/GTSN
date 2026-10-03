"""Compact route heads for the frozen epoch-15 Pi3 simplification study."""
import copy

import torch
from torch import nn

from tsn.models.cartesian_policy import CartesianPolicy, goal_xyz
from tsn.models.consensus_policy import ConsensusHead


VARIANTS = ('control', 'no_uncertainty', 'mean_memory', 'current_only',
            'single_memory', 'deterministic_current')


class CompactPolicy(CartesianPolicy):
    """Fixed-horizon controller without adaptive scheduling or temporal blending."""
    def forward(self, rgb, state, K, pose):
        chunk = super().forward(rgb, state, K, pose)
        # Preserve the reference controller's floating-point round trip exactly.
        anchor = state[:, :7].float()*torch.pi
        return (chunk+anchor[:, None])-anchor[:, None]


class CompactRouteHead(nn.Module):
    """Retain only modules used by the requested route computation.

    The current-only head needs neither attention keys nor historical frames.
    Deterministic variants remove the distribution and its reliability gate.
    """
    def __init__(self, original, uncertainty=True, temporal='attention'):
        super().__init__()
        if temporal not in ('attention', 'mean', 'current'):
            raise ValueError(temporal)
        self.uncertainty, self.temporal = uncertainty, temporal
        for name in ('visual', 'geometry', 'frame', 'query', 'output'):
            setattr(self, name, copy.deepcopy(getattr(original, name)))
        if uncertainty:
            self.distribution = copy.deepcopy(original.distribution)
        if temporal == 'attention':
            self.key = copy.deepcopy(original.key)

    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        if self.temporal == 'current':
            tokens, geometry, poses = tokens[:, -1:], geometry[:, -1:], poses[:, -1:]
            ages, mask = ages[:, -1:], mask[:, -1:]
        visual = self.visual(tokens.float())
        mean = geometry[..., :3].float()
        aux = {}
        if self.uncertainty:
            dist = self.distribution(visual).float()
            mean = mean + .25*dist[..., :3].tanh()
            logvar = -5+4*dist[..., 3:].tanh()
            visual = visual * ((-.5*logvar.detach().mean(-1)).softmax(-1)*16)[..., None]
            aux = {'mean': mean[:, -1], 'logvar': logvar[:, -1]}
        geom = self.geometry(torch.cat((mean, geometry[..., 3:].float()), -1))
        frames = self.frame(torch.cat((torch.cat((visual, geom), -1).flatten(2),
                                      poses.flatten(2).float(), ages[..., None].float()/60), -1))
        query = self.query(torch.cat((state.float(), tcp.flatten(1).float(),
                                     (goal_xyz(state)-tcp[:, :3, 3])/.3), -1))
        if self.temporal == 'current':
            context = frames[:, -1]
        elif self.temporal == 'mean':
            context = (frames*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
        else:
            scores = (self.key(frames)*query[:, None]).sum(-1)/16
            context = (scores.masked_fill(~mask, -torch.inf).softmax(-1)[..., None]*frames).sum(1)
        out = .3*self.output(torch.cat((query, context), -1)).reshape(-1, 6, 3).tanh()
        return out.float(), {**aux, 'risk': out.new_zeros(len(out))}


def compact_head(model, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    route = CompactRouteHead(model.heads['memory'],
        uncertainty=variant not in ('no_uncertainty', 'deterministic_current'),
        temporal=('mean' if variant == 'mean_memory' else
                  'current' if variant in ('current_only', 'deterministic_current') else 'attention'))
    heads = [route] if variant == 'single_memory' else [copy.deepcopy(model.heads['state']), route]
    return ConsensusHead(heads)


def load_compact_policy(path, device='cpu'):
    """Load the standalone exported model without constructing discarded heads."""
    from tsn.common.checkpoint import load_checkpoint
    from tsn.models.cartesian_policy import CartesianHead, PandaKinematics
    from tsn.models.factory import make_maps, make_policy
    from tsn.models.fresh_consensus import StateRouteHead

    checkpoint = load_checkpoint(path)
    if checkpoint.get('architecture') != 'compact_consensus':
        raise ValueError('Expected a compact_consensus checkpoint')
    variant = checkpoint['variant']
    if variant not in VARIANTS:
        raise ValueError(variant)
    config = checkpoint['config']['model']
    backbone = make_policy(config, initialize_backbone=False)
    backbone.load_state_dict(checkpoint['backbone'], strict=True)
    route = CompactRouteHead(CartesianHead('memory'),
        uncertainty=variant not in ('no_uncertainty', 'deterministic_current'),
        temporal=('mean' if variant == 'mean_memory' else
                  'current' if variant in ('current_only', 'deterministic_current') else 'attention'))
    head = ConsensusHead([route] if variant == 'single_memory' else [StateRouteHead(), route])
    head.load_state_dict(checkpoint['head'], strict=True)
    if variant == 'single_memory':
        head = head.heads[0]
    history_length = 1 if variant in ('current_only', 'deterministic_current') else 4
    policy = CompactPolicy(backbone, head, PandaKinematics(), servo_radius=.08, execute=15,
                           orientation='baseline', history_length=history_length).to(device).eval()
    return policy, make_maps(config).to(device), checkpoint
