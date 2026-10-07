"""Learned route preservation over explicit, remembered metric geometry.

Only a small positive trust-cost network is learned. Surface coordinates,
hand queries and the candidate lattice stay explicit and inspectable. The
energy is a proximity heuristic, never a collision probability.
"""
import torch
from torch import nn

from tsn.models.clearance_policy import ClearancePolicy
from tsn.models.compact_policy import goal_xyz, load_route_components
from tsn.common.checkpoint import load_checkpoint


class RouteTrust(nn.Module):
    """Predict a bounded positive correction cost from metric route context."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.network[-1].weight)
        # .02 + .10 * sigmoid(logit(.6)) = the established .08 initialization.
        nn.init.constant_(self.network[-1].bias, 0.4054651081081644)

    def forward(self, features):
        return .02 + .10 * self.network(features.float()).sigmoid().squeeze(-1)


def trust_features(waypoints, tcp, goal, risk):
    delta = goal-tcp[:, :3, 3]
    return torch.cat((delta/.3, waypoints[:, 14]-tcp[:, :3, 3],
                      risk[:, :1], risk.amin(-1, keepdim=True)), -1)


def surface_risk(candidates, rotation, tcp, points, valid, hand_extent=True, margin=.04):
    """Batched swept-hand queries against metric surfaces; empty evidence is zero."""
    valid = valid & torch.isfinite(points).all(-1)
    valid = valid & ((points-tcp[:, None, :3, 3]).norm(dim=-1) > .07)
    safe = torch.where(valid[..., None], points.float(), torch.zeros_like(points).float())
    lengths = (0., .05, .10) if hand_extent else (0.,)
    hand = torch.stack([candidates-rotation[:, None, :, :, 2]*length for length in lengths], -2)
    sampled = hand[:, :, 2:15:3]
    distances = (sampled[..., None, :]-safe[:, None, None, None]).square().sum(-1)
    proximity = (-.5*distances/points.new_tensor(margin).square()).exp()
    return proximity.masked_fill(~valid[:, None, None, None], 0).amax(-1).mean((-1, -2))


class GeometricEnergyPolicy(ClearancePolicy):
    """Four surface clouds → swept-hand energy → learned route trust → IK."""
    def __init__(self, backbone, head, kinematics, trust, ablation='full'):
        if ablation not in ('full', 'no_history', 'current_frame', 'tcp_only', 'no_trust', 'fixed_trust'):
            raise ValueError('Unknown geometric-energy ablation')
        super().__init__(backbone, head, kinematics)
        self.trust, self.ablation = trust, ablation
        # Proposal-feature history stays identical in every ablation.
        self.route_history_length = 1 if ablation == 'current_frame' else 4
        self.schedule = 'fixed15_geometric_energy_'+ablation

    def perceive(self, rgb, state, K, pose):
        result = super().perceive(rgb, state, K, pose)
        if self.ablation in ('no_history', 'current_frame'):
            self.clouds = self.clouds[-1:]
        return result

    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        candidates, offsets, fade = self.route_candidates(waypoints, tcp)
        points = torch.cat([p for p, _ in self.clouds], 1)
        valid = torch.cat([v for _, v in self.clouds], 1)
        risk = surface_risk(candidates, rotation, tcp, points, valid,
                            hand_extent=self.ablation != 'tcp_only')
        penalty = self.trust(trust_features(waypoints, tcp, self.current_goal, risk))
        if self.ablation == 'no_trust':
            penalty = torch.zeros_like(penalty)
        elif self.ablation == 'fixed_trust':
            penalty = torch.full_like(penalty, .08)
        cost = risk+penalty[:, None]*(offsets.norm(dim=-1)/.05).square()[None]
        chosen = cost.argmin(-1)
        self.last_risk = float(risk[0, 0])
        self.diagnostics.append(dict(step=self.step, risk=self.last_risk, choice=int(chosen[0]),
            correction_m=float(offsets[chosen[0]].norm()*fade[0]), points=int(valid.sum()),
            trust_penalty=float(penalty[0]), selected_risk=float(risk[0, chosen[0]])))
        return candidates[torch.arange(len(waypoints), device=waypoints.device), chosen]


def load_geometric_energy(path, device='cpu', ablation='full'):
    checkpoint = load_checkpoint(path)
    if checkpoint.get('architecture') != 'geometric_energy':
        raise ValueError('Expected a geometric-energy adapter checkpoint')
    backbone, head, kinematics, maps, parent = load_route_components(checkpoint['parent_checkpoint'], device)
    import hashlib
    from pathlib import Path
    if hashlib.sha256(Path(checkpoint['parent_checkpoint']).read_bytes()).hexdigest() != checkpoint['parent_sha256']:
        raise ValueError('Frozen proposal checkpoint changed')
    trust = RouteTrust().to(device)
    trust.load_state_dict(checkpoint['trust'], strict=True)
    policy = GeometricEnergyPolicy(backbone, head, kinematics, trust, ablation).to(device).eval()
    return policy, maps, dict(parent, geometric_energy=checkpoint['training'])
