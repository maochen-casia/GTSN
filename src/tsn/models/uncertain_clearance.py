"""Uncertainty-dependent obstacle inflation around RGB-predicted surfaces.

The uncertainty head predicts a quantile of metric reconstruction error. It is
not a collision predictor. The C1 map, proposal, candidate lattice, hand probes,
trust, IK and execution cadence remain unchanged.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from tsn.common.checkpoint import load_checkpoint
from tsn.models.adaptive_geometry import AdaptiveGeometryPolicy
from tsn.models.geometric_energy import load_geometric_energy, trust_features


class PointUncertainty(nn.Module):
    """Predict a bounded 90th percentile of position error, in metres."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(74, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, math.log(.035/.16))

    def forward(self, features):
        return .005+.195*self.network(features.float()).sigmoid().squeeze(-1)


def uncertainty_features(visual, points, pose, tcp):
    """Align 4×4 visual cells with the existing 20×20 predicted point grid.

    Inputs: visual (B,16,64), metric points (B,400,3), camera-to-base pose
    (B,4,4), measured TCP (B,4,4). No depth target or future route enters.
    """
    if points.shape[1:] != (400, 3) or visual.shape[1:] != (16, 64):
        raise ValueError('Uncertainty features require the established point/token grids')
    p = torch.nan_to_num(points.float(), nan=0., posinf=0., neginf=0.)
    grid = p.reshape(-1, 20, 20, 3)
    row = (grid[:, 1:]-grid[:, :-1]).norm(dim=-1)
    col = (grid[:, :, 1:]-grid[:, :, :-1]).norm(dim=-1)
    row = torch.cat((row, row[:, -1:]), 1)
    col = torch.cat((col, col[:, :, -1:]), 2)
    curvature = (grid-torch.roll(grid, 1, 1)*.5-torch.roll(grid, -1, 1)*.5).norm(dim=-1)
    roughness = torch.stack((row, col, curvature), -1).reshape(-1, 400, 3)/.1
    appearance = visual.float().reshape(-1, 4, 4, 64).repeat_interleave(5, 1).repeat_interleave(5, 2).reshape(-1, 400, 64)
    normalized = (p-p.new_tensor([.65, 0, .22]))/p.new_tensor([.55, .55, .5])
    # Row-vector form of inverse camera rotation, for point-to-camera context.
    camera = torch.einsum('bni,bij->bnj', p-pose[:, None, :3, 3].float(), pose[:, :3, :3].float())/.5
    distance = (p-tcp[:, None, :3, 3].float()).norm(dim=-1, keepdim=True)/.5
    features = torch.cat((appearance, normalized, roughness.clamp(0, 5), camera.clamp(-5, 5), distance.clamp(0, 5)), -1)
    return torch.nan_to_num(features, nan=0., posinf=5., neginf=-5.)


def padding_factor(radius):
    """No extra padding below 15 mm; full padding for errors of 100 mm or more."""
    return ((radius-.015)/.085).clamp(0, 1)


def inflated_surface_risk(candidates, rotation, tcp, points, padding, margin=.04):
    """Point-specific obstacle inflation, then the established swept-hand field."""
    if not len(points):
        return candidates.new_zeros(candidates.shape[:2])
    valid = torch.isfinite(points).all(-1) & torch.isfinite(padding)
    valid &= (points-tcp[0, :3, 3]).norm(dim=-1) > .07
    safe = torch.where(valid[:, None], points.float(), torch.zeros_like(points).float())
    lengths = (0., .05, .10)
    hand = torch.stack([candidates-rotation[:, None, :, :, 2]*length for length in lengths], -2)[:, :, 2:15:3]
    distance = (hand[..., None, :]-safe).square().sum(-1).clamp_min(0).sqrt()
    effective = (distance-padding.clamp_min(0)).clamp_min(0)
    proximity = (-.5*effective.square()/points.new_tensor(margin).square()).exp()
    return proximity.masked_fill(~valid, 0).amax(-1).mean((-1, -2))


class UncertainClearancePolicy(AdaptiveGeometryPolicy):
    REFINEMENT_MODES = ('adaptive', 'uniform', 'fixed', 'none')

    def __init__(self, backbone, head, kinematics, trust, uncertainty,
                 refinement='adaptive', max_padding=.02, uniform_factor=.5):
        if refinement not in self.REFINEMENT_MODES:
            raise ValueError('Unknown clearance refinement mode')
        if not math.isfinite(max_padding) or not 0 <= max_padding <= .05:
            raise ValueError('Maximum uncertainty padding must lie in [0, .05] metres')
        if not math.isfinite(uniform_factor) or not 0 <= uniform_factor <= 1:
            raise ValueError('Uniform padding factor must lie in [0, 1]')
        super().__init__(backbone, head, kinematics, trust, mode='unconfirmed')
        self.uncertainty = uncertainty
        self.refinement, self.max_padding, self.uniform_factor = refinement, max_padding, uniform_factor
        self.schedule = f'fixed15_uncertain_clearance_{refinement}_{max_padding:.3f}'

    def reset_episode(self):
        super().reset_episode()
        self.uncertainty_clouds = []
        self.map_uncertainty = None

    def perceive(self, rgb, state, K, pose):
        previous_points = self.geometry_memory.points
        previous_uncertainty = self.map_uncertainty
        result = super().perceive(rgb, state, K, pose)
        _, tokens, _ = result
        with torch.autocast(device_type=state.device.type, enabled=False):
            tcp = self.kinematics(state[:, :7].float()*math.pi)
            points, _ = self.clouds[-1]
            visual = self.head.visual(tokens.float())
            radius = self.uncertainty(uncertainty_features(visual, points, pose, tcp)).detach()
            self.uncertainty_clouds.append(radius)
            self.uncertainty_clouds = self.uncertainty_clouds[-4:]
            anchors = self.geometry_memory.points
            if len(anchors):
                nearest = torch.cdist(anchors, points[0].float()).argmin(-1)
                value = radius[0, nearest].clone()
                if previous_points is not None and len(previous_points):
                    # Direct subtraction preserves exact anchor identity;
                    # matrix-multiply cdist can cancel at submillimetre scales.
                    distance, old = (anchors[:, None]-previous_points[None]).square().sum(-1).min(-1)
                    kept = distance < 1e-12
                    value[kept] = previous_uncertainty[old[kept]]
                # Keep the creation-time uncertainty of actual anchored points.
                self.map_uncertainty = value
            else:
                self.map_uncertainty = points.new_zeros(0)
        return result

    def refinement_config(self):
        return dict(mode=self.refinement, max_padding_m=self.max_padding,
                    uniform_factor=self.uniform_factor, error_quantile=.9,
                    padding_factor='clip((predicted_error_quantile_m-.015)/.085, 0, 1)',
                    field='Gaussian of max(surface distance - point-specific padding, 0)',
                    map_uncertainty='Creation-time error quantile plus observed positional disagreement')

    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        if self.refinement == 'none':
            self.last_risk = 0.
            self.diagnostics.append(dict(step=self.step, risk=0., choice=0, correction_m=0.,
                points=int(sum(v.sum() for _, v in self.clouds)), refinement='none',
                mean_padding_m=0., max_padding_m=0., route_feature_frames=len(self.history)))
            return waypoints
        if self.refinement == 'fixed' or self.max_padding == 0:
            result = super().refine_waypoints(waypoints, rotation, tcp, near, pose)
            self.diagnostics[-1].update(refinement='fixed', mean_padding_m=0., max_padding_m=0.)
            return result
        candidates, offsets, fade = self.route_candidates(waypoints, tcp)
        points = torch.cat([p for p, _ in self.clouds], 1)[0]
        valid = torch.cat([v for _, v in self.clouds], 1)[0] & torch.isfinite(points).all(-1)
        radius = torch.cat(self.uncertainty_clouds, 1)[0]
        memory = self.geometry_memory
        old = memory.last < max(0, self.step-45)
        points = torch.cat((points[valid], memory.points[old]))
        radius = torch.cat((radius[valid], self.map_uncertainty[old]+memory.scatter[old].sqrt()))
        if self.refinement == 'uniform':
            factor = torch.full_like(radius, self.uniform_factor)
        else:
            factor = padding_factor(radius)
        padding = self.max_padding*factor
        risk = inflated_surface_risk(candidates, rotation, tcp, points, padding)
        penalty = self.trust(trust_features(waypoints, tcp, self.current_goal, risk))
        cost = risk+penalty[:, None]*(offsets.norm(dim=-1)/.05).square()[None]
        chosen = cost.argmin(-1)
        self.last_risk = float(risk[0, 0])
        self.diagnostics.append(dict(step=self.step, risk=self.last_risk, choice=int(chosen[0]),
            correction_m=float(offsets[chosen[0]].norm()*fade[0]), points=len(points),
            trust_penalty=float(penalty[0]), selected_risk=float(risk[0, chosen[0]]),
            refinement=self.refinement, mean_padding_m=float(padding.mean()) if len(padding) else 0.,
            max_padding_m=float(padding.max()) if len(padding) else 0.,
            mean_predicted_error_m=float(radius.mean()) if len(radius) else 0.,
            old_only_points=int(old.sum()), route_feature_frames=len(self.history)))
        return candidates[torch.arange(len(waypoints), device=waypoints.device), chosen]


def load_uncertain_clearance(path, device='cpu', refinement='adaptive', max_padding=.02):
    saved = load_checkpoint(path)
    if saved.get('architecture') != 'uncertainty_clearance':
        raise ValueError('Expected an uncertainty-clearance checkpoint')
    energy_path = Path(saved['energy_checkpoint'])
    if hashlib.sha256(energy_path.read_bytes()).hexdigest() != saved['energy_sha256']:
        raise ValueError('Frozen geometric-energy checkpoint changed')
    original, maps, checkpoint = load_geometric_energy(energy_path, device)
    uncertainty = PointUncertainty().to(device)
    uncertainty.load_state_dict(saved['uncertainty'], strict=True)
    policy = UncertainClearancePolicy(original.backbone, original.head, original.kinematics,
        original.trust, uncertainty, refinement, max_padding, saved['uniform_factor']).to(device).eval()
    return policy, maps, dict(checkpoint, uncertainty_training=saved['training'])
