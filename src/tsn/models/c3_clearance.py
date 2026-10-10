"""C3: predict point error and choose uncertainty-padded body-clearance corrections."""
import math

import torch
from torch import nn


def uncertainty_features(visual, points, pose, tcp, grid_hw=(20, 20)):
    """Match 4x4 visual cells to a configurable regular scene-point grid."""
    height, width = grid_hw
    count = height*width
    if points.shape[1:] != (count, 3) or visual.shape[1:] != (16, 64) or height % 4 or width % 4:
        raise ValueError('Point grid must match its dimensions and the 4x4 visual cells')
    p = torch.nan_to_num(points.float(), nan=0., posinf=0., neginf=0.)
    grid = p.reshape(-1, height, width, 3)
    row = (grid[:, 1:]-grid[:, :-1]).norm(dim=-1)
    col = (grid[:, :, 1:]-grid[:, :, :-1]).norm(dim=-1)
    row, col = torch.cat((row, row[:, -1:]), 1), torch.cat((col, col[:, :, -1:]), 2)
    curvature = (grid-.5*torch.roll(grid, 1, 1)-.5*torch.roll(grid, -1, 1)).norm(dim=-1)
    roughness = torch.stack((row, col, curvature), -1).reshape(-1, count, 3)/.1
    appearance = visual.float().reshape(-1, 4, 4, 64).repeat_interleave(height//4, 1).repeat_interleave(width//4, 2).reshape(-1, count, 64)
    normalized = (p-p.new_tensor([.65, 0, .22]))/p.new_tensor([.55, .55, .5])
    camera = torch.einsum('bni,bij->bnj', p-pose[:, None, :3, 3], pose[:, :3, :3])/.5
    distance = (p-tcp[:, None, :3, 3]).norm(dim=-1, keepdim=True)/.5
    return torch.cat((appearance, normalized, roughness.clamp(0, 5), camera.clamp(-5, 5), distance.clamp(0, 5)), -1)


def route_candidates(waypoints, tcp, goal):
    """Fourteen side/up offsets, smoothly applied along 30 proposed positions."""
    direction = goal-tcp[:, :3, 3]
    horizontal = direction.clone(); horizontal[:, 2] = 0
    horizontal /= horizontal.norm(dim=-1, keepdim=True).clamp_min(.01)
    side = torch.stack((-horizontal[:, 1], horizontal[:, 0], torch.zeros_like(horizontal[:, 0])), -1)
    up = torch.zeros_like(side); up[:, 2] = 1
    coordinates = waypoints.new_tensor([(0, 0), (-.015, 0), (.015, 0), (-.03, 0), (.03, 0),
        (-.05, 0), (.05, 0), (0, .02), (0, .04), (0, .06), (-.03, .03), (.03, .03), (-.05, .04), (.05, .04)])
    offsets = coordinates[None, :, :1]*side[:, None]+coordinates[None, :, 1:]*up[:, None]
    ramp = torch.linspace(1/30, 1, 30, device=waypoints.device).sin()*1.1883951
    fade = ((direction.norm(dim=-1)-.025)/.055).clamp(0, 1)
    candidates = waypoints[:, None]+offsets[:, :, None]*ramp[None, None, :, None]*fade[:, None, None, None]
    return candidates, offsets


class UncertaintyClearance(nn.Module):
    """Bounded metric-error head and positive route-preservation coefficient.

    Predictions are uncertainty signals, not certified error bounds. Geometry
    scoring is delegated to C2; episode memory belongs to C1.
    """
    def __init__(self, max_padding=.03, margin=.04, point_grid_hw=(20, 20)):
        super().__init__()
        if not 0 <= max_padding <= .05 or not math.isfinite(margin) or margin <= 0:
            raise ValueError('Invalid clearance distances')
        self.max_padding, self.margin = max_padding, margin
        self.point_grid_hw = tuple(point_grid_hw)
        self.error_head = nn.Sequential(nn.Linear(74, 64), nn.SiLU(), nn.Linear(64, 1))
        self.trust_head = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.error_head[-1].weight)
        nn.init.constant_(self.error_head[-1].bias, math.log(.035/.16))
        nn.init.zeros_(self.trust_head[-1].weight)
        nn.init.constant_(self.trust_head[-1].bias, math.log(1.5))

    def predict_error(self, visual, points, pose, tcp):
        return .005+.195*self.error_head(uncertainty_features(visual, points, pose, tcp, self.point_grid_hw)).sigmoid().squeeze(-1)

    def padding(self, radius):
        return self.max_padding*((radius-.015)/.085).clamp(0, 1)

    def costs(self, waypoints, rotation, tcp, goal, points, radius, fingers, embodiment, camera_extrinsic=None,
              point_weights=None, context=None):
        """Single-scene candidates (14,30,3) and costs (14,), shared by train/eval."""
        candidates, offsets = route_candidates(waypoints[None], tcp[None], goal[None])
        risk = embodiment.contact_risk(candidates[0], rotation, tcp, points,
                                      self.padding(radius), fingers, self.margin, camera_extrinsic, point_weights, context)
        features = torch.cat(((goal-tcp[:3, 3])/.3, waypoints[14]-tcp[:3, 3], risk[:1], risk.amin()[None]))
        trust = .02+.10*self.trust_head(features).sigmoid().squeeze(-1)
        return candidates[0], risk+trust*(offsets[0].norm(dim=-1)/.05).square()

    def refine(self, waypoints, rotation, tcp, goal, points, radius, fingers, embodiment, camera_extrinsic=None,
               point_weights=None, context=None):
        candidates, costs = self.costs(waypoints, rotation, tcp, goal, points, radius, fingers, embodiment,
                                       camera_extrinsic, point_weights, context)
        return candidates[costs.argmin()]
