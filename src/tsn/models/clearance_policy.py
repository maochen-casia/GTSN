"""Preserve a learned route while refining clearance against remembered surfaces.

The revised research configuration uses deterministic mode, a .04 m proximity
scale, and correction penalty .08. Uncertainty inflation remains available for
the original study and its controls; it is not part of the revised method.
"""
import torch
from torch.nn import functional as F

from tsn.models.compact_policy import CompactPolicy, goal_xyz


class ClearancePolicy(CompactPolicy):
    def __init__(self, backbone, head, kinematics, mode='full', margin=.04, uncertainty=.5, penalty=.15):
        if mode not in ('full', 'deterministic', 'current', 'tcp'):
            raise ValueError(mode)
        self.mode, self.margin, self.uncertainty, self.penalty = mode, margin, uncertainty, penalty
        super().__init__(backbone, head, kinematics)
        self.schedule = 'fixed15_clearance_' + mode

    def reset_episode(self):
        super().reset_episode()
        self.clouds = []
        self.last_risk = 0.
        self.diagnostics = []

    def perceive(self, rgb, state, K, pose):
        action, tokens, geometry, dense = self.backbone(rgb, state, K, pose,
                                                       return_features=True, return_maps=True)
        if len(state) != 1:
            raise ValueError('Streaming clearance controller expects one episode')
        raw = self.head.distribution(self.head.visual(tokens.float())).float()
        std = (-5+4*raw[..., 3:].tanh()).exp().sqrt()
        # Distributions describe pooled cells. They are used as a heuristic
        # localization margin, never a calibrated dense collision probability.
        scale = dense.new_tensor([.55, .55, .5]).float()
        center = dense.new_tensor([.65, 0, .22]).float()
        points = F.interpolate(dense[:, :3].float(), (20, 20), mode='nearest-exact').flatten(2).transpose(1, 2)
        points = points*scale + center
        sigma = F.interpolate(std.transpose(1, 2).reshape(1, 3, 4, 4), (20, 20), mode='nearest-exact')
        sigma = (sigma.flatten(2).transpose(1, 2)*scale).square().mean(-1).sqrt().clamp(max=.08)
        valid = ((points[..., 0] > .10) & (points[..., 0] < 1.05) &
                 (points[..., 1].abs() < .60) & (points[..., 2] > .04) & (points[..., 2] < .65))
        self.clouds.append((points.detach(), sigma.detach(), valid.detach()))
        self.clouds = self.clouds[-(1 if self.mode == 'current' else 4):]
        self.current_goal = goal_xyz(state)
        return action, tokens, geometry

    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        points, sigma, valid = [torch.cat([x[i] for x in self.clouds], 1) for i in range(3)]
        # Suppress potential robot/self points close to the measured TCP. This
        # is deliberately only an approximate mask; no simulator segmentation.
        valid = valid & ((points-tcp[:, None, :3, 3]).norm(dim=-1) > .07)
        goal_delta = self.current_goal-tcp[:, :3, 3]
        direction = goal_delta.clone()
        direction[:, 2] = 0
        direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(.01)
        side = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])), -1)
        up = torch.zeros_like(side)
        up[:, 2] = 1
        coordinates = waypoints.new_tensor([(0, 0), (-.015, 0), (.015, 0), (-.03, 0), (.03, 0),
                      (-.05, 0), (.05, 0), (0, .02), (0, .04), (0, .06),
                      (-.03, .03), (.03, .03), (-.05, .04), (.05, .04)])
        offsets = coordinates[:, :1]*side + coordinates[:, 1:]*up
        # Smoothly ramp the correction from the observed pose. Replanning still
        # occurs every fifteen steps, using the same thirty-step prediction.
        ramp = torch.linspace(1/30, 1, 30, device=waypoints.device).sin()*1.1883951
        fade = ((goal_delta.norm(dim=-1)-.025)/.055).clamp(0, 1)
        candidates = waypoints[:, None] + offsets[None, :, None]*ramp[None, None, :, None]*fade[:, None, None, None]
        # Approximate the hand by three samples on its axis behind the TCP.
        # This queries body clearance rather than end-effector position alone.
        lengths = [0.] if self.mode == 'tcp' else [0., .05, .10]
        body = torch.stack([candidates - rotation[:, None, :, :, 2]*length for length in lengths], -2)
        sampled = body[:, :, 2:15:3]
        distances = (sampled[..., None, :]-points[:, None, None, None]).square().sum(-1)
        inflation = 0. if self.mode == 'deterministic' else self.uncertainty*sigma
        radius = self.margin+inflation
        if not isinstance(radius, torch.Tensor):
            radius = torch.full_like(sigma, radius)
        proximity = (-.5*distances/radius[:, None, None, None].square()).exp()
        proximity = proximity.masked_fill(~valid[:, None, None, None], 0)
        # Worst observed surface point, then average over the swept hand.
        risk = proximity.amax(-1).mean((-1, -2))
        cost = risk + self.penalty*(offsets.norm(dim=-1)/.05).square()[None]
        chosen = cost.argmin(-1)
        self.last_risk = float(risk[0, 0])
        self.diagnostics.append(dict(step=self.step, risk=self.last_risk, choice=int(chosen[0]),
                                    correction_m=float(offsets[chosen[0]].norm()*fade[0]),
                                    points=int(valid.sum())))
        return candidates[torch.arange(len(waypoints), device=waypoints.device), chosen]
