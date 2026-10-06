"""Deterministic hand clearance against four remembered RGB surface clouds.

The frozen learned route supplies the proposal. Fourteen bounded corrections
are scored using a fixed Gaussian proximity scale and a correction penalty.
The score is a geometric heuristic, not a collision probability.
"""
import math

import torch
from torch.nn import functional as F

from tsn.models.compact_policy import RouteController, goal_xyz, load_route_components


class ClearancePolicy(RouteController):
    """Refine a learned route using three samples along the hand axis.

    Streaming control accepts one episode. Each remembered cloud contains
    detached base-frame XYZ (1, 400, 3), metres, and validity (1, 400).
    """
    schedule = 'fixed15_clearance_deterministic'
    cloud_grid = (20, 20)
    hand_lengths = (0., .05, .10)
    correction_coordinates = (
        (0, 0), (-.015, 0), (.015, 0), (-.03, 0), (.03, 0),
        (-.05, 0), (.05, 0), (0, .02), (0, .04), (0, .06),
        (-.03, .03), (.03, .03), (-.05, .04), (.05, .04),
    )

    def __init__(self, backbone, head, kinematics, *, margin=.04, penalty=.08):
        """Use a positive proximity scale (metres) and nonnegative penalty."""
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError('Clearance margin must be finite and positive')
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError('Clearance penalty must be finite and nonnegative')
        super().__init__(backbone, head, kinematics)
        self.margin, self.penalty = margin, penalty

    def reset_episode(self):
        """Clear observation/surface memory and diagnostics before an episode."""
        super().reset_episode()
        self.clouds = []
        self.current_goal = None
        self.last_risk = 0.
        self.diagnostics = []
        self.current_points = None

    def perceive(self, rgb, state, K, pose):
        """Extract route features and remember a 20 x 20 metric surface cloud.

        Inputs are RGB (1, H, W, 3), normalized state (1, 16), intrinsics
        (1, 3, 3), and camera-to-base pose (1, 4, 4). Returns joint offsets
        (1, 30, 7), pooled tokens (1, 16, 768), and maps (1, 16, 6).
        Only predicted XYZ and a workspace mask enter the clearance memory.
        """
        if len(state) != 1:
            raise ValueError('Streaming clearance controller expects one episode')
        action, tokens, geometry, dense = self.backbone(
            rgb, state, K, pose, return_features=True, return_maps=True)
        if hasattr(self.head, 'stream'):
            from tsn.models.current_view_policy import metric_points
            self.current_points = metric_points(dense)
        points = F.interpolate(dense[:, :3].float(), self.cloud_grid,
                               mode='nearest-exact').flatten(2).transpose(1, 2)
        # Preserve the saved controller's decode, including autocast rounding
        # of constants to the dense map dtype before converting to float32.
        scale = dense.new_tensor([.55, .55, .5]).float()
        center = dense.new_tensor([.65, 0, .22]).float()
        points = points * scale + center
        valid = ((points[..., 0] > .10) & (points[..., 0] < 1.05) &
                 (points[..., 1].abs() < .60) & (points[..., 2] > .04) & (points[..., 2] < .65))
        self.clouds.append((points.detach(), valid.detach()))
        self.clouds = self.clouds[-self.history_length:]
        self.current_goal = goal_xyz(state)
        return action, tokens, geometry

    def route_candidates(self, waypoints, tcp):
        """Generate side/up corrections, ramped along the route and goal-faded.

        Returns candidates (1, 14, 30, 3), offsets (14, 3), and fade (1,).
        Candidate zero preserves the proposal exactly. Corrections vanish
        within .025 m of the goal and reach full strength at .08 m.
        """
        goal_delta = self.current_goal - tcp[:, :3, 3]
        direction = goal_delta.clone()
        direction[:, 2] = 0
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(.01)
        side = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])), -1)
        up = torch.zeros_like(side)
        up[:, 2] = 1
        coordinates = waypoints.new_tensor(self.correction_coordinates)
        offsets = coordinates[:, :1] * side + coordinates[:, 1:] * up
        ramp = torch.linspace(1/30, 1, 30, device=waypoints.device).sin() * 1.1883951
        fade = ((goal_delta.norm(dim=-1) - .025) / .055).clamp(0, 1)
        candidates = (waypoints[:, None] + offsets[None, :, None] *
                      ramp[None, None, :, None] * fade[:, None, None, None])
        return candidates, offsets, fade

    def hand_proximity(self, candidates, rotation, tcp):
        """Score swept-hand proximity at steps 3, 6, 9, 12, 15 of each route.

        Three axial samples (TCP, .05 m, .10 m behind it) query remembered
        surfaces. Suppress points within .07 m of the measured TCP as an
        approximate self mask. Returns risk (1, 14) and valid point count.
        """
        points = torch.cat([cloud[0] for cloud in self.clouds], 1)
        valid = torch.cat([cloud[1] for cloud in self.clouds], 1)
        valid = valid & ((points - tcp[:, None, :3, 3]).norm(dim=-1) > .07)
        hand = torch.stack([candidates - rotation[:, None, :, :, 2] * length
                            for length in self.hand_lengths], -2)
        sampled = hand[:, :, 2:15:3]
        # (batch, candidate, step, hand sample, remembered point).
        distances = (sampled[..., None, :] - points[:, None, None, None]).square().sum(-1)
        radius = points.new_tensor(self.margin)
        proximity = (-.5 * distances / radius.square()).exp()
        proximity = proximity.masked_fill(~valid[:, None, None, None], 0)
        risk = proximity.amax(-1).mean((-1, -2))
        return risk, int(valid.sum())

    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        """Select the lowest proximity-plus-correction cost before joint IK.

        Waypoints (1, 30, 3), rotations (1, 30, 3, 3), and TCP (1, 4, 4)
        use the base frame. The controller hook also passes the terminal-servo
        flag and camera pose; fading uses metric goal distance instead.
        Diagnostics retain the proposal's risk and the selected correction.
        """
        candidates, offsets, fade = self.route_candidates(waypoints, tcp)
        risk, points = self.hand_proximity(candidates, rotation, tcp)
        cost = risk + self.penalty * (offsets.norm(dim=-1) / .05).square()[None]
        chosen = cost.argmin(-1)
        self.last_risk = float(risk[0, 0])
        self.diagnostics.append(dict(
            step=self.step, risk=self.last_risk, choice=int(chosen[0]),
            correction_m=float(offsets[chosen[0]].norm() * fade[0]), points=points))
        return candidates[torch.arange(len(waypoints), device=waypoints.device), chosen]


def load_clearance_policy(path, device='cpu', *, margin=.04, penalty=.08,
                          allow_training_source=False):
    """Load deterministic hand clearance from the existing learned checkpoint.

    Returns the evaluation policy, depth-based training supervision maps, and
    saved metadata. Depth maps are never inputs to the deployed policy.
    Earlier memory-branch checkpoints may initialize training when requested.
    """
    backbone, head, kinematics, maps, checkpoint = load_route_components(
        path, device, allow_training_source=allow_training_source)
    policy = ClearancePolicy(backbone, head, kinematics, margin=margin, penalty=penalty)
    if hasattr(head, 'stream'):
        policy.schedule = 'fixed15_persistent_' + checkpoint['variant']
    elif checkpoint.get('route_history_length') == 1:
        policy.route_history_length = 1
        policy.schedule = 'fixed15_current_clearance_deterministic'
    return policy.to(device).eval(), maps, checkpoint
