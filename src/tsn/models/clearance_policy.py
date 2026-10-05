"""Preserve a learned route while refining clearance against remembered surfaces.

The revised research configuration uses deterministic mode, a .04 m proximity
scale, and correction penalty .08. Uncertainty inflation remains available for
the original study and its controls; it is not part of the revised method.
"""
import torch
from torch.nn import functional as F

from tsn.models.compact_policy import CompactPolicy, goal_xyz


class ClearancePolicy(CompactPolicy):
    """Score bounded route corrections against up to four predicted point clouds.

    This streaming extension requires batch size one. Each cloud stores base
    XYZ (1, 400, 3) in metres, scalar uncertainty (1, 400) in metres, and a
    boolean workspace-valid mask (1, 400). Geometry is detached from autograd.
    """
    def __init__(self, backbone, head, kinematics, mode='full', margin=.04, uncertainty=.5, penalty=.15):
        """Configure surface memory and the cost of correcting a learned route.

        Args:
            backbone (Pi3MapPolicy): Perception supporting features/dense maps.
            head (CompactRouteHead): Route head exposing visual/distribution
                layers for estimating pooled point uncertainty.
            kinematics (PandaKinematics): Robot forward/inverse kinematics.
            mode (str): 'full' uses memory and uncertainty; 'deterministic'
                removes uncertainty inflation; 'current' keeps one cloud;
                'tcp' queries only the TCP instead of three hand samples.
            margin (float): Positive base Gaussian proximity scale in metres.
            uncertainty (float): Multiplier of predicted metric uncertainty.
            penalty (float): Weight on squared correction length / .05 m.

        Returns:
            None. Initializes inherited observation memory and cloud diagnostics.

        Raises:
            ValueError: The requested mode is unknown.
        """
        if mode not in ('full', 'deterministic', 'current', 'tcp'):
            raise ValueError(mode)
        self.mode, self.margin, self.uncertainty, self.penalty = mode, margin, uncertainty, penalty
        super().__init__(backbone, head, kinematics)
        self.schedule = 'fixed15_clearance_' + mode

    def reset_episode(self):
        """Reset observation/cloud memory and clearance diagnostics.

        Args:
            None.

        Returns:
            None. Clears clouds and diagnostics, sets last_risk and step to zero.
        """
        super().reset_episode()
        self.clouds = []
        self.last_risk = 0.
        self.diagnostics = []

    def perceive(self, rgb, state, K, pose):
        """Predict features and append a metric surface cloud for refinement.

        Args:
            rgb (torch.Tensor): Uint8 RGB (1, H, W, 3), values 0..255.
            state (torch.Tensor): Floating normalized policy state (1, 16);
                layout follows CompactPolicy's module docstring.
            K (torch.Tensor): Floating intrinsics (1, 3, 3), sensor pixels.
            pose (torch.Tensor): Floating camera-to-base transform (1, 4, 4).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Floating action
            offsets (1, 30, 7), radians; pooled tokens (1, 16, 768); and pooled
            geometry (1, 16, 6), in normalized XYZ / projected goal / visible
            goal / future-action order. Dtypes follow backbone autocast.

        Notes:
            Stores detached float32 points (1, 400, 3), uncertainty (1, 400),
            and boolean validity (1, 400) from a 20 x 20 grid. Uncertainty is
            the RMS metric coordinate standard deviation, capped at .08 m.
            Also updates current_goal (1, 3), metres. Keeps one cloud in
            'current' mode and at most four otherwise.

        Raises:
            ValueError: The input contains more or fewer than one episode.
        """
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
        """Choose one of 14 bounded route corrections using remembered surfaces.

        Args:
            waypoints (torch.Tensor): Floating base XYZ targets (1, 30, 3), m.
            rotation (torch.Tensor): Floating target TCP-to-base rotations
                (1, 30, 3, 3), used to place hand samples behind each TCP.
            tcp (torch.Tensor): Floating current TCP-to-base pose (1, 4, 4).
            near (torch.Tensor): Boolean servo flag (1,), accepted for the
                parent hook; fading here uses metric goal distance instead.
            pose (torch.Tensor): Floating camera-to-base pose (1, 4, 4),
                accepted for the parent hook and unused here.

        Returns:
            torch.Tensor: Selected floating base-frame waypoints (1, 30, 3),
            metres, on the input device. The zero-offset candidate preserves
            the proposal exactly. Corrections fade to zero within .025 m of
            the goal and reach full strength at .08 m.

        Notes:
            Requires clouds/current_goal from perceive. Evaluates steps
            3, 6, 9, 12, 15 of each candidate with one or three hand samples.
            Updates last_risk with the ORIGINAL candidate's proximity score
            and appends the selected correction to diagnostics. The score is
            a Gaussian proximity heuristic, not a collision probability.
        """
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
        # Candidate axis C=14; body has shape (1, C, 30, S, 3), S=1 or 3.
        # Approximate the hand by three samples on its axis behind the TCP.
        # This queries body clearance rather than end-effector position alone.
        lengths = [0.] if self.mode == 'tcp' else [0., .05, .10]
        body = torch.stack([candidates - rotation[:, None, :, :, 2]*length for length in lengths], -2)
        sampled = body[:, :, 2:15:3]
        distances = (sampled[..., None, :]-points[:, None, None, None]).square().sum(-1)
        # Distances (1, C, 5, S, P) compare sampled hand positions with all
        # P remembered surface points without materializing repeated clouds.
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
