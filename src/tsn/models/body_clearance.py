"""Query candidate IK realizations with arm and hand geometry in base coordinates.

All body points come from the robot's known kinematics. Scene evidence comes
only from RGB predictions. Two redundant IK seeds can change elbow posture
without changing the requested Cartesian route.
"""
import torch

from tsn.models.clearance_policy import ClearancePolicy


class BodyClearancePolicy(ClearancePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule = 'fixed15_body_clearance_' + self.mode

    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        self.proposed_waypoints = waypoints
        self.proposed_rotation = rotation
        self.current_tcp = tcp
        return waypoints

    def body_points(self, q):
        kin = self.kinematics
        transform = torch.eye(4, device=q.device).expand(*q.shape[:-1], 4, 4).clone()
        joints = []
        for origin, index in zip(kin.origins, kin.indices):
            transform = transform @ origin
            if index >= 0:
                joints.append(transform[..., :3, 3])
                c, s = q[..., index].cos(), q[..., index].sin()
                rotation = torch.eye(4, device=q.device).expand_as(transform).clone()
                rotation[..., 0, 0], rotation[..., 1, 1] = c, c
                rotation[..., 0, 1], rotation[..., 1, 0] = -s, s
                transform = transform @ rotation
        points = joints[3:]
        points += [(joints[i]+joints[i+1])/2 for i in (3, 4, 5)]
        points += [transform[..., :3, 3]-length*transform[..., :3, 2] for length in (0., .05, .10)]
        return torch.stack(points, -2)

    def forward(self, rgb, state, K, pose):
        original = super().forward(rgb, state, K, pose)
        with torch.autocast(device_type=state.device.type, enabled=False):
            return self.choose_realization(original, state)

    def choose_realization(self, original, state):
        q = state[:, :7].float()*torch.pi
        baseline = original.float()+q[:, None]
        waypoints, rotation, tcp = self.proposed_waypoints, self.proposed_rotation, self.current_tcp
        delta = self.current_goal-tcp[:, :3, 3]
        direction = delta.clone()
        direction[:, 2] = 0
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(.01)
        side = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])), -1)
        up = torch.zeros_like(side)
        up[:, 2] = 1
        coordinates = q.new_tensor([(0, 0), (-.015, 0), (.015, 0), (-.03, 0), (.03, 0),
                     (-.05, 0), (.05, 0), (0, .02), (0, .04), (0, .06),
                     (-.03, .03), (.03, .03), (-.05, .04), (.05, .04), (0, 0), (0, 0)])
        offsets = coordinates[:, :1]*side+coordinates[:, 1:]*up
        ramp = torch.linspace(1/30, 1, 30, device=q.device).sin()*1.1883951
        fade = ((delta.norm(dim=-1)-.025)/.055).clamp(0, 1)
        targets = waypoints[:, None]+offsets[None, :, None]*ramp[None, None, :, None]*fade[:, None, None, None]
        initial = baseline[:, None].expand(-1, 16, -1, -1).clone()
        posture = q.new_tensor([1., 0., -1., 0., 1., 0., 0.])
        initial[:, 14] += .2*ramp[None, :, None]*posture*fade[:, None, None]
        initial[:, 15] -= .2*ramp[None, :, None]*posture*fade[:, None, None]
        rotations = rotation[:, None].expand(-1, 16, -1, -1, -1)
        candidates = self.kinematics.inverse(initial.reshape(-1, 7), targets.reshape(-1, 3),
                                              rotations.reshape(-1, 3, 3)).reshape(1, 16, 30, 7)
        candidates[:, 0] = baseline
        sample = candidates[:, :, 2:15:3]
        body = self.body_points(sample)
        points, sigma, valid = [torch.cat([x[i] for x in self.clouds], 1) for i in range(3)]
        current_body = self.body_points(q)
        distance_self = (points[:, :, None]-current_body[:, None]).norm(dim=-1).amin(-1)
        valid = valid & (distance_self > .035)
        if not valid.any():
            self.last_risk = 0.
            self.diagnostics.append(dict(step=self.step, risk=0., choice=0, correction_m=0.,
                        joint_correction_rad=0., posture_only=False, points=0))
            return original
        distances = (body[..., None, :]-points[:, None, None, None]).square().sum(-1)
        inflation = torch.zeros_like(sigma) if self.mode == 'deterministic' else self.uncertainty*sigma
        # Known robot-body proxy sizes, not environment-specific scene geometry.
        radii = q.new_tensor([.055]*7+[self.margin]*3)
        effective = radii[None, None, None, :, None]+inflation[:, None, None, None]
        proximity = (-.5*distances/effective.square()).exp().masked_fill(~valid[:, None, None, None], 0)
        risk = proximity.amax(-1).amax(-1).mean(-1)
        intervention = (offsets.norm(dim=-1)/.05).square()
        intervention[14:] = .25
        reached = self.kinematics(candidates)[..., :3, 3]
        residual = (reached-targets).norm(dim=-1).mean(-1)
        cost = risk+self.penalty*intervention[None]+(residual/.02).square()
        chosen = int(cost.argmin(-1)[0])
        self.last_risk = float(risk[0, 0])
        self.diagnostics.append(dict(step=self.step, risk=self.last_risk, choice=chosen,
                 correction_m=float(offsets[chosen].norm()*fade[0]),
                 joint_correction_rad=float((candidates[0, chosen]-baseline[0]).norm(dim=-1).mean()),
                 posture_only=chosen >= 14, points=int(valid.sum())))
        if chosen == 0:
            return original
        return candidates[:, chosen]-q[:, None]
