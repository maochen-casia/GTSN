"""Task-relative Cartesian route prediction with calibrated robot kinematics."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PandaKinematics(nn.Module):
    """URDF-derived FK/Jacobian; no scene or simulator state is used."""
    def __init__(self):
        super().__init__()
        from mani_skill import PACKAGE_ASSET_DIR
        from scipy.spatial.transform import Rotation
        root = ET.parse(Path(PACKAGE_ASSET_DIR) / 'robots/panda/panda_v3.urdf').getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, link = [], 'panda_hand_tcp'
        while link in by_child:
            joint = by_child[link]
            chain.append(joint)
            link = joint.find('parent').get('link')
        chain.reverse()
        origins, axes, limits, indices = [], [], [], []
        for joint in chain:
            origin = np.eye(4, dtype=np.float32)
            tag = joint.find('origin')
            if tag is not None:
                origin[:3, 3] = np.fromstring(tag.get('xyz', '0 0 0'), sep=' ')
                origin[:3, :3] = Rotation.from_euler('xyz', np.fromstring(tag.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            origins.append(origin)
            if joint.get('type') == 'revolute':
                indices.append(len(axes))
                axes.append(np.fromstring(joint.find('axis').get('xyz'), sep=' '))
                lim = joint.find('limit')
                limits.append([float(lim.get('lower')), float(lim.get('upper'))])
            else:
                indices.append(-1)
        assert len(axes) == 7
        self.indices = indices
        self.register_buffer('origins', torch.tensor(np.stack(origins)))
        self.register_buffer('axes', torch.tensor(np.stack(axes), dtype=torch.float32))
        self.register_buffer('limits', torch.tensor(limits))

    def forward(self, q, jacobian=False):
        q = q.float()
        t = torch.eye(4, device=q.device).expand(*q.shape[:-1], 4, 4).clone()
        positions, directions = [], []
        for origin, index in zip(self.origins, self.indices):
            t = t @ origin
            if index >= 0:
                axis = self.axes[index]
                positions.append(t[..., :3, 3])
                directions.append((t[..., :3, :3] @ axis[..., None])[..., 0])
                # All Panda actuated axes in the URDF are local z.
                if not torch.equal(axis, axis.new_tensor([0., 0., 1.])):
                    raise ValueError('Unsupported non-z joint axis')
                c, s = q[..., index].cos(), q[..., index].sin()
                r = torch.eye(4, device=q.device).expand_as(t).clone()
                r[..., 0, 0], r[..., 1, 1] = c, c
                r[..., 0, 1], r[..., 1, 0] = -s, s
                t = t @ r
        if not jacobian:
            return t
        axis = torch.stack(directions, -1)
        difference = t[..., :3, 3, None] - torch.stack(positions, -1)
        linear = torch.linalg.cross(axis, difference, dim=-2)
        return t, torch.cat((linear, axis), -2)

    def inverse(self, q, xyz, rotation, iterations=12):
        """Batched bounded DLS tracking, initialized at measured joints."""
        q = q.float().clone()
        xyz, rotation = xyz.float(), rotation.float()
        for _ in range(iterations):
            pose, jac = self(q, True)
            dp = xyz-pose[..., :3, 3]
            dr = .5 * torch.linalg.cross(pose[..., :3, :3], rotation, dim=-2).sum(-1)
            error = torch.cat((dp, dr), -1)
            weights = error.new_tensor([1, 1, 1, .3, .3, .3])
            j = jac * weights[..., None]
            error = error * weights
            system = j @ j.transpose(-1, -2) + .0001*torch.eye(6, device=q.device)
            dq = (j.transpose(-1, -2) @ torch.linalg.solve(system, error[..., None]))[..., 0]
            dq = dq * (.12 / dq.abs().amax(-1, keepdim=True).clamp_min(.12))
            q = (q + dq).maximum(self.limits[:, 0]+.01).minimum(self.limits[:, 1]-.01)
        return q


def goal_xyz(state):
    return state[..., 9:12].float() * state.new_tensor([.55, .55, .5]) + state.new_tensor([.65, 0, .22])


class CartesianHead(nn.Module):
    def __init__(self, mode='visual'):
        super().__init__()
        self.mode = mode
        self.visual = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 64), nn.SiLU())
        self.distribution = nn.Linear(64, 6)
        self.geometry = nn.Sequential(nn.Linear(6, 32), nn.SiLU())
        self.frame = nn.Sequential(nn.Linear(16*96+16+1, 256), nn.SiLU(), nn.LayerNorm(256))
        self.query = nn.Sequential(nn.Linear(16+16+3, 256), nn.SiLU(), nn.LayerNorm(256))
        self.key = nn.Linear(256, 256)
        self.output = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 18))
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        if self.mode != 'memory':
            tokens, geometry, poses, ages, mask = tokens[:, -1:], geometry[:, -1:], poses[:, -1:], ages[:, -1:], mask[:, -1:]
        visual = self.visual(tokens.float())
        dist = self.distribution(visual).float()
        mean = geometry[..., :3].float() + .25*dist[..., :3].tanh()
        logvar = -5+4*dist[..., 3:].tanh()
        uncertain = self.mode in ('uncertainty', 'memory')
        geom = torch.cat((mean if uncertain else geometry[..., :3].float(), geometry[..., 3:].float()), -1)
        if uncertain:
            # Reliability gates normalized over spatial cells, detached from action gradients.
            reliability = (-.5*logvar.detach().mean(-1)).softmax(-1)*16
            visual = visual * reliability[..., None]
        frames = self.frame(torch.cat((torch.cat((visual, self.geometry(geom)), -1).flatten(2), poses.flatten(2).float(), ages[..., None].float()/60), -1))
        query = self.query(torch.cat((state.float(), tcp.flatten(1).float(), (goal_xyz(state)-tcp[:, :3, 3])/.3), -1))
        scores = (self.key(frames)*query[:, None]).sum(-1)/16
        scores = scores.masked_fill(~mask, -torch.inf)
        context = (scores.softmax(-1)[..., None]*frames).sum(1)
        if self.mode == 'state':
            context = torch.zeros_like(context)
        out = .3*self.output(torch.cat((query, context), -1)).reshape(-1, 6, 3).tanh()
        return out.float(), {'mean': mean[:, -1], 'logvar': logvar[:, -1], 'risk': logvar[:, -1].detach().exp().mean((1, 2)).sqrt()}


class CartesianPolicy(nn.Module):
    uses_predicted_maps = True
    chunk_size = 30

    def __init__(self, backbone, head, kinematics, servo_radius=.08, execute=15, baseline=False, orientation='current'):
        super().__init__()
        self.backbone, self.head, self.kinematics = backbone, head, kinematics
        self.servo_radius, self.execute, self.baseline = servo_radius, execute, baseline
        self.orientation = orientation
        self.schedule = f'cartesian_{execute}_servo{servo_radius}'
        self.reset_episode()

    def reset_episode(self):
        self.history, self.step, self.last_risk = [], 0, 0.

    def observe_step(self, step):
        self.step = step

    def execution_horizon(self, default):
        return self.execute

    def forward(self, rgb, state, K, pose):
        base, tokens, geom = self.backbone(rgb, state, K, pose, return_features=True)
        self.history.append((tokens.detach(), geom.detach(), pose.detach(), self.step))
        self.history = self.history[-4:]
        q = state[:, :7].float()*math.pi
        with torch.autocast(device_type=q.device.type, enabled=False):
            tcp = self.kinematics(q)
        goal = goal_xyz(state)
        near = (goal-tcp[:, :3, 3]).norm(dim=-1) < self.servo_radius
        if self.baseline and not near.any():
            return base
        if self.baseline:
            waypoints = tcp[:, None, :3, 3] + (goal-tcp[:, :3, 3])[:, None]*torch.linspace(1/30, 1, 30, device=q.device)[None, :, None]
        else:
            h = self.history
            args = [torch.stack([x[i] for x in h], 1) for i in range(3)]
            age = q.new_tensor([[self.step-x[3] for x in h]])
            delta, aux = self.head(*args, age, torch.ones_like(age, dtype=torch.bool), state, tcp)
            self.last_risk = float(aux['risk'][0])
            # Piecewise linear interpolation through t=0,5,...,30.
            knots = torch.cat((torch.zeros_like(delta[:, :1]), delta), 1)
            times = torch.arange(1, 31, device=q.device)/5
            lo = times.long().clamp(max=5)
            alpha = (times-lo)[None, :, None]
            displacement = knots[:, lo]*(1-alpha)+knots[:, lo+1]*alpha
            waypoints = tcp[:, None, :3, 3] + displacement
            direct = tcp[:, None, :3, 3]+(goal-tcp[:, :3, 3])[:, None]*torch.linspace(1/30, 1, 30, device=q.device)[None, :, None]
            waypoints = torch.where(near[:, None, None], direct, waypoints)
        # Preserve current orientation during route following; the benchmark's
        # primary success condition is XYZ. Orientation success is still reported.
        with torch.autocast(device_type=q.device.type, enabled=False):
            initial = q[:, None].expand(-1, 30, -1)
            rotation = tcp[:, None, :3, :3].expand(-1, 30, -1, -1)
            if self.orientation == 'baseline':
                candidate = q[:, None] + base.float()
                candidate = candidate.maximum(self.kinematics.limits[:, 0]).minimum(self.kinematics.limits[:, 1])
                predicted_rotation = self.kinematics(candidate)[..., :3, :3]
                initial = torch.where(near[:, None, None], initial, candidate)
                rotation = torch.where(near[:, None, None, None], rotation, predicted_rotation)
            targets = self.kinematics.inverse(initial.reshape(-1, 7), waypoints.reshape(-1, 3), rotation.reshape(-1, 3, 3))
        return targets.reshape(-1, 30, 7)-q[:, None]


class GoalServoPolicy(nn.Module):
    """Apply one terminal controller identically to any learned RGB policy."""
    uses_predicted_maps = True
    chunk_size = 30

    def __init__(self, policy, kinematics, radius=.08):
        super().__init__()
        self.policy, self.kinematics, self.radius = policy, kinematics, radius
        self.schedule = f'fixed15_goal_servo_{radius}'
        self.last_risk = 0.

    def reset_episode(self):
        if hasattr(self.policy, 'reset_episode'):self.policy.reset_episode()

    def observe_step(self, step):
        if hasattr(self.policy, 'observe_step'):self.policy.observe_step(step)

    def forward(self, rgb, state, K, pose):
        chunk = self.policy(rgb,state,K,pose)
        self.last_risk = getattr(self.policy,'last_risk',0.)
        q=state[:,:7].float()*math.pi
        with torch.autocast(device_type=q.device.type,enabled=False):
            tcp=self.kinematics(q)
            goal=goal_xyz(state)
            near=(goal-tcp[:,:3,3]).norm(dim=-1)<self.radius
            if not near.any():return chunk
            trajectory=tcp[:,None,:3,3]+(goal-tcp[:,:3,3])[:,None]*torch.linspace(1/30,1,30,device=q.device)[None,:,None]
            targets=self.kinematics.inverse(q[:,None].expand(-1,30,-1).reshape(-1,7),trajectory.reshape(-1,3),tcp[:,None,:3,:3].expand(-1,30,-1,-1).reshape(-1,3,3)).reshape(-1,30,7)
            return torch.where(near[:,None,None],targets-q[:,None],chunk.float())
