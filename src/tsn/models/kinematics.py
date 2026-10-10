"""Task-relative Cartesian route prediction with calibrated robot kinematics."""
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from tsn.simulation.robot import robot_spec


class PandaKinematics(nn.Module):
    """URDF-derived Panda/FR3 FK and Jacobian; no simulator state is used."""
    def __init__(self, robot='panda'):
        """Load the configured Franka base-to-TCP chain."""
        super().__init__()
        from scipy.spatial.transform import Rotation
        spec = robot_spec(robot)
        root = ET.parse(spec['urdf']).getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, link = [], spec['tcp']
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
        """Compute TCP forward kinematics and optionally its spatial Jacobian."""
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
        """Track target poses with bounded damped least-squares IK updates."""
        q = q.float().clone()
        xyz, rotation = xyz.float(), rotation.float()
        for _ in range(iterations):
            pose, jac = self(q, True)
            dp = xyz-pose[..., :3, 3]
            dr = .5 * torch.linalg.cross(pose[..., :3, :3], rotation, dim=-2).sum(-1)
            error = torch.cat((dp, dr), -1)
            weights = error.new_tensor([1, 1, 1, .3, .3, .3])
            # Downweight orientation relative to translation, then solve a
            # damped 6D task-space system rather than invert a 7D joint matrix.
            j = jac * weights[..., None]
            error = error * weights
            system = j @ j.transpose(-1, -2) + .0001*torch.eye(6, device=q.device)
            dq = (j.transpose(-1, -2) @ torch.linalg.solve(system, error[..., None]))[..., 0]
            dq = dq * (.12 / dq.abs().amax(-1, keepdim=True).clamp_min(.12))
            q = (q + dq).maximum(self.limits[:, 0]+.01).minimum(self.limits[:, 1]-.01)
        return q

    def arm_points(self, q):
        """Seven moving joint-origin probes in the base frame, from measured q.

        These describe arm posture for learned context, rather than collision
        volumes. The fixed tool distance field still defines physical contact.
        """
        q = q.float()
        transform = torch.eye(4, device=q.device).expand(*q.shape[:-1], 4, 4).clone()
        points = []
        for origin, index in zip(self.origins, self.indices):
            transform = transform@origin
            if index >= 0:
                points.append(transform[..., :3, 3])
                c, s = q[..., index].cos(), q[..., index].sin()
                rotation = torch.eye(4, device=q.device).expand_as(transform).clone()
                rotation[..., 0, 0], rotation[..., 1, 1] = c, c
                rotation[..., 0, 1], rotation[..., 1, 0] = -s, s
                transform = transform@rotation
        return torch.stack(points, -2)
