"""Task-relative Cartesian route prediction with calibrated robot kinematics."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn


class PandaKinematics(nn.Module):
    """URDF-derived FK/Jacobian; no scene or simulator state is used."""
    def __init__(self):
        """Load the Panda base-to-TCP chain from ManiSkill's bundled URDF.

        Args:
            None.

        Returns:
            None. Registers float32 buffers: origins (J, 4, 4) for all J
            chain joints including fixed joints, axes (7, 3) for revolute
            joints, and lower/upper angle limits (7, 2) in radians. Origin
            translations are metres; indices maps each chain joint to an
            actuated index or -1 for a fixed joint.
        """
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
        """Compute TCP forward kinematics and optionally its spatial Jacobian.

        Args:
            q (torch.Tensor): Floating arm angles (..., 7), radians. Any
                leading batch/horizon dimensions are retained; cast to float32.
            jacobian (bool): Also return derivatives of TCP position/rotation.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Float32
            TCP-to-base transform (..., 4, 4), with translation in metres.
            If jacobian=True, returns (transform, J), where J is float32
            (..., 6, 7). The first three rows are base-frame linear velocity
            per joint rate (m/rad), the last three angular velocity per joint
            rate. Buffers and input must reside on the same device.

        Raises:
            ValueError: A revolute URDF axis is not the supported local z axis.
        """
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
        """Track target poses with bounded damped least-squares IK updates.

        Args:
            q (torch.Tensor): Floating seed angles (..., 7), radians; copied
                before iteration. Seeds may be measured or proposed joints.
            xyz (torch.Tensor): Floating target base-frame positions (..., 3), m.
            rotation (torch.Tensor): Floating target TCP-to-base rotation
                matrices (..., 3, 3), with matching leading dimensions.
            iterations (int): Number of updates, default 12; no early stopping.

        Returns:
            torch.Tensor: Float32 fitted angles (..., 7), radians. Each update
            limits the largest joint change to .12 rad and clamps angles .01
            rad inside URDF limits. The result is approximate; target pose
            residuals are not returned and convergence is not guaranteed.
        """
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
