"""One robot and goal representation used for both demonstrations and rollouts."""

import math

import torch

from tsn.features.maps import MapSettings


def policy_state(qpos: torch.Tensor, goal_pose: torch.Tensor, settings: MapSettings) -> torch.Tensor:
    """Map qpos (B,9) and base-frame XYZ+wxyz goal (B,7) to state (B,16).

    Arm angles scale by pi, finger positions by 0.04 m. Goal XYZ uses the same
    center/scale as point maps. Canonical unit quaternions remove sign ambiguity.
    """
    if qpos.ndim != 2 or qpos.shape[1] != 9 or goal_pose.shape != (len(qpos), 7):
        raise ValueError("Expected batched Panda qpos (B,9) and goal pose (B,7)")
    robot = qpos.float() / qpos.new_tensor([math.pi] * 7 + [0.04, 0.04])
    xyz = ((goal_pose[:, :3] - goal_pose.new_tensor(settings.point_center_m)) /
           goal_pose.new_tensor(settings.point_scale_m)).clamp(-2, 2)
    quaternion = goal_pose[:, 3:].float()
    norm = quaternion.norm(dim=-1, keepdim=True)
    if torch.any(norm < 1e-8):
        raise ValueError("Goal orientation has a zero quaternion")
    quaternion = quaternion / norm
    quaternion = torch.where(quaternion[:, :1] < 0, -quaternion, quaternion)
    return torch.cat((robot, xyz, quaternion), dim=-1)

