"""Robot proxy coordinates and no-evidence behavior of IK realization scoring."""
import unittest

import torch

from tsn.models.body_clearance import BodyClearancePolicy
from tsn.models.compact_policy import CompactRouteHead
from tsn.models.kinematics import PandaKinematics


class BodyClearanceTests(unittest.TestCase):
    def make_policy(self):
        return BodyClearancePolicy(torch.nn.Identity(), CompactRouteHead(), PandaKinematics())

    def test_body_tcp_matches_original_forward_kinematics(self):
        policy = self.make_policy()
        q = torch.tensor([[0., .4, 0., -1.96, 0., 2.35, .78], [.1, .3, -.2, -1.8, .1, 2.1, .8]])
        body = policy.body_points(q)
        self.assertEqual(body.shape, (2, 10, 3))
        torch.testing.assert_close(body[:, 7], policy.kinematics(q)[:, :3, 3], rtol=0, atol=0)
        torch.testing.assert_close((body[:, 7]-body[:, 9]).norm(dim=-1), torch.full((2,), .1))

    def test_no_surface_evidence_preserves_exact_joint_chunk(self):
        policy = self.make_policy()
        q = torch.tensor([[0., .4, 0., -1.96, 0., 2.35, .78]])
        state = torch.zeros(1, 16)
        state[:, :7] = q/torch.pi
        tcp = policy.kinematics(q)
        policy.current_tcp = tcp
        policy.current_goal = tcp[:, :3, 3]+torch.tensor([[0., .2, 0.]])
        policy.proposed_waypoints = tcp[:, None, :3, 3].expand(-1, 30, -1)
        policy.proposed_rotation = tcp[:, None, :3, :3].expand(-1, 30, -1, -1)
        policy.clouds = [(torch.zeros(1, 4, 3), torch.ones(1, 4)*.01, torch.zeros(1, 4, dtype=torch.bool))]
        original = torch.zeros(1, 30, 7)
        with torch.inference_mode():
            actual = policy.choose_realization(original, state)
        torch.testing.assert_close(original, actual, rtol=0, atol=0)
