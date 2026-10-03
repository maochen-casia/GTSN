"""The optional geometric correction must preserve clear routes and resets."""
import unittest

import torch

from tsn.models.clearance_policy import ClearancePolicy
from tsn.models.compact_policy import CompactRouteHead


class ClearanceTests(unittest.TestCase):
    def make_policy(self, mode='full'):
        return ClearancePolicy(torch.nn.Identity(), CompactRouteHead(), torch.nn.Identity(), mode=mode)

    def inputs(self):
        tcp = torch.eye(4)[None]
        tcp[:, :3, 3] = torch.tensor([.5, -.2, .25])
        waypoints = tcp[:, None, :3, 3] + torch.tensor([0., .10, 0.])[None, None]*torch.linspace(1/30, 1, 30)[None, :, None]
        rotation = torch.eye(3).repeat(1, 30, 1, 1)
        return waypoints, rotation, tcp, torch.tensor([False]), tcp

    def test_empty_observed_cloud_is_exact_identity(self):
        policy = self.make_policy()
        inputs = self.inputs()
        policy.current_goal = torch.tensor([[.5, .3, .25]])
        policy.clouds = [(torch.zeros(1, 3, 3), torch.ones(1, 3)*.04, torch.zeros(1, 3, dtype=torch.bool))]
        actual = policy.refine_waypoints(*inputs)
        torch.testing.assert_close(actual, inputs[0], rtol=0, atol=0)

    def test_near_goal_correction_fades_to_zero(self):
        policy = self.make_policy()
        inputs = self.inputs()
        policy.current_goal = inputs[2][:, :3, 3]+torch.tensor([[0., .01, 0.]])
        policy.clouds = [(torch.tensor([[[.5, -.1, .25]]]), torch.tensor([[.03]]), torch.tensor([[True]]))]
        actual = policy.refine_waypoints(*inputs)
        torch.testing.assert_close(actual, inputs[0], rtol=0, atol=0)

    def test_obstacle_changes_plan_with_finite_bounded_correction(self):
        policy = self.make_policy()
        policy.penalty = .01
        inputs = self.inputs()
        policy.current_goal = torch.tensor([[.5, .3, .25]])
        policy.clouds = [(torch.tensor([[[.52, -.17, .15]]]), torch.tensor([[.03]]), torch.tensor([[True]]))]
        actual = policy.refine_waypoints(*inputs)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float((actual-inputs[0]).abs().max()), 0.)
        self.assertLess(float((actual-inputs[0]).norm(dim=-1).max()), .065)
        policy.reset_episode()
        self.assertEqual(policy.clouds, [])
        self.assertEqual(policy.diagnostics, [])

    def test_swept_hand_detects_surface_below_tcp(self):
        full, tcp = self.make_policy(), self.make_policy('tcp')
        inputs = self.inputs()
        for policy in (full, tcp):
            policy.current_goal = torch.tensor([[.5, .3, .25]])
            policy.clouds = [(torch.tensor([[[.5, -.16, .14]]]), torch.tensor([[.001]]), torch.tensor([[True]]))]
            policy.refine_waypoints(*inputs)
        self.assertGreater(full.last_risk, tcp.last_risk)
