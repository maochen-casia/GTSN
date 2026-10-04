"""Distinguish contradictory geometry from occluded or unobserved geometry."""
import unittest

import torch

from tsn.models.grounded_geometry import GroundedGeometryPolicy, contradicted_points
from tsn.models.clearance_policy import ClearancePolicy
from tsn.models.compact_policy import CompactRouteHead


class GroundedGeometryTests(unittest.TestCase):
    def test_metric_calibration_recomputes_workspace_validity(self):
        class FixedBackbone(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return (torch.zeros(1, 30, 7), torch.zeros(1, 16, 768),
                        torch.zeros(1, 16, 6), torch.zeros(1, 6, 80, 80))
        head = CompactRouteHead()
        policy = GroundedGeometryPolicy(FixedBackbone(), head, torch.nn.Identity(),
            mode='deterministic', geometry_update='calibrated', translation_m=[0., 0., -.3])
        args = (torch.zeros(1, 80, 80, 3, dtype=torch.uint8), torch.zeros(1, 16),
                torch.eye(3)[None], torch.eye(4)[None])
        policy.perceive(*args)
        self.assertFalse(bool(policy.clouds[-1][2].any()))
        torch.testing.assert_close(policy.clouds[-1][0][0, 0], torch.tensor([.65, 0., -.08]))

    def test_pooled_mean_uses_existing_head_metric_correction(self):
        class FixedBackbone(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return (torch.zeros(1, 30, 7), torch.zeros(1, 16, 768),
                        torch.zeros(1, 16, 6), torch.zeros(1, 6, 80, 80))
        head = CompactRouteHead()
        with torch.no_grad():
            head.distribution.weight.zero_()
            head.distribution.bias.copy_(torch.tensor([.2, -.1, .3, 0., 0., 0.]))
        policy = GroundedGeometryPolicy(FixedBackbone(), head, torch.nn.Identity(),
                                        mode='deterministic', geometry_update='pooled_mean')
        policy.perceive(torch.zeros(1, 80, 80, 3, dtype=torch.uint8), torch.zeros(1, 16),
                        torch.eye(3)[None], torch.eye(4)[None])
        expected = torch.tensor([.65, 0., .22])+.25*torch.tensor([.2, -.1, .3]).tanh()*torch.tensor([.55, .55, .5])
        torch.testing.assert_close(policy.clouds[-1][0], expected.expand(1, 400, 3))

    def test_alignment_intervention_preserves_proposal_inputs_and_shifts_each_cloud_once(self):
        class FixedBackbone(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return (torch.zeros(1, 30, 7), torch.zeros(1, 16, 768),
                        torch.zeros(1, 16, 6), torch.zeros(1, 6, 80, 80))
        backbone, head, kin = FixedBackbone(), CompactRouteHead(), torch.nn.Identity()
        original = ClearancePolicy(backbone, head, kin, mode='deterministic')
        shifted = GroundedGeometryPolicy(backbone, head, kin, mode='deterministic', geometry_update='shift_pos')
        args = (torch.zeros(1, 80, 80, 3, dtype=torch.uint8), torch.zeros(1, 16),
                torch.eye(3)[None], torch.eye(4)[None])
        for _ in range(2):
            for first, second in zip(original.perceive(*args), shifted.perceive(*args)):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
        for first, second in zip(original.clouds, shifted.clouds):
            torch.testing.assert_close(second[0], first[0]+torch.tensor([.1, 0., 0.]), rtol=0, atol=0)
            torch.testing.assert_close(first[2], second[2])
        shifted.reset_episode()
        self.assertEqual(shifted.clouds, [])

    def test_only_visible_contradictions_are_removed(self):
        history = torch.tensor([[[0., 0., .4], [0., 0., .8], [1., 0., .4], [0., 0., .59]]])
        current = torch.tensor([[[0., 0., .6]]])
        removed = contradicted_points(history, current, torch.tensor([[True]]),
                                      torch.eye(4)[None], .04)
        self.assertEqual(removed.tolist(), [[True, False, False, False]])

    def test_invalid_current_evidence_cannot_erase_memory(self):
        points = torch.tensor([[[0., 0., .3]]])
        removed = contradicted_points(points, points*2, torch.tensor([[False]]),
                                      torch.eye(4)[None], .04)
        self.assertFalse(bool(removed.any()))

    def test_joint_coordinate_transform_preserves_visibility_decision(self):
        old = torch.tensor([[[0., 0., .4], [0., 0., .8], [1., 0., .4]]])
        new = torch.tensor([[[0., 0., .6]]])
        valid = torch.tensor([[True]])
        pose = torch.eye(4)[None]
        expected = contradicted_points(old, new, valid, pose, .04)
        transform = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        shift = torch.tensor([.6, -.2, .1])
        moved = pose.clone()
        moved[:, :3, :3] = transform
        moved[:, :3, 3] = shift
        actual = contradicted_points(old@transform.T+shift, new@transform.T+shift,
                                     valid, moved, .04)
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
