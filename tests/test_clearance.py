"""Deterministic hand clearance preserves routes, surface memory, and resets."""
import unittest
from unittest.mock import Mock

import torch

from tsn.models.clearance_policy import ClearancePolicy, load_clearance_policy
from tsn.models.compact_policy import CompactRouteHead, RouteController, compact_state


class ClearanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def make_policy(self, backbone=None, head=None, **kwargs):
        return ClearancePolicy(backbone if backbone is not None else torch.nn.Identity(),
                               head if head is not None else CompactRouteHead(),
                               torch.nn.Identity(), **kwargs)

    def inputs(self):
        tcp = torch.eye(4)[None]
        tcp[:, :3, 3] = torch.tensor([.5, -.2, .25])
        waypoints = tcp[:, None, :3, 3] + torch.tensor([0., .10, 0.])[None, None] * torch.linspace(1/30, 1, 30)[None, :, None]
        rotation = torch.eye(3).repeat(1, 30, 1, 1)
        return waypoints, rotation, tcp, torch.tensor([False]), tcp

    def test_default_is_selected_deterministic_hand_method(self):
        policy = self.make_policy()
        self.assertEqual((policy.margin, policy.penalty), (.04, .08))
        self.assertEqual(policy.schedule, 'fixed15_clearance_deterministic')
        with self.assertRaises(TypeError):
            RouteController(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity())

    def test_invalid_scoring_parameters_are_rejected(self):
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(margin=value), self.assertRaises(ValueError):
                self.make_policy(margin=value)
        for value in (-1., float('nan'), float('inf')):
            with self.subTest(penalty=value), self.assertRaises(ValueError):
                self.make_policy(penalty=value)

    def test_no_valid_surfaces_preserves_proposal_exactly(self):
        policy = self.make_policy()
        inputs = self.inputs()
        policy.current_goal = torch.tensor([[.5, .3, .25]])
        policy.clouds = [(torch.zeros(1, 3, 3), torch.zeros(1, 3, dtype=torch.bool))]
        actual = policy.refine_waypoints(*inputs)
        torch.testing.assert_close(actual, inputs[0], rtol=0, atol=0)
        self.assertEqual(policy.diagnostics[-1]['choice'], 0)
        self.assertEqual(policy.last_risk, 0.)

    def test_near_goal_correction_fades_to_zero(self):
        policy = self.make_policy()
        inputs = self.inputs()
        policy.current_goal = inputs[2][:, :3, 3] + torch.tensor([[0., .01, 0.]])
        policy.clouds = [(torch.tensor([[[.5, -.1, .25]]]), torch.tensor([[True]]))]
        actual = policy.refine_waypoints(*inputs)
        torch.testing.assert_close(actual, inputs[0], rtol=0, atol=0)
        self.assertEqual(policy.diagnostics[-1]['correction_m'], 0.)

    def test_obstacle_changes_plan_with_finite_bounded_correction(self):
        policy = self.make_policy()
        inputs = self.inputs()
        policy.current_goal = torch.tensor([[.5, .3, .25]])
        policy.clouds = [(torch.tensor([[[.52, -.17, .15]]]), torch.tensor([[True]]))]
        actual = policy.refine_waypoints(*inputs)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float((actual - inputs[0]).abs().max()), 0.)
        self.assertLess(float((actual - inputs[0]).norm(dim=-1).max()), .065)
        self.assertGreater(policy.last_risk, 0.)
        self.assertEqual(policy.diagnostics[-1]['points'], 1)
        policy.reset_episode()
        self.assertEqual(policy.clouds, [])
        self.assertEqual(policy.diagnostics, [])
        self.assertIsNone(policy.current_goal)
        self.assertEqual(policy.last_risk, 0.)

    def test_swept_hand_detects_surface_below_tcp(self):
        policy = self.make_policy()
        waypoints, rotation, tcp, _, _ = self.inputs()
        policy.current_goal = torch.tensor([[.5, .3, .25]])
        point = torch.tensor([[[.5, -.16, .14]]])
        policy.clouds = [(point, torch.tensor([[True]]))]
        candidates, _, _ = policy.route_candidates(waypoints, tcp)
        risk, _ = policy.hand_proximity(candidates, rotation, tcp)
        tcp_distance = (waypoints[:, 2:15:3] - point).square().sum(-1)
        tcp_proximity = (-.5 * tcp_distance / policy.margin**2).exp().mean()
        self.assertGreater(float(risk[0, 0]), float(tcp_proximity) * 5)

    def test_points_near_measured_tcp_are_suppressed(self):
        policy = self.make_policy()
        inputs = self.inputs()
        policy.current_goal = torch.tensor([[.5, .3, .25]])
        policy.clouds = [(inputs[2][:, None, :3, 3].clone(), torch.tensor([[True]]))]
        actual = policy.refine_waypoints(*inputs)
        torch.testing.assert_close(actual, inputs[0], rtol=0, atol=0)
        self.assertEqual(policy.diagnostics[-1]['points'], 0)

    def test_perception_uses_only_predicted_points_and_keeps_four_clouds(self):
        dense = torch.zeros(1, 6, 4, 4, requires_grad=True)
        outputs = (torch.zeros(1, 30, 7), torch.zeros(1, 16, 768), torch.zeros(1, 16, 6), dense)
        backbone = Mock(return_value=outputs)
        # The clearance stage must not call the learned distribution head.
        head = Mock(side_effect=AssertionError('Clearance queried route-head uncertainty'))
        policy = self.make_policy(backbone=backbone, head=head)
        state = torch.zeros(1, 16)
        args = (torch.zeros(1, 4, 4, 3, dtype=torch.uint8), state,
                torch.eye(3)[None], torch.eye(4)[None])
        for index in range(6):
            with torch.no_grad():
                dense[:, 0] = index * .01
            actual = policy.perceive(*args)
            for observed, expected in zip(actual, outputs[:3]):
                self.assertIs(observed, expected)
        self.assertEqual(len(policy.clouds), 4)
        for index, (points, valid) in enumerate(policy.clouds, start=2):
            self.assertEqual(points.shape, (1, 400, 3))
            self.assertEqual(valid.shape, (1, 400))
            self.assertFalse(points.requires_grad)
            self.assertTrue(valid.all())
            torch.testing.assert_close(points[0, 0], torch.tensor([.65 + index * .0055, 0., .22]))
        head.assert_not_called()
        backbone.assert_called_with(*args, return_features=True, return_maps=True)

    def test_multi_episode_input_is_rejected_before_perception(self):
        backbone = Mock()
        policy = self.make_policy(backbone=backbone)
        with self.assertRaisesRegex(ValueError, 'one episode'):
            policy.perceive(torch.zeros(2, 1, 1, 3), torch.zeros(2, 16), None, None)
        backbone.assert_not_called()
        self.assertEqual(policy.clouds, [])

    def test_bfloat16_surface_decode_preserves_original_rounding(self):
        outputs = (torch.zeros(1, 30, 7), torch.zeros(1, 16, 768),
                   torch.zeros(1, 16, 6), torch.zeros(1, 6, 4, 4, dtype=torch.bfloat16))
        policy = self.make_policy(backbone=Mock(return_value=outputs))
        policy.perceive(torch.zeros(1, 4, 4, 3), torch.zeros(1, 16),
                        torch.eye(3)[None], torch.eye(4)[None])
        points, valid = policy.clouds[0]
        # Constants were historically rounded to the dense map dtype first.
        expected = torch.tensor([.6484375, 0., .2197265625]).expand(1, 400, 3)
        torch.testing.assert_close(points, expected, rtol=0, atol=0)
        self.assertTrue(valid.all())

    def test_loader_constructs_clearance_from_shared_components(self):
        from unittest.mock import patch
        components = (torch.nn.Identity(), CompactRouteHead(), torch.nn.Identity(), object(), {'splits': {}})
        with patch('tsn.models.clearance_policy.load_route_components', return_value=components) as load:
            policy, maps, metadata = load_clearance_policy('model.pt', margin=.03, penalty=.1)
        self.assertIsInstance(policy, ClearancePolicy)
        self.assertFalse(policy.training)
        self.assertEqual((policy.margin, policy.penalty), (.03, .1))
        self.assertIs(maps, components[3])
        self.assertIs(metadata, components[4])
        load.assert_called_once_with('model.pt', 'cpu', allow_training_source=False)

    def test_removed_evidence_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'single_memory'):
            compact_state({'architecture': 'route_evidence', 'base_checkpoint': 'unused.pt'})
