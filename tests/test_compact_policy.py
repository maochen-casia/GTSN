"""Compact-only execution never repairs the learned route for clearance."""
import unittest
from unittest.mock import patch

import torch

from tsn.models.clearance_policy import ClearancePolicy
from tsn.models.compact_policy import CompactPolicy, CompactRouteHead, load_compact_policy
from tsn.models.kinematics import PandaKinematics
from tsn.models.persistent_policy import PersistentSceneHead, metric_points


class Perception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, rgb, state, K, pose, return_features=False, return_maps=False):
        self.calls.append((return_features, return_maps))
        values = (torch.zeros(1, 30, 7), torch.zeros(1, 16, 768), torch.zeros(1, 16, 6))
        return (*values, torch.zeros(1, 6, 20, 20)) if return_maps else values


class CompactPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def inputs(self, kin):
        q = torch.tensor([[0., .4, 0., -1.96, 0., 2.35, .78]])
        tcp = kin(q)
        state = torch.zeros(1, 16)
        state[:, :7] = q/torch.pi
        state[:, 9:12] = (tcp[:, :3, 3]+torch.tensor([[.2, 0., 0.]])-torch.tensor([.65, 0, .22]))/torch.tensor([.55, .55, .50])
        return torch.zeros(1, 1, 1, 3, dtype=torch.uint8), state, torch.eye(3)[None], torch.eye(4)[None]

    def test_identity_refinement_preserves_waypoints_exactly(self):
        model = CompactPolicy(Perception(), CompactRouteHead(), torch.nn.Identity())
        waypoints = torch.randn(1, 30, 3)
        self.assertIs(model.refine_waypoints(waypoints, None, None, None, None), waypoints)
        self.assertFalse(hasattr(model, 'clouds'))
        self.assertFalse(hasattr(model, 'route_candidates'))

    def test_inverse_kinematics_receives_uncorrected_interpolated_route(self):
        class Knots(torch.nn.Module):
            def forward(self, *args):
                delta = torch.zeros(1, 6, 3)
                delta[0, :, 0] = torch.arange(1, 7)*.01
                return delta, {}
        kin = PandaKinematics()
        model = CompactPolicy(Perception(), Knots(), kin).eval()
        args = self.inputs(kin)
        tcp = kin(args[1][:, :7]*torch.pi)
        with torch.inference_mode(), patch.object(kin, 'inverse', side_effect=lambda initial, xyz, rotation: initial) as inverse:
            model(*args)
        expected = tcp[:, None, :3, 3].expand(1, 30, 3).clone()
        expected[0, :, 0] += torch.arange(1, 31)*.002
        torch.testing.assert_close(inverse.call_args.args[1], expected.reshape(30, 3), rtol=0, atol=1e-7)

    def test_matches_shared_controller_when_clearance_is_identity(self):
        for head in (CompactRouteHead(), PersistentSceneHead(use_points=False), PersistentSceneHead()):
            if isinstance(head, PersistentSceneHead):
                torch.nn.init.normal_(head.memory_residual.weight, std=.01)
            kin = PandaKinematics()
            compact = CompactPolicy(Perception(), head, kin).eval()
            clearance = ClearancePolicy(Perception(), head, kin).eval()
            args = self.inputs(kin)
            with torch.inference_mode(), patch.object(clearance, 'refine_waypoints', side_effect=lambda waypoints, *args: waypoints):
                for step in (0, 15, 30):
                    compact.observe_step(step)
                    clearance.observe_step(step)
                    torch.testing.assert_close(compact(*args), clearance(*args), rtol=0, atol=0)
            self.assertFalse(hasattr(compact, 'clouds'))
            if isinstance(head, PersistentSceneHead):
                self.assertIsNotNone(compact.scene_memory)

    def test_finer_points_requested_only_by_detailed_head(self):
        for head in (CompactRouteHead(), PersistentSceneHead(use_points=False), PersistentSceneHead()):
            backbone = Perception()
            model = CompactPolicy(backbone, head, torch.nn.Identity())
            model.perceive(torch.zeros(1, 1, 1, 3, dtype=torch.uint8), torch.zeros(1, 16), None, None)
            detailed = isinstance(head, PersistentSceneHead) and head.use_points
            self.assertEqual(backbone.calls, [(True, detailed)])
            if detailed:
                torch.testing.assert_close(model.current_points, metric_points(torch.zeros(1, 6, 20, 20)))
            else:
                self.assertIsNone(model.current_points)
            model.reset_episode()
            self.assertIsNone(model.current_points)

    def test_loader_uses_compact_class_and_preserves_current_only_history(self):
        components = (Perception(), CompactRouteHead(), torch.nn.Identity(), object(), {'route_history_length': 1})
        with patch('tsn.models.compact_policy.load_route_components', return_value=components) as load:
            policy, maps, metadata = load_compact_policy('model.pt')
        self.assertIsInstance(policy, CompactPolicy)
        self.assertNotIsInstance(policy, ClearancePolicy)
        self.assertEqual(policy.route_history_length, 1)
        self.assertFalse(policy.training)
        self.assertIs(maps, components[3])
        load.assert_called_once_with('model.pt', 'cpu')

    def test_rejects_multiple_streams_before_perception(self):
        backbone = Perception()
        model = CompactPolicy(backbone, CompactRouteHead(), torch.nn.Identity())
        with self.assertRaisesRegex(ValueError, 'one episode'):
            model.perceive(None, torch.zeros(2, 16), None, None)
        self.assertEqual(backbone.calls, [])
