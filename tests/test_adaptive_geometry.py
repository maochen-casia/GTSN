import unittest

import torch

from tsn.models.adaptive_geometry import ConsensusGeometry, weighted_surface_risk, AdaptiveGeometryPolicy
from tsn.models.geometric_energy import RouteTrust, GeometricEnergyPolicy, surface_risk
from tsn.models.compact_policy import CompactRouteHead
from tsn.models.kinematics import PandaKinematics


class PersistentGeometryTests(unittest.TestCase):
    def update(self, memory, points, step):
        points = torch.tensor(points, dtype=torch.float32).reshape(-1, 3)
        memory.update(points, torch.ones(len(points), dtype=torch.bool), step,
                      torch.tensor([.2, -.4, .3]), torch.tensor([.8, .4, .3]))

    def test_independent_view_support_not_pixel_density(self):
        memory = ConsensusGeometry()
        self.update(memory, [[.6, 0, .3], [.601, 0, .3], [.602, 0, .3]], 0)
        self.assertEqual(len(memory.points), 1)
        self.assertEqual(float(memory.support[0]), 1)
        self.assertEqual(float(memory.confidence()[0]), 0)
        self.update(memory, [[.601, 0, .3], [.602, 0, .3]], 15)
        self.assertEqual(float(memory.support[0]), 2)
        self.assertGreater(float(memory.confidence()[0]), .49)
        torch.testing.assert_close(memory.points[0], torch.tensor([.6, 0, .3]))

    def test_geometry_survives_far_beyond_four_frames(self):
        memory = ConsensusGeometry()
        for step in (0, 15, 30):
            self.update(memory, [[.6, 0, .3]], step)
        for step in range(45, 301, 15):
            self.update(memory, [[.8, .3, .4]], step)
        self.assertEqual(int(memory.last[0]), 30)
        self.assertGreater(float(memory.confidence()[0]), .99)
        self.assertEqual(memory.step-int(memory.last[0]), 270)

    def test_duplicate_and_reverse_writes_rejected(self):
        memory = ConsensusGeometry()
        self.update(memory, [[.6, 0, .3]], 15)
        for step in (15, 0):
            with self.assertRaises(ValueError):
                self.update(memory, [[.6, 0, .3]], step)

    def test_capacity_is_bounded_and_empty_observations_are_safe(self):
        memory = ConsensusGeometry(capacity=2)
        self.update(memory, [], 0)
        self.update(memory, [[.6, 0, .3], [.7, 0, .3], [.8, 0, .3]], 15)
        self.assertEqual(len(memory.points), 2)
        self.assertEqual(memory.state_bytes(), 2*(3*4+2*4+2*8))
        self.update(memory, [[float('nan'), 0, .3]], 30)
        self.assertEqual(len(memory.points), 2)

    def test_disagreement_reduces_weight_without_inventing_position(self):
        exact, noisy = ConsensusGeometry(), ConsensusGeometry()
        for memory in (exact, noisy):
            self.update(memory, [[.6, 0, .3]], 0)
        for step in (15, 30):
            self.update(exact, [[.6, 0, .3]], step)
            self.update(noisy, [[.62, 0, .3]], step)
        self.assertLess(float(noisy.confidence()[0]), float(exact.confidence()[0]))
        torch.testing.assert_close(noisy.points[0], exact.points[0])

    def test_zero_weight_and_empty_evidence_have_zero_risk(self):
        candidates = torch.zeros(1, 14, 30, 3)
        rotation = torch.eye(3).expand(1, 30, 3, 3)
        tcp = torch.eye(4)[None]
        for points, weights in ((torch.zeros(0, 3), torch.zeros(0)),
                                (torch.tensor([[0., 0, -.1]]), torch.zeros(1))):
            risk = weighted_surface_risk(candidates, rotation, tcp, points, weights)
            self.assertTrue(torch.equal(risk, torch.zeros_like(risk)))

    def test_weight_one_agrees_with_established_hand_queries(self):
        torch.manual_seed(7)
        candidates = torch.randn(1, 14, 30, 3)*.2
        rotation = torch.eye(3).expand(1, 30, 3, 3)
        tcp = torch.eye(4)[None]
        points = torch.randn(1, 40, 3)*.2
        before = surface_risk(candidates, rotation, tcp, points, torch.ones(1, 40, dtype=torch.bool))
        after = weighted_surface_risk(candidates, rotation, tcp, points[0], torch.ones(40))
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)

    def test_current_control_matches_current_frame_policy_and_reset(self):
        current = AdaptiveGeometryPolicy(*[torch.nn.Identity() for _ in range(3)], RouteTrust(), 'current')
        reference = GeometricEnergyPolicy(*[torch.nn.Identity() for _ in range(3)], RouteTrust(), 'current_frame')
        self.assertEqual(current.route_history_length, 1)
        tcp = torch.eye(4)[None]
        route = torch.zeros(1, 30, 3)
        rotation = torch.eye(3).expand(1, 30, 3, 3)
        for model in (current, reference):
            model.clouds = [(torch.tensor([[[.1, .1, .1]]]), torch.ones(1, 1, dtype=torch.bool))]
            model.current_goal = torch.tensor([[.3, .3, .3]])
        args = (route, rotation, tcp, torch.tensor([False]), tcp)
        torch.testing.assert_close(current.refine_waypoints(*args), reference.refine_waypoints(*args), atol=0, rtol=0)
        self.update(current.geometry_memory, [[.6, 0, .3]], 0)
        current.reset_episode()
        self.assertIsNone(current.geometry_memory.points)
        self.assertEqual(current.history, [])
        self.assertEqual(current.clouds, [])
        self.assertEqual(current.diagnostics, [])

    def test_current_frame_output_is_inert_to_all_previous_rgb_and_features(self):
        class Perception(torch.nn.Module):
            def forward(self, rgb, *args, **kwargs):
                value = rgb.float().mean()
                return (torch.zeros(1, 30, 7), torch.ones(1, 16, 768)*value,
                        torch.ones(1, 16, 6)*value, torch.full((1, 6, 20, 20), -10.))
        torch.manual_seed(11)
        model = AdaptiveGeometryPolicy(Perception(), CompactRouteHead(), PandaKinematics(), RouteTrust(), 'current').eval()
        model.route_actions = lambda delta, *args: delta
        state = torch.zeros(1, 16)
        state[:, :7] = torch.tensor([[0., .4, 0., -1.96, 0., 2.35, .78]])/torch.pi
        outputs = []
        with torch.inference_mode():
            for prefix in (0, 100):
                model.reset_episode()
                for step in range(0, 105, 15):
                    model.observe_step(step)
                    rgb = torch.full((1, 1, 1, 3), prefix if step < 90 else 7, dtype=torch.uint8)
                    output = model(rgb, state, torch.eye(3)[None], torch.eye(4)[None])
                    self.assertEqual(len(model.history), 1)
                    self.assertEqual(len(model.clouds), 1)
                    self.assertIsNone(model.geometry_memory.points)
                outputs.append(output)
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
