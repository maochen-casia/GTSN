import unittest

import torch
from torch.nn import functional as F

from tsn.cli.uncertainty import pinball
from tsn.models.adaptive_geometry import AdaptiveGeometryPolicy
from tsn.models.compact_policy import CompactPolicy, CompactRouteHead
from tsn.models.geometric_energy import RouteTrust
from tsn.models.kinematics import PandaKinematics
from tsn.models.uncertain_clearance import (
    PointUncertainty, UncertainClearancePolicy, uncertainty_features,
    padding_factor, inflated_surface_risk,
)


class UncertainClearanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_error_quantile_is_bounded_and_has_training_gradients(self):
        model = PointUncertainty()
        radius = model(torch.randn(3, 400, 74))
        torch.testing.assert_close(radius, torch.full_like(radius, .04))
        radius.mean().backward()
        self.assertGreater(model.network[-1].weight.grad.norm(), 0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4865)
        with torch.no_grad():
            model.network[-1].weight.fill_(100)
        radius = model(torch.randn(3, 400, 74))
        self.assertTrue(((radius>=.005)&(radius<=.2000001)).all())

    def test_visual_cells_are_aligned_and_nonfinite_points_are_safe(self):
        visual = torch.arange(16).float()[None, :, None].expand(1, 16, 64)
        points = torch.zeros(1, 400, 3)
        points[0, 0] = float('nan')
        features = uncertainty_features(visual, points, torch.eye(4)[None], torch.eye(4)[None])
        self.assertEqual(features.shape, (1, 400, 74))
        self.assertTrue(torch.isfinite(features).all())
        grid = features[0, :, 0].reshape(20, 20)
        self.assertEqual(float(grid[0, 0]), 0)
        self.assertEqual(float(grid[0, 5]), 1)
        self.assertEqual(float(grid[5, 0]), 4)
        self.assertEqual(float(grid[-1, -1]), 15)

    def test_high_uncertainty_has_larger_bounded_padding(self):
        values = padding_factor(torch.tensor([0., .015, .04, .1, .2]))
        torch.testing.assert_close(values[:2], torch.zeros(2))
        torch.testing.assert_close(values[-2:], torch.ones(2))
        self.assertTrue((values[1:]>=values[:-1]).all())

    def test_larger_padding_increases_proximity_and_is_translation_invariant(self):
        candidates = torch.zeros(1, 14, 30, 3)
        rotation = torch.eye(3).expand(1, 30, 3, 3)
        tcp = torch.eye(4)[None]
        points = torch.tensor([[.09, 0., -.1]])
        base = inflated_surface_risk(candidates, rotation, tcp, points, torch.zeros(1))
        padded = inflated_surface_risk(candidates, rotation, tcp, points, torch.full((1,), .03))
        self.assertTrue((padded>base).all())
        shift = torch.tensor([.4, -.2, .3])
        tcp[:, :3, 3] += shift
        changed = inflated_surface_risk(candidates+shift, rotation, tcp, points+shift, torch.full((1,), .03))
        torch.testing.assert_close(padded, changed, rtol=1e-5, atol=1e-6)

    def test_empty_and_invalid_evidence_has_zero_risk(self):
        candidates = torch.zeros(1, 14, 30, 3)
        rotation = torch.eye(3).expand(1, 30, 3, 3)
        tcp = torch.eye(4)[None]
        for points, padding in ((torch.empty(0, 3), torch.empty(0)),
                                (torch.full((1, 3), float('nan')), torch.tensor([.02]))):
            risk = inflated_surface_risk(candidates, rotation, tcp, points, padding)
            torch.testing.assert_close(risk, torch.zeros_like(risk))

    def test_pinball_targets_upper_quantile_and_empty_labels_are_inert(self):
        target = torch.tensor([[.1]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        under = torch.tensor([[.05]], requires_grad=True)
        pinball(under, target, mask).backward()
        self.assertLess(float(under.grad), 0)
        over = torch.tensor([[.15]], requires_grad=True)
        pinball(over, target, mask).backward()
        self.assertGreater(float(over.grad), 0)
        self.assertEqual(float(pinball(over, target, ~mask)), 0)

    def test_fixed_and_none_match_C1_and_compact_complete_commands(self):
        class Backbone(torch.nn.Module):
            def forward(self, rgb, *args, **kwargs):
                index = float(rgb[0, 0, 0, 0])
                dense = torch.zeros(1, 6, 20, 20)
                dense[:, 0] = index*.008
                dense[:, 1] = torch.linspace(-.3, .3, 20)[None, None]
                dense[:, 2] = .2
                geometry = F.adaptive_avg_pool2d(dense, (4, 4)).flatten(2).transpose(1, 2)
                output = (torch.zeros(1, 30, 7), torch.ones(1, 16, 768)*index, geometry)
                return (*output, dense) if kwargs.get('return_maps') else output
        torch.manual_seed(8)
        modules = (Backbone(), CompactRouteHead(), PandaKinematics(), RouteTrust())
        q = torch.tensor([[0., .4, 0., -1.96, 0., 2.35, .78]])
        state = torch.zeros(1, 16)
        state[:, :7] = q/torch.pi
        state[:, 9:12] = torch.tensor([[.15, .4, .2]])
        for mode in ('fixed', 'none'):
            current = UncertainClearancePolicy(*modules, PointUncertainty(), mode).eval()
            reference = (AdaptiveGeometryPolicy(*modules, 'unconfirmed') if mode=='fixed' else
                         CompactPolicy(*modules[:3])).eval()
            with torch.inference_mode():
                for step in range(0, 180, 15):
                    rgb = torch.full((1, 1, 1, 3), step//15, dtype=torch.uint8)
                    outputs = []
                    for policy in (reference, current):
                        policy.observe_step(step)
                        outputs.append(policy(rgb, state, torch.eye(3)[None], torch.eye(4)[None]))
                    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            self.assertIsNotNone(current.map_uncertainty)
            current.reset_episode()
            self.assertIsNone(current.map_uncertainty)
            self.assertEqual(current.uncertainty_clouds, [])

    def test_invalid_parameters_rejected(self):
        modules = [torch.nn.Identity() for _ in range(4)]
        for padding in (float('nan'), -.01, .06):
            with self.assertRaises(ValueError):
                UncertainClearancePolicy(*modules, PointUncertainty(), max_padding=padding)


if __name__ == '__main__':
    unittest.main()
