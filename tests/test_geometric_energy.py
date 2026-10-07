import unittest

import torch

from tsn.models.geometric_energy import RouteTrust, GeometricEnergyPolicy, surface_risk
from tsn.models.clearance_policy import ClearancePolicy


class GeometricEnergyTests(unittest.TestCase):
    def test_empty_and_nonfinite_evidence_is_safe(self):
        candidates = torch.zeros(2, 14, 30, 3)
        rotation = torch.eye(3).expand(2, 30, 3, 3)
        tcp = torch.eye(4).expand(2, 4, 4)
        points = torch.full((2, 8, 3), float('nan'))
        risk = surface_risk(candidates, rotation, tcp, points, torch.ones(2, 8, dtype=torch.bool))
        self.assertTrue(torch.equal(risk, torch.zeros_like(risk)))

    def test_hand_extent_detects_surface_behind_tcp(self):
        candidates = torch.zeros(1, 14, 30, 3)
        rotation = torch.eye(3).expand(1, 30, 3, 3)
        tcp = torch.eye(4)[None]
        points = torch.tensor([[[0., 0., -.10]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        embodied = surface_risk(candidates, rotation, tcp, points, valid)
        point = surface_risk(candidates, rotation, tcp, points, valid, hand_extent=False)
        self.assertTrue((embodied>point).all())

    def test_rigid_coordinate_change_preserves_proximity(self):
        torch.manual_seed(1)
        candidates = torch.randn(2, 14, 30, 3)*.2
        points = torch.randn(2, 10, 3)*.2
        tcp = torch.eye(4).repeat(2, 1, 1)
        rotation = torch.eye(3).expand(2, 30, 3, 3)
        valid = torch.ones(2, 10, dtype=torch.bool)
        before = surface_risk(candidates, rotation, tcp, points, valid)
        shift = torch.tensor([.3, -.2, .1])
        tcp[:, :3, 3] += shift
        after = surface_risk(candidates+shift, rotation, tcp, points+shift, valid)
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)

    def test_trust_is_positive_bounded_and_trainable(self):
        model = RouteTrust()
        features = torch.randn(10, 8)
        coefficient = model(features)
        torch.testing.assert_close(coefficient, torch.full((10,), .08))
        coefficient.sum().backward()
        self.assertGreater(model.network[-1].weight.grad.norm(), 0)
        with torch.no_grad():
            model.network[-1].weight.fill_(100)
        coefficient = model(features)
        self.assertTrue(((coefficient>=.02)&(coefficient<=.12+1e-7)).all())

    def test_fixed_trust_reproduces_clearance_decision_exactly(self):
        torch.manual_seed(3)
        modules = [torch.nn.Identity() for _ in range(3)]
        reference = ClearancePolicy(*modules)
        candidate = GeometricEnergyPolicy(*modules, RouteTrust(), 'fixed_trust')
        tcp = torch.eye(4)[None]
        tcp[:, :3, 3] = torch.tensor([.5, -.2, .25])
        route = tcp[:, None, :3, 3]+torch.randn(1, 30, 3)*.02
        rotations = torch.eye(3).expand(1, 30, 3, 3)
        clouds = [(torch.rand(1, 400, 3)*.3+torch.tensor([.4, -.2, .1]),
                   torch.rand(1, 400)>.25) for _ in range(4)]
        for policy in (reference, candidate):
            policy.clouds = clouds
            policy.current_goal = torch.tensor([[.7, .2, .25]])
        inputs = (route, rotations, tcp, torch.tensor([False]), tcp)
        torch.testing.assert_close(reference.refine_waypoints(*inputs), candidate.refine_waypoints(*inputs), rtol=0, atol=0)
        self.assertEqual(reference.diagnostics[-1]['choice'], candidate.diagnostics[-1]['choice'])

    def test_episode_reset_clears_surfaces_and_diagnostics(self):
        policy = GeometricEnergyPolicy(*[torch.nn.Identity() for _ in range(3)], RouteTrust())
        policy.clouds = [(torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool))]
        policy.diagnostics = [dict(choice=1)]
        policy.reset_episode()
        self.assertEqual(policy.clouds, [])
        self.assertEqual(policy.diagnostics, [])


if __name__ == '__main__':
    unittest.main()
