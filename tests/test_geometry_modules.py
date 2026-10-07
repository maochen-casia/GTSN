"""Persistence, physical extent and uncertainty margins of the retained model."""
import math
import unittest

import torch

from tsn.models.c1_memory import PersistentGeometry
from tsn.models.c2_embodiment import EmbodimentGeometry, box_signed_distance
from tsn.models.c3_clearance import UncertaintyClearance, route_candidates


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_old_geometry_survives_the_four_observation_window(self):
        memory = PersistentGeometry()
        original = torch.tensor([[.4, .2, .2]])
        memory.update(original, torch.tensor([True]), torch.tensor([.08]), 0, torch.zeros(3), torch.ones(3))
        for step in (15, 30, 45, 60, 75):
            memory.update(torch.empty(0, 3), torch.empty(0, dtype=torch.bool), torch.empty(0), step, torch.zeros(3), torch.ones(3))
        points, radius = memory.query()
        torch.testing.assert_close(points, original)
        torch.testing.assert_close(radius, torch.tensor([.08]))
        memory.reset()
        self.assertIsNone(memory.points); self.assertEqual(len(memory.recent), 0)

    def test_anchor_and_creation_uncertainty_remain_fixed_under_reobservation(self):
        memory = PersistentGeometry()
        p = torch.tensor([[.4, .2, .2]])
        memory.update(p.repeat(8, 1), torch.ones(8, dtype=torch.bool), torch.full((8,), .08), 0, torch.zeros(3), torch.ones(3))
        memory.update(p+.001, torch.ones(1, dtype=torch.bool), torch.tensor([.15]), 15, torch.zeros(3), torch.ones(3))
        torch.testing.assert_close(memory.points, p)
        torch.testing.assert_close(memory.radius, torch.tensor([.08]))
        self.assertEqual(float(memory.support[0]), 2.)
        self.assertGreater(float(memory.scatter[0]), 0.)
        with self.assertRaises(ValueError):memory.update(p, torch.ones(1, dtype=torch.bool), torch.tensor([.1]), 15, torch.zeros(3), torch.ones(3))

    def test_capacity_and_actual_observation_anchors(self):
        memory = PersistentGeometry(capacity=3)
        points = torch.tensor([[.2, .2, .2], [.3, .2, .2], [.4, .2, .2], [.5, .2, .2], [.6, .2, .2]])
        memory.update(points, torch.ones(5, dtype=torch.bool), torch.full((5,), .05), 0, torch.zeros(3), torch.ones(3))
        self.assertEqual(len(memory.points), 3)
        self.assertTrue(torch.isin(memory.points[:, 0], points[:, 0]).all())

    def test_box_distance_is_metric_inside_and_outside(self):
        points = torch.tensor([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.]])
        distance = box_signed_distance(points, torch.zeros(1, 3), torch.eye(3)[None], torch.ones(1, 3))[:, 0]
        torch.testing.assert_close(distance, torch.tensor([-1., 1., math.sqrt(2)]))

    def test_physical_regions_capture_width_wrist_and_measured_fingers(self):
        body = EmbodimentGeometry()
        points = torch.tensor([[0., .08, -.08], [0., 0., -.14], [0., .04, .005]])
        closed = body.signed_distances(points, torch.zeros(2))
        opened = body.signed_distances(points, torch.full((2,), .02))
        self.assertEqual(closed.shape, (3, 6))
        self.assertLess(float(closed[0, 1]), 0.)
        self.assertLess(float(closed[1, 4]), 0.)
        self.assertFalse(torch.equal(closed[:, 2:4], opened[:, 2:4]))
        self.assertTrue(body.self_mask(points[:2], torch.eye(4), torch.zeros(2)).all())

    def test_contact_response_is_pose_covariant_and_padding_monotone(self):
        body = EmbodimentGeometry(); tcp = torch.eye(4)
        candidates = torch.tensor([.5, .2, .3]).expand(2, 30, 3).clone()
        candidates[1, :, 0] += .04
        rotations = torch.eye(3).expand(30, 3, 3)
        points = torch.tensor([[.55, .2, .25], [.45, .25, .2]])
        fingers = torch.full((2,), .02)
        original = body.contact_risk(candidates, rotations, tcp, points, torch.zeros(2), fingers)
        inflated = body.contact_risk(candidates, rotations, tcp, points, torch.full((2,), .03), fingers)
        self.assertTrue((inflated >= original).all())
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        shift = torch.tensor([.1, -.2, .05]); pose = tcp.clone()
        pose[:3, :3] = rotation; pose[:3, 3] = shift
        transformed = body.contact_risk(candidates@rotation.T+shift, rotation@rotations,
            pose, points@rotation.T+shift, torch.zeros(2), fingers)
        torch.testing.assert_close(original, transformed, rtol=1e-5, atol=1e-6)

    def test_empty_and_nonfinite_scene_have_zero_risk(self):
        body = EmbodimentGeometry()
        args = (torch.zeros(14, 30, 3), torch.eye(3).expand(30, 3, 3), torch.eye(4))
        for points, padding in ((torch.empty(0, 3), torch.empty(0)), (torch.full((1, 3), torch.nan), torch.tensor([torch.nan]))):
            torch.testing.assert_close(body.contact_risk(*args, points, padding, torch.zeros(2)), torch.zeros(14))

    def test_uncertainty_bounds_padding_and_terminal_fade(self):
        clearance = UncertaintyClearance()
        points = torch.tensor([.5, .2, .3]).expand(2, 400, 3)
        radius = clearance.predict_error(torch.zeros(2, 16, 64), points, torch.eye(4).expand(2, 4, 4), torch.eye(4).expand(2, 4, 4))
        self.assertTrue(((radius >= .005) & (radius <= .2)).all())
        torch.testing.assert_close(clearance.padding(torch.tensor([.005, .015, .1, .2])), torch.tensor([0., 0., .03, .03]))
        waypoints = torch.zeros(1, 30, 3)
        candidates, _ = route_candidates(waypoints, torch.eye(4)[None], torch.tensor([[.01, 0., 0.]]))
        torch.testing.assert_close(candidates, waypoints[:, None].expand(-1, 14, -1, -1))
