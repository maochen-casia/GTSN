"""Triggering suppresses interventions without changing the underlying refiner."""
import unittest

import torch

from tsn.models.clearance_policy import ClearancePolicy
from tsn.models.compact_policy import CompactRouteHead
from tsn.models.triggered_clearance import TriggeredClearancePolicy


class TriggeredClearanceTests(unittest.TestCase):
    def prepare(self, constructor, **options):
        policy = constructor(torch.nn.Identity(), CompactRouteHead(), torch.nn.Identity(), penalty=.01, **options)
        policy.current_goal = torch.tensor([[.5, .3, .25]])
        policy.clouds = [(torch.tensor([[[.52, -.17, .15]]]), torch.tensor([[.03]]), torch.tensor([[True]]))]
        tcp = torch.eye(4)[None]
        tcp[:, :3, 3] = torch.tensor([.5, -.2, .25])
        path = tcp[:, None, :3, 3]+torch.tensor([0., .10, 0.])[None, None]*torch.linspace(1/30, 1, 30)[None, :, None]
        inputs = path, torch.eye(3).repeat(1, 30, 1, 1), tcp, torch.tensor([False]), tcp
        return policy, inputs

    def test_zero_trigger_exactly_matches_ungated_refiner(self):
        old, inputs = self.prepare(ClearancePolicy)
        new, _ = self.prepare(TriggeredClearancePolicy, trigger=0.)
        torch.testing.assert_close(old.refine_waypoints(*inputs), new.refine_waypoints(*inputs), rtol=0, atol=0)

    def test_high_trigger_preserves_the_proposal_and_memory(self):
        policy, inputs = self.prepare(TriggeredClearancePolicy, trigger=1.)
        actual = policy.refine_waypoints(*inputs)
        torch.testing.assert_close(actual, inputs[0], rtol=0, atol=0)
        self.assertEqual(len(policy.clouds), 1)
        self.assertFalse(policy.diagnostics[-1]['trigger_active'])
        self.assertGreater(policy.diagnostics[-1]['proposed_correction_m'], 0.)
        self.assertEqual(policy.diagnostics[-1]['correction_m'], 0.)
