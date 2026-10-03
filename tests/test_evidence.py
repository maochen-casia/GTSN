"""Causality, geometric integration, and baseline preservation tests."""
import unittest

import torch

from tsn.models.compact_policy import CompactRouteHead
from tsn.models.evidence_policy import EvidenceRouteHead


class EvidenceTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(7)
        return (torch.randn(2, 4, 16, 768), torch.randn(2, 4, 16, 6)*.1,
                torch.eye(4).repeat(2, 4, 1, 1), torch.tensor([[45., 30, 15, 0]]).repeat(2, 1),
                torch.tensor([[False, False, True, True]]).repeat(2, 1),
                torch.randn(2, 16)*.1, torch.eye(4).repeat(2, 1, 1))

    def test_zero_initialization_preserves_baseline_exactly(self):
        inputs = self.inputs()
        base = CompactRouteHead().eval()
        head = EvidenceRouteHead(base).eval()
        with torch.no_grad():
            torch.testing.assert_close(head(*inputs)[0], base(*inputs)[0], rtol=0, atol=0)

    def test_masked_observations_cannot_change_route(self):
        inputs = list(self.inputs())
        head = EvidenceRouteHead().eval()
        torch.nn.init.normal_(head.output[-1].weight, std=.02)
        with torch.no_grad():
            before = head(*inputs)[0]
            for i in (0, 1, 2):
                inputs[i][:, :2] += 100
            after = head(*inputs)[0]
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_expected_kernel_matches_numerical_gaussian_average(self):
        torch.manual_seed(13)
        offset = torch.tensor([.08, -.10, .03])
        variance = torch.tensor([.001, .002, .0005])
        samples = offset + torch.randn(200000, 3)*variance.sqrt()
        measured = (-.5*samples.square().sum(-1)/.12**2).exp().mean()
        exact = EvidenceRouteHead.expected_proximity(offset, variance, .12).exp()
        self.assertLess(abs(float(measured-exact)), .003)

    def test_frozen_proposal_and_variance_receive_no_action_gradient(self):
        head = EvidenceRouteHead()
        inputs = self.inputs()
        torch.nn.init.normal_(head.output[-1].weight, std=.02)
        head(*inputs)[0].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in head.base.parameters()))
        self.assertTrue(torch.isfinite(head.output[-1].weight.grad).all())
        self.assertGreater(float(head.visual[1].weight.grad.abs().sum()), 0.)

    def test_current_ablation_retrieves_no_past_evidence(self):
        head = EvidenceRouteHead(variant='current')
        _, auxiliary = head(*self.inputs())
        self.assertEqual(float(auxiliary['past_mass'].abs().sum()), 0.)
        self.assertTrue(((auxiliary['support'] >= 0) & (auxiliary['support'] <= 1)).all())
