import unittest
from unittest.mock import patch

import torch

from tsn.models.cartesian_policy import CartesianHead
from tsn.models.consensus_policy import ConsensusHead, ConsensusPolicy
from tsn.models.simplified_consensus import CompactPolicy, CompactRouteHead


class SimplificationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.original = CartesianHead('memory')
        self.args = (torch.randn(2, 4, 16, 768), torch.randn(2, 4, 16, 6),
                     torch.eye(4).expand(2, 4, 4, 4), torch.tensor([[45., 30, 15, 0]]).expand(2, -1),
                     torch.tensor([[False, False, False, True], [True, True, True, True]]),
                     torch.randn(2, 16), torch.eye(4).expand(2, 4, 4))

    def test_control_preserves_predictions_and_distribution(self):
        expected, expected_aux = self.original(*self.args)
        result, aux = CompactRouteHead(self.original)(*self.args)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        for key in ('mean', 'logvar'):
            torch.testing.assert_close(aux[key], expected_aux[key], rtol=0, atol=0)

    def test_current_does_not_use_history_or_attention_parameters(self):
        head = CompactRouteHead(self.original, temporal='current')
        expected = head(*self.args)[0]
        changed = [x.clone() for x in self.args]
        for x in changed[:4]:
            x[:, :-1] = 999
        torch.testing.assert_close(head(*changed)[0], expected, rtol=0, atol=0)
        self.assertFalse(hasattr(head, 'key'))

    def test_mean_ignores_masked_frames_and_deterministic_has_no_distribution(self):
        head = CompactRouteHead(self.original, uncertainty=False, temporal='mean')
        expected = head(*self.args)[0]
        changed = [x.clone() for x in self.args]
        for x in changed[:4]:
            x[0, :-1] = 999
        torch.testing.assert_close(head(*changed)[0], expected)
        self.assertFalse(hasattr(head, 'distribution'))
        expected.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_fixed_controller_matches_reference_without_temporal_buffer(self):
        modules = [torch.nn.Identity() for _ in range(3)]
        compact = CompactPolicy(*modules, history_length=1)
        original = ConsensusPolicy(*modules)
        state = torch.randn(1, 16)
        for step in (0, 15, 30, 0):
            if step == 0:
                compact.reset_episode()
                original.reset_episode()
            chunk = torch.randn(1, 30, 7)*.03
            compact.observe_step(step)
            original.observe_step(step)
            with patch('tsn.models.cartesian_policy.CartesianPolicy.forward', return_value=chunk):
                torch.testing.assert_close(compact(None, state, None, None),
                                           original(None, state, None, None), rtol=0, atol=0)
        self.assertFalse(hasattr(compact, 'previous'))
        self.assertEqual(compact.execution_horizon(30), 15)
        with self.assertRaises(ValueError):
            CompactPolicy(*modules, history_length=0)

    def test_single_head_needs_no_consensus_wrapper(self):
        head = CompactRouteHead(self.original)
        original, original_aux = ConsensusHead([head])(*self.args)
        actual, aux = head(*self.args)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        torch.testing.assert_close(aux['risk'], original_aux['risk'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
