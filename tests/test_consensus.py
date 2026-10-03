import unittest
from unittest.mock import patch
import torch
from tsn.models.consensus_policy import ConsensusHead, ConsensusPolicy


class ConstantHead(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, *args):
        return torch.full((2, 6, 3), self.value), {'risk': torch.zeros(2)}


class ConsensusTests(unittest.TestCase):
    def test_metric_disagreement_and_weighted_mean(self):
        head = ConsensusHead([ConstantHead(0.), ConstantHead(.02)], [1., 1.])
        result, aux = head()
        torch.testing.assert_close(result, torch.full((2, 6, 3), .01))
        torch.testing.assert_close(aux['risk'], torch.full((2,), .01*3**.5))
        singleton = ConsensusHead([ConstantHead(.02)])
        self.assertEqual(singleton()[1]['risk'].sum().item(), 0.)

    def test_invalid_weights(self):
        for weights in ([-1, 2], [0, 0], [1]):
            with self.assertRaises(ValueError):
                ConsensusHead([ConstantHead(0.), ConstantHead(1.)], weights)

    def test_reset_and_disagreement_schedule(self):
        policy = ConsensusPolicy(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity(), disagreement_threshold=.01)
        policy.previous = (15, torch.zeros(1, 30, 7))
        policy.last_risk = .02
        self.assertEqual(policy.execution_horizon(15), 5)
        policy.reset_episode()
        self.assertIsNone(policy.previous)
        self.assertEqual(policy.execution_horizon(15), 15)

    def test_temporal_average_aligns_absolute_targets_by_elapsed_steps(self):
        policy = ConsensusPolicy(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity(), temporal_weight=.25)
        state = torch.zeros(1, 16)
        initial = torch.arange(30).float()[None, :, None].expand(1, 30, 7)/100
        with patch('tsn.models.cartesian_policy.CartesianPolicy.forward', return_value=initial.clone()):
            torch.testing.assert_close(policy(None, state, None, None), initial)
        policy.observe_step(15)
        state[:, :7] = .1/torch.pi
        current = torch.ones(1, 30, 7)*.2
        with patch('tsn.models.cartesian_policy.CartesianPolicy.forward', return_value=current.clone()):
            actual = policy(None, state, None, None)
        expected = current.clone()
        expected[:, :15] = .75*.3 + .25*initial[:, 15:] - .1
        torch.testing.assert_close(actual, expected)
        policy.reset_episode()
        with patch('tsn.models.cartesian_policy.CartesianPolicy.forward', return_value=current.clone()):
            torch.testing.assert_close(policy(None, state, None, None), current)


if __name__ == '__main__':
    unittest.main()
