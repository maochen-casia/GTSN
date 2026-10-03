"""GTSN regression checks for causal history, uncertainty, and online reset."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from tsn.models.gtsn_policy import GTSNHead, GTSNPolicy, geometry_nll


class GTSNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_history_does_not_cross_episode_or_recovery(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_gtsn.py'
        spec = importlib.util.spec_from_file_location('gtsn_runner', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ep = np.array([0]*25 + [1]*25 + [0])
        frame = np.array(list(range(0, 50, 2))*2 + [40])
        source = np.array([0]*50 + [1])
        history, age, mask = module.build_history(ep, frame, source)
        self.assertTrue((ep[history] == ep[:, None]).all())
        self.assertEqual(mask[-1].sum(), 1)
        self.assertEqual(mask[25].sum(), 1)
        self.assertTrue((age[mask] >= 0).all())
        self.assertEqual(frame[history[24, -2]], 32)

    def test_padding_cannot_change_prediction_and_distribution_has_gradient(self):
        model = GTSNHead('memory')
        torch.nn.init.normal_(model.output[-1].weight, std=.01)
        tokens = torch.randn(2, 4, 16, 768)
        geometry = torch.randn(2, 4, 16, 6)
        poses = torch.eye(4).expand(2, 4, 4, 4).clone()
        ages = torch.tensor([[0, 0, 15, 0]]*2)
        mask = torch.tensor([[False, False, True, True]]*2)
        state, base = torch.randn(2, 16), torch.zeros(2, 30, 7)
        output, aux = model(tokens, geometry, poses, ages, mask, state, base)
        changed = tokens.clone()
        changed[:, :2] *= 1000
        altered, _ = model(changed, geometry, poses, ages, mask, state, base)
        torch.testing.assert_close(output, altered)
        loss = geometry_nll(aux, torch.zeros(2, 16, 3), torch.ones(2, 16, dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.distribution.weight.grad.abs().sum().item(), 0)

    def test_reset_and_adaptive_schedule(self):
        policy = GTSNPolicy(torch.nn.Identity(), GTSNHead(), 'adaptive', .1)
        policy.history = [1, 2]
        policy.last_risk = .2
        self.assertEqual(policy.execution_horizon(15), 5)
        policy.reset_episode()
        self.assertEqual(policy.history, [])
        self.assertEqual(policy.execution_horizon(15), 15)

    def test_autocast_uncertainty_is_exportable_float32(self):
        model = GTSNHead('memory')
        with torch.autocast('cpu', dtype=torch.bfloat16):
            _, aux = model(torch.randn(1, 1, 16, 768), torch.randn(1, 1, 16, 6),
                           torch.eye(4)[None, None], torch.zeros(1, 1),
                           torch.ones(1, 1, dtype=torch.bool), torch.randn(1, 16),
                           torch.zeros(1, 30, 7))
        self.assertEqual(aux['risk'].dtype, torch.float32)
        self.assertEqual(aux['logvar'].dtype, torch.float32)
        self.assertTrue(np.isfinite(aux['risk'].detach().numpy()).all())

    def test_streaming_memory_matches_explicit_history_and_stays_bounded(self):
        class Perception(torch.nn.Module):
            def forward(self, rgb, state, calibration, pose, return_features=False):
                value = rgb.float().mean((1, 2, 3))
                return (torch.zeros(len(rgb), 30, 7),
                        value[:, None, None].expand(-1, 16, 768),
                        value[:, None, None].expand(-1, 16, 6) / 255)
        head = GTSNHead('memory').eval()
        torch.nn.init.normal_(head.output[-1].weight, std=.01)
        online = GTSNPolicy(Perception(), head).eval()
        state, K, pose = torch.randn(1, 16), torch.eye(3)[None], torch.eye(4)[None]
        for frame in range(5):
            online.observe_step(15*frame)
            output = online(torch.full((1, 2, 2, 3), frame, dtype=torch.uint8), state, K, pose)
        self.assertEqual(len(online.history), 4)
        h = online.history
        expected, _ = head(torch.stack([x[0] for x in h], 1), torch.stack([x[1] for x in h], 1),
                           torch.stack([x[2] for x in h], 1), torch.tensor([[45, 30, 15, 0]]),
                           torch.ones(1, 4, dtype=torch.bool), state, torch.zeros(1, 30, 7))
        torch.testing.assert_close(output, expected)

    def test_action_gradient_cannot_shrink_variance(self):
        head = GTSNHead('uncertainty')
        torch.nn.init.normal_(head.output[-1].weight, std=.01)
        action, _ = head(torch.randn(1, 1, 16, 768), torch.randn(1, 1, 16, 6),
                         torch.eye(4)[None, None], torch.zeros(1, 1),
                         torch.ones(1, 1, dtype=torch.bool), torch.randn(1, 16),
                         torch.zeros(1, 30, 7))
        action.square().sum().backward()
        torch.testing.assert_close(head.distribution.weight.grad[3:],
                                   torch.zeros_like(head.distribution.weight.grad[3:]))


if __name__ == '__main__':
    unittest.main()
