"""Corrective sequence training must match the online compact observation buffer."""
import unittest

import torch

from tsn.models.compact_policy import CompactRouteHead
from tsn.training.corrective import compact_sequence, training_loss


class CompactCorrectiveTests(unittest.TestCase):
    def test_causal_windows_padding_and_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        b, t = 2, 7
        head = CompactRouteHead()
        batch = dict(tokens=torch.randn(b, t, 16, 768), geometry=torch.randn(b, t, 16, 6)*.1,
                     pose=torch.eye(4).expand(b, t, 4, 4).clone(),
                     tcp=torch.eye(4).expand(b, t, 4, 4).clone(), state=torch.randn(b, t, 16)*.1,
                     frame=torch.arange(t)[None].expand(b, -1)*15,
                     valid=torch.ones(b, t, dtype=torch.bool))
        batch['valid'][0, 4:] = False
        actual, _ = compact_sequence(head, batch)
        for episode in range(b):
            for step in range(int(batch['valid'][episode].sum())):
                start = max(0, step-3)
                features = [batch[k][episode:episode+1, start:step+1]
                            for k in ('tokens', 'geometry', 'pose')]
                ages = batch['frame'][episode:episode+1, step:step+1]-batch['frame'][episode:episode+1, start:step+1]
                expected, _ = head(*features, ages, torch.ones_like(ages, dtype=torch.bool),
                                   batch['state'][episode:episode+1, step],
                                   batch['tcp'][episode:episode+1, step])
                torch.testing.assert_close(actual[episode, step], expected[0], rtol=2e-5, atol=2e-7)
        changed = {k: v.clone() for k, v in batch.items()}
        changed['tokens'][:, 4:] += 100
        torch.testing.assert_close(compact_sequence(head, changed)[0][:, :4], actual[:, :4], rtol=0, atol=0)
        batch.update(supervision_valid=batch['valid'].clone(), action_weight=torch.ones(b, t),
                     waypoint=torch.zeros_like(actual))
        batch['supervision_valid'][1, 5:] = False
        loss = training_loss(actual, batch)
        masked = dict(batch, waypoint=batch['waypoint'].clone())
        masked['waypoint'][1, 5:] = 1e6
        torch.testing.assert_close(training_loss(actual, masked), loss)
        loss.backward()
        self.assertTrue(torch.isfinite(head.output[-1].weight.grad).all())
        self.assertGreater(float(head.output[-1].weight.grad.abs().sum()), 0.)
