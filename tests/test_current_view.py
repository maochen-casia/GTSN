"""Causal current-view geometry, four-frame carry, and residual-only training."""
import unittest

import torch

from tsn.models.current_view_policy import CurrentViewHead
from tsn.training.corrective import training_loss


class CurrentViewTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def inputs(self):
        batch, time = 2, 7
        tokens = torch.randn(batch, time, 16, 768)
        geometry = torch.randn(batch, time, 16, 6) * .1
        pose = torch.eye(4).expand(batch, time, 4, 4).clone()
        times = torch.arange(time).float()[None].expand(batch, -1) * 15
        valid = torch.ones(batch, time, dtype=torch.bool)
        state = torch.randn(batch, time, 16) * .1
        tcp = pose.clone()
        points = torch.rand(batch, time, 400, 3) * torch.tensor([.8, .9, .5]) + torch.tensor([.15, -.45, .08])
        return [tokens, geometry, pose, times, valid, state, tcp, points]

    def test_stream_chunk_reset_and_padding(self):
        head = CurrentViewHead()
        torch.nn.init.normal_(head.history_output[-1].weight, std=.02)
        args = self.inputs()
        prediction, _, final = head.sequence(*args)
        output, memory = [], None
        for t in range(7):
            value, _, memory = head.stream(args[0][:, t], args[1][:, t], args[2][:, t],
                torch.full((2,), 15.), args[5][:, t], args[6][:, t], args[7][:, t], memory)
            output.append(value)
        torch.testing.assert_close(torch.stack(output, 1), prediction, rtol=2e-4, atol=2e-6)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].shape, (2, 4, 258))
        _, _, prefix = head.sequence(*[x[:, :3] for x in args])
        suffix = [x[:, 3:] for x in args]
        suffix[3] = suffix[3]-args[3][:, 2:3]
        torch.testing.assert_close(head.sequence(*suffix, initial=prefix)[0], prediction[:, 3:], rtol=2e-4, atol=2e-6)
        resets = torch.zeros(2, 7, dtype=torch.bool); resets[:, 3] = True
        torch.testing.assert_close(head.sequence(*args, resets=resets)[0][:, 3:],
                                   head.sequence(*[x[:, 3:] for x in args])[0], rtol=2e-4, atol=2e-6)
        args[4][0, 4:] = False
        _, _, padded = head.sequence(*args)
        _, _, short = head.sequence(*[x[:1, :4] for x in args])
        torch.testing.assert_close(padded[0][:1], short[0], rtol=2e-5, atol=2e-5)

    def test_future_is_inert_and_old_point_clouds_are_not_retained(self):
        head = CurrentViewHead()
        args = self.inputs(); expected = head.sequence(*args)[0]
        changed = [x.clone() for x in args]
        changed[0][:, 4:] += 10
        torch.testing.assert_close(head.sequence(*changed)[0][:, :4], expected[:, :4], rtol=0, atol=0)
        changed = [x.clone() for x in args]
        changed[7][:, 0] = float('nan')
        torch.testing.assert_close(head.sequence(*changed)[0][:, 1:], expected[:, 1:], rtol=0, atol=0)

    def test_empty_points_are_finite_and_disable_correction(self):
        head = CurrentViewHead()
        torch.nn.init.normal_(head.history_output[-1].weight, std=.02)
        args = self.inputs(); args[7][:] = float('nan')
        prediction, aux, _ = head.sequence(*args)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertEqual(float(aux['correction'].abs().sum()), 0.)

    def test_only_residual_trains_and_roundtrip_is_exact(self):
        head = CurrentViewHead()
        for name, parameter in head.named_parameters():
            parameter.requires_grad_(name.startswith('history_output.'))
        frozen = {k: v.clone() for k, v in head.state_dict().items() if not k.startswith('history_output.')}
        args = self.inputs()
        prediction, aux, _ = head.sequence(*args)
        torch.testing.assert_close(prediction, aux['current_prediction'], rtol=0, atol=0)
        batch = dict(valid=args[4], supervision_valid=args[4], waypoint=torch.ones_like(prediction)*.1)
        optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=.01)
        training_loss(prediction, batch).backward(); optimizer.step()
        self.assertGreater(float(head.history_output[-1].weight.grad.abs().sum()), 0.)
        for name, value in frozen.items():
            torch.testing.assert_close(head.state_dict()[name], value, rtol=0, atol=0)
        loaded = CurrentViewHead.from_checkpoint(dict(architecture='current_view', head_options=head.options(), head=head.state_dict()))
        torch.testing.assert_close(loaded.sequence(*args)[0], head.sequence(*args)[0], rtol=0, atol=0)

    def test_legacy_persistent_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'current-view'):
            CurrentViewHead.from_checkpoint(dict(memory_options=dict(always_reset_persistent=False)))
