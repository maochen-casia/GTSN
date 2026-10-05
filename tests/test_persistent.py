"""Causal correctness and scan/stream equivalence for persistent memory."""
import unittest

import torch
import numpy as np

from tsn.models.compact_policy import CompactRouteHead
from tsn.models.persistent_policy import PersistentSceneHead, affine_scan, compact_sequence
from tsn.models.clearance_policy import ClearancePolicy
from tsn.models.kinematics import PandaKinematics
from tsn.training.persistent import training_loss, observation_sequences


class PersistentTests(unittest.TestCase):
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

    def test_scan_matches_recurrence_values_and_gradients_with_reset(self):
        a = torch.rand(2, 9, 3, 1, requires_grad=True)
        b = torch.randn(2, 9, 3, 5, requires_grad=True)
        reset = torch.zeros_like(a)
        reset[:, 4] = 1
        a = a * (1-reset)
        initial = torch.randn(2, 3, 5)
        expected, running = [], initial
        for time in range(9):
            running = a[:, time] * running + b[:, time]
            expected.append(running)
        expected = torch.stack(expected, 1)
        actual = affine_scan(a, b, initial)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        grad_actual = torch.autograd.grad(actual.square().sum(), (a, b), retain_graph=True)
        grad_expected = torch.autograd.grad(expected.square().sum(), (a, b))
        for x, y in zip(grad_actual, grad_expected):
            torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)

    def test_sequence_matches_streaming_and_chunk_carry(self):
        for detail in (False, True):
            head = PersistentSceneHead(use_points=detail).eval()
            torch.nn.init.normal_(head.memory_residual.weight, std=.01)
            args = self.inputs()
            expected, _, final = head.sequence(*args)
            memory, output = None, []
            for time in range(args[0].shape[1]):
                prediction, _, memory = head.stream(args[0][:, time], args[1][:, time], args[2][:, time],
                    torch.full((2,), 15.), args[5][:, time], args[6][:, time], args[7][:, time], memory)
                output.append(prediction)
            torch.testing.assert_close(torch.stack(output, 1), expected, rtol=2e-4, atol=2e-6)
            for x, y in zip(final, memory):
                torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)
            prefix = [x[:, :3] for x in args]
            _, _, memory = head.sequence(*prefix)
            suffix = [x[:, 3:] for x in args]
            suffix[3] = suffix[3] - args[3][:, 2:3]
            actual, _, _ = head.sequence(*suffix, initial=memory)
            torch.testing.assert_close(actual, expected[:, 3:], rtol=2e-4, atol=2e-6)

    def test_future_and_padding_are_inert(self):
        head = PersistentSceneHead()
        torch.nn.init.normal_(head.memory_residual.weight, std=.01)
        args = self.inputs()
        expected, _, final = head.sequence(*args)
        changed = [x.clone() for x in args]
        changed[0][:, 4:] += 100
        changed[7][:, 4:] *= 2
        torch.testing.assert_close(head.sequence(*changed)[0][:, :4], expected[:, :4], rtol=0, atol=0)
        args[4][0, 4:] = False
        _, _, final = head.sequence(*args)
        _, _, short = head.sequence(*[x[:1, :4] for x in args])
        for x, y in zip(final, short):
            torch.testing.assert_close(x[:1], y, rtol=2e-5, atol=2e-5)

    def test_initialization_matches_current_baseline_and_finite_gradients(self):
        compact = CompactRouteHead()
        head = PersistentSceneHead()
        head.initialize(compact)
        args = self.inputs()
        args[4][0, 4:] = False
        current, _ = compact_sequence(compact, *args[:7], current_only=True)
        actual, aux, _ = head.sequence(*args)
        torch.testing.assert_close(actual[args[4]], current[args[4]], rtol=2e-5, atol=2e-6)
        torch.nn.init.normal_(head.memory_residual.weight, std=.01)
        actual, aux, _ = head.sequence(*args)
        loss = actual.square().mean() + aux['point_mean'].square().mean()
        loss.backward()
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_static_geometry_survives_more_than_four_observations_and_reset(self):
        head = PersistentSceneHead()
        args = self.inputs()
        args[7][:, 1:] = 3  # All later geometry is outside the workspace.
        _, aux, _ = head.sequence(*args)
        torch.testing.assert_close(aux['memory_points'][:, 0], aux['memory_points'][:, -1])
        self.assertTrue(aux['memory_valid'][:, -1].any())
        resets = torch.zeros(2, 7, dtype=torch.bool)
        resets[:, 5] = True
        _, aux, _ = head.sequence(*args, resets=resets)
        self.assertFalse(aux['memory_valid'][:, -1].any())

    def test_four_frame_baseline_matches_original_head(self):
        args = self.inputs()
        head = CompactRouteHead()
        prediction, _ = compact_sequence(head, *args[:7])
        for time in range(7):
            start = max(0, time-3)
            expected, _ = head(args[0][:, start:time+1], args[1][:, start:time+1],
                args[2][:, start:time+1], args[3][:, time:time+1]-args[3][:, start:time+1],
                args[4][:, start:time+1], args[5][:, time], args[6][:, time])
            torch.testing.assert_close(prediction[:, time], expected, rtol=2e-5, atol=2e-6)

    def test_controller_uses_fixed_memory_and_resets(self):
        class Perception(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return (torch.zeros(1, 30, 7), torch.zeros(1, 16, 768),
                        torch.zeros(1, 16, 6), torch.full((1, 6, 20, 20), .1))
        kin = PandaKinematics()
        q = torch.tensor([[0., .4, 0., -1.96, 0., 2.35, .78]])
        tcp = kin(q)
        state = torch.zeros(1, 16)
        state[:, :7] = q/torch.pi
        state[:, 9:12] = (tcp[:, :3, 3]+torch.tensor([[.01, .01, 0]])-torch.tensor([.65, 0, .22]))/torch.tensor([.55, .55, .50])
        model = ClearancePolicy(Perception(), PersistentSceneHead(), kin).eval()
        sizes = []
        with torch.inference_mode():
            for step in range(0, 120, 15):
                model.observe_step(step)
                result = model(torch.zeros(1, 1, 1, 3, dtype=torch.uint8), state, torch.eye(3)[None], torch.eye(4)[None])
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(model.history, [])
                sizes.append(sum(x.numel() for x in model.scene_memory))
        self.assertEqual(len(set(sizes)), 1)
        model.reset_episode()
        self.assertIsNone(model.scene_memory)
        self.assertIsNone(model.memory_step)
        self.assertIsNone(model.current_points)

    def test_complete_training_loss_under_autocast(self):
        head = PersistentSceneHead()
        args = self.inputs()
        batch = dict(zip(('tokens', 'geometry', 'pose', 'frame', 'valid', 'state', 'tcp', 'points'), args))
        batch.update(waypoint=torch.zeros(2, 7, 6, 3), teacher=torch.zeros(2, 7, 16, 3),
                     geometry_valid=torch.ones(2, 7, 16, dtype=torch.bool),
                     point_teacher=args[7].clone(), point_valid=torch.ones(2, 7, 400, dtype=torch.bool))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            prediction, aux, _ = head.sequence(*args)
            loss = training_loss(head, prediction, aux, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_sequences_use_real_cadence_and_isolate_recoveries(self):
        arrays = dict(episode=np.array([0]*20+[1]*10+[0, 0]),
                      frame=np.array(list(range(0, 40, 2))+list(range(10))+[0, 0]),
                      source=np.array([0]*30+[1, 1]))
        sequences = observation_sequences(arrays)
        np.testing.assert_array_equal(arrays['frame'][sequences[0]], [0, 14, 30])
        np.testing.assert_array_equal(arrays['frame'][sequences[1]], [0])
        self.assertEqual(sequences[2].tolist(), [30])
        self.assertEqual(sequences[3].tolist(), [31])

    def test_sparse_geometry_requires_finer_points(self):
        head = PersistentSceneHead()
        with self.assertRaisesRegex(ValueError, 'unpooled metric points'):
            head.sequence(*self.inputs()[:7])
