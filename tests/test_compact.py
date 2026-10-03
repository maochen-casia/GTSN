"""Regression coverage for causal memory, checkpoint compatibility, and training."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tsn.common.config import create_output
from tsn.data.history import build_history
from tsn.models.compact_policy import CompactPolicy, CompactRouteHead, compact_state
from tsn.models.kinematics import PandaKinematics
from tsn.training.losses import geometry_nll
from tsn.training.runner import FeatureCache, fit_head, sampling_weights


class CompactTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def inputs(self):
        return [torch.randn(2, 4, 16, 768), torch.randn(2, 4, 16, 6),
                torch.eye(4).expand(2, 4, 4, 4).clone(), torch.tensor([[45., 30, 15, 0]]).expand(2, -1),
                torch.tensor([[False, False, False, True], [True, True, True, True]]),
                torch.randn(2, 16), torch.eye(4).expand(2, 4, 4)]

    def test_masked_frames_are_inert_and_all_parameters_train(self):
        head = CompactRouteHead()
        args = self.inputs()
        expected, auxiliary = head(*args)
        changed = [x.clone() for x in args]
        for value in changed[:4]:
            value[0, :-1] = 999
        torch.testing.assert_close(head(*changed)[0], expected, rtol=0, atol=0)
        loss = expected.square().mean() + .001 * geometry_nll(
            auxiliary, torch.zeros(2, 16, 3), torch.ones(2, 16, dtype=torch.bool))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertGreater(head.distribution.weight.grad.abs().sum().item(), 0)

    def test_no_observed_geometry_has_zero_auxiliary_loss(self):
        _, auxiliary = CompactRouteHead()(*self.inputs())
        loss = geometry_nll(auxiliary, torch.zeros(2, 16, 3), torch.zeros(2, 16, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)

    def test_original_export_unwraps_only_one_unit_weight_head(self):
        state = CompactRouteHead().state_dict()
        old = {'weights': torch.ones(1), **{'heads.0.' + k: v for k, v in state.items()}}
        checkpoint = dict(architecture='compact_consensus', variant='single_memory', backbone={}, head=old)
        _, actual = compact_state(checkpoint)
        CompactRouteHead().load_state_dict(actual, strict=True)
        checkpoint['variant'] = 'control'
        with self.assertRaises(ValueError):
            compact_state(checkpoint)
        checkpoint['variant'] = 'single_memory'
        checkpoint['head']['weights'] = torch.tensor([.5])
        with self.assertRaises(ValueError):
            compact_state(checkpoint)

    def test_baseline_initialization_extracts_memory_only(self):
        state = CompactRouteHead().state_dict()
        model = {'heads.memory.' + k: v for k, v in state.items()}
        model.update({'backbone.example': torch.ones(1), 'heads.state.unused': torch.zeros(1)})
        with self.assertRaises(ValueError):
            compact_state({'model': model})
        backbone, head = compact_state({'model': model}, allow_training_source=True)
        self.assertEqual(set(backbone), {'example'})
        self.assertEqual(set(head), set(state))

    def test_history_never_crosses_episode_or_recovery_boundaries(self):
        episodes = np.array([0, 0, 0, 0, 1, 1, 0])
        frames = np.array([0, 14, 30, 46, 0, 30, 90])
        sources = np.array([0, 0, 0, 0, 0, 0, 1])
        history, ages, mask = build_history(episodes, frames, sources)
        self.assertTrue(np.all(episodes[history] == episodes[:, None]))
        self.assertTrue(np.all(frames[history] <= frames[:, None]))
        self.assertEqual(mask[-1].sum(), 1)
        self.assertEqual(history[3, -2], 2)
        self.assertEqual(ages[3, -2], 16)

    def test_sampling_keeps_source_and_route_ratios(self):
        routes = np.tile(np.array([0, 1, 1, 2, 2]), 2)
        sources = np.repeat([0, 1], 5)
        weights = sampling_weights(routes, sources).numpy()
        for source, fraction in enumerate((.65, .35)):
            for route, probability in enumerate((.2, .4, .4)):
                self.assertAlmostEqual(weights[(sources == source) & (routes == route)].sum(), fraction * probability)
        with self.assertRaises(ValueError):
            sampling_weights(routes[:5], sources[:5])

    def test_ik_reaches_nearby_target_with_joint_limits(self):
        kin = PandaKinematics()
        q = torch.tensor([[0., .4, 0., -1.96, 0., 2.35, .78]])
        pose = kin(q)
        target = pose[:, :3, 3] + torch.tensor([[.02, -.03, .015]])
        result = kin.inverse(q, target, pose[:, :3, :3])
        self.assertLess((kin(result)[:, :3, 3] - target).norm().item(), 1e-4)
        self.assertTrue((result >= kin.limits[:, 0]).all())
        self.assertTrue((result <= kin.limits[:, 1]).all())

    def test_controller_memory_resets_and_servo_reaches_goal(self):
        class Perception(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return torch.zeros(1, 30, 7), torch.zeros(1, 16, 768), torch.zeros(1, 16, 6)
        kin = PandaKinematics()
        q = torch.tensor([[0., .4, 0., -1.96, 0., 2.35, .78]])
        goal = kin(q)[:, :3, 3] + torch.tensor([[.02, -.01, .015]])
        state = torch.zeros(1, 16)
        state[:, :7] = q / torch.pi
        state[:, 9:12] = (goal - torch.tensor([.65, 0, .22])) / torch.tensor([.55, .55, .5])
        model = CompactPolicy(Perception(), CompactRouteHead(), kin).eval()
        with torch.inference_mode():
            for step in range(0, 90, 15):
                model.observe_step(step)
                chunk = model(torch.zeros(1, 1, 1, 3, dtype=torch.uint8), state, torch.eye(3)[None], torch.eye(4)[None])
                self.assertLessEqual(len(model.history), 4)
        self.assertEqual(model.execution_horizon(30), 15)
        self.assertLess((kin(q + chunk[:, -1])[:, :3, 3] - goal).norm().item(), 1e-4)
        model.reset_episode()
        self.assertEqual(model.history, [])
        self.assertEqual(model.step, 0)

    def test_one_epoch_cache_training_selects_and_reloads_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            count = 6
            values = dict(tokens=np.zeros((count, 16, 768), np.float16),
                          geometry=np.zeros((count, 16, 6), np.float32),
                          pose=np.tile(np.eye(4, dtype=np.float32), (count, 1, 1)),
                          tcp=np.tile(np.eye(4, dtype=np.float32), (count, 1, 1)),
                          state=np.zeros((count, 16), np.float32),
                          waypoint=np.zeros((count, 6, 3), np.float32),
                          teacher=np.zeros((count, 16, 3), np.float32),
                          geometry_valid=np.ones((count, 16), bool),
                          route=np.tile(np.arange(3), 2), source=np.repeat([0, 1], 3),
                          history=np.repeat(np.arange(count)[:, None], 4, 1),
                          ages=np.zeros((count, 4), np.float32), mask=np.ones((count, 4), bool))
            for name, value in values.items():
                np.save(root / (name + '.npy'), value)
            data = FeatureCache(root, torch.device('cpu'))
            head = CompactRouteHead()
            summary = fit_head(head, data, data, dict(epochs=1, learning_rate=3e-5,
                               weight_decay=1e-4, batch_size=3, seed=31), root, 6)
            self.assertEqual(summary['completed_epochs'], 1)
            saved = torch.load(root / 'head.pt', weights_only=True)
            for key, value in head.state_dict().items():
                torch.testing.assert_close(value, saved[key], rtol=0, atol=0)

    def test_output_refuses_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_output(root)
            (root / 'keep').write_text('untouched')
            with self.assertRaises(FileExistsError):
                create_output(root)
            self.assertEqual((root / 'keep').read_text(), 'untouched')
