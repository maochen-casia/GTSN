"""Verify baseline algorithm semantics and deployment isolation."""
import copy
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from tsn.baselines.carp import ActionTokenizer, NextScalePolicy
from tsn.baselines.observations import point_cloud, rgb_image, observation_state
from tsn.baselines.policies import BaselinePolicy, consistency_loss
from tsn.common.config import output_path


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.config = json.loads((Path(__file__).resolve().parents[1]/'configs/cam_var.json').read_text())
        self.batch = {'rgb': torch.randint(256, (2, 2, 84, 112, 3), dtype=torch.uint8),
            'points': torch.randn(2, 2, 512, 3), 'state': torch.randn(2, 2, 34),
            'target': torch.randn(2, 30, 7)*.1}

    def policy(self, method):
        config = copy.deepcopy(self.config)
        config['baseline'] = {'method': method, 'inference_steps': 5}
        return BaselinePolicy(config)

    def test_all_methods_optimize_and_generate_complete_chunks(self):
        for method in ('diffusion_policy', 'dp3', 'flowpolicy', 'carp'):
            with self.subTest(method=method):
                model = self.policy(method)
                loss = model.loss(self.batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                model.eval()
                with torch.no_grad():
                    chunk = model.sample(self.batch)
                self.assertEqual(chunk.shape, (2, 30, 7))
                self.assertTrue(torch.isfinite(chunk).all())

    def test_rgb_policies_ignore_depth_and_future_labels_at_inference(self):
        for method in ('diffusion_policy', 'carp'):
            model = self.policy(method).eval()
            changed = {**self.batch, 'points': self.batch['points']*100,
                       'target': torch.full_like(self.batch['target'], float('nan'))}
            with torch.no_grad():
                a = model.sample(self.batch, torch.Generator().manual_seed(1))
                b = model.sample(changed, torch.Generator().manual_seed(1))
            self.assertTrue(torch.equal(a, b))

    def test_flow_consistency_vanishes_for_correct_straight_flow(self):
        target, noise = torch.randn(3, 30, 7), torch.randn(3, 30, 7)
        class ExactVelocity(nn.Module):
            def forward(self, x, t, condition):
                return target-noise
        loss = consistency_loss(ExactVelocity(), target, noise[:, 0], noise, torch.tensor([.2, .7, .995]))
        self.assertLess(float(loss), 1e-12)
        class WrongVelocity(nn.Module):
            def forward(self, x, t, condition):
                return torch.zeros_like(x)
        self.assertGreater(float(consistency_loss(WrongVelocity(), target, noise[:, 0], noise,
                                                  torch.tensor([.2, .7, .995]))), 1e-5)

    def test_carp_scale_mask_prevents_teacher_future_leakage(self):
        head = NextScalePolicy(4, ActionTokenizer(), width=32, layers=1).eval()
        condition = torch.randn(2, 4)
        contexts = [torch.randn(2, size, 56) for size in head.tokenizer.scales]
        changed = [x.clone() for x in contexts]
        changed[-1].add_(100)
        with torch.no_grad():
            first = head.logits(contexts, condition)
            second = head.logits(changed, condition)
        self.assertTrue(torch.allclose(first[:, :49], second[:, :49], atol=1e-6))

    def test_calibrated_point_cloud_is_finite_and_empty_views_supported(self):
        K = torch.tensor([[[100., 0, 2.], [0, 100., 2.], [0, 0, 1.]]])
        pose = torch.eye(4)[None]
        pose[:, 0, 3] = .5
        cloud = point_cloud(torch.ones(1, 5, 5)*.3, K, pose)
        self.assertEqual(cloud.shape, (1, 512, 3))
        self.assertTrue(torch.isfinite(cloud).all())
        self.assertTrue(torch.allclose(cloud[..., 2], torch.full((1, 512), .3)))
        self.assertEqual(float(point_cloud(torch.full((1, 5, 5), float('nan')), K, pose).abs().sum()), 0.)

    def test_live_history_resets_and_rgb_accepts_native_resolution(self):
        model = self.policy('diffusion_policy').eval()
        rgb = torch.zeros(1, 96, 128, 3, dtype=torch.uint8)
        self.assertEqual(rgb_image(rgb).shape, (1, 84, 112, 3))
        model.history.append((torch.ones(1), torch.ones(1)))
        model.reset_episode()
        self.assertEqual(len(model.history), 0)
        self.assertEqual(model.generator.initial_seed(), self.config['train']['seed'])

    def test_real_experiment_storage_is_accepted(self):
        path = Path('/home/datasets_v2/chenmao/experiments/baseline-test/training')
        self.assertEqual(output_path(path), path)
        with self.assertRaises(ValueError):
            output_path('/home/datasets_v2/chenmao/tsn-1k-var/overwrite')
