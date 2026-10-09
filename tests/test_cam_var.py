"""Contribution removals must apply consistently to training and live control."""
import copy
import unittest
from unittest.mock import patch

import torch

from test_main_model import fixture, TinyPerception
from tsn.models.policy import NavigationPolicy
from tsn.models.c2_embodiment import TCPGeometry, EmbodimentGeometry
from tsn.training.runner import encode_observations, training_loss
from tsn.data.history import build_history
import numpy as np


def batch_fixture(qpos, goal, pose, K, rgb, slots):
    return dict(qpos=qpos, goal_pose=goal, K=K, T_B_C=pose,
        target=torch.zeros(1, 30, 7), future_ee=goal[:, None, :3].expand(-1, 30, -1),
        valid_future=torch.ones(1, 30, dtype=torch.bool), depth=torch.ones(1, 24, 32),
        frame_index=torch.tensor([45]), history_rgb=rgb[:, None].expand(-1, slots, -1, -1, -1),
        history_qpos=qpos[:, None].expand(-1, slots, -1),
        history_goal_pose=goal[:, None].expand(-1, slots, -1),
        history_K=K[:, None].expand(-1, slots, -1, -1),
        history_T_B_C=pose[:, None].expand(-1, slots, -1, -1),
        history_ages=torch.arange(slots-1, -1, -1)[None].float()*15,
        history_mask=torch.ones(1, slots, dtype=torch.bool))


class ContributionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_without_c1_uses_one_image_and_forgets_previous_observations(self):
        config, _, maps, qpos, goal, state, pose, K, rgb = fixture()
        config['model']['contributions'] = dict(c1=False, c2=True, c3=True)
        model = NavigationPolicy(config['model'], perception=TinyPerception())
        batch = batch_fixture(qpos, goal, pose, K, rgb, 1)
        with patch.object(model.perception, 'forward', wraps=model.perception.forward) as perception:
            features = encode_observations(model, maps, batch)
            self.assertEqual(perception.call_count, 1)
            self.assertEqual(features['tokens'].shape[1], 1)
        history, ages, mask = build_history(np.array([0, 0, 0]), np.array([0, 15, 30]),
                                           np.zeros(3), length=1)
        np.testing.assert_array_equal(history[:, 0], np.arange(3))
        self.assertTrue(mask.all()); self.assertFalse(ages.any())
        model.eval()
        with torch.inference_mode():
            model(rgb*0, state, K, pose)
            model.observe_step(15)
            actual = model(rgb, state, K, pose)
            self.assertEqual(len(model.history), 1)
            model.reset_episode()
            model.observe_step(15)
            expected = model(rgb, state, K, pose)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_without_c2_does_not_score_or_filter_the_palm(self):
        body, tcp = EmbodimentGeometry(), TCPGeometry()
        point = torch.tensor([[0., .08, -.08]])
        self.assertTrue(body.self_mask(point, torch.eye(4), torch.zeros(2)).item())
        self.assertFalse(tcp.self_mask(point, torch.eye(4), torch.zeros(2)).item())
        args = (torch.zeros(1, 30, 3), torch.eye(3).expand(30, 3, 3), torch.eye(4),
                point, torch.zeros(1), torch.zeros(2))
        self.assertGreater(float(body.contact_risk(*args)), float(tcp.contact_risk(*args)))
        self.assertEqual(len(tcp.state_dict()), 0)

    def test_each_ablation_has_finite_training_gradients_and_control(self):
        config, _, maps, qpos, goal, state, pose, K, rgb = fixture()
        for removed in ('c1', 'c2', 'c3'):
            with self.subTest(removed=removed):
                settings = copy.deepcopy(config['model'])
                settings['contributions'] = {key: key != removed for key in ('c1', 'c2', 'c3')}
                model = NavigationPolicy(settings, perception=TinyPerception())
                batch = batch_fixture(qpos, goal, pose, K, rgb, 1 if removed == 'c1' else 4)
                loss, terms = training_loss(model, maps, batch, config['train']['loss_weights'])
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(model.route.output[-1].weight.grad).all())
                if removed == 'c3':
                    self.assertEqual(float(terms['uncertainty']), 0)
                    self.assertEqual(float(terms['trust']), 0)
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.clearance.parameters()))
                    with patch.object(model.clearance, 'refine', side_effect=AssertionError('C3 must be bypassed')):
                        with torch.inference_mode():
                            self.assertTrue(torch.isfinite(model(rgb, state, K, pose)).all())
                else:
                    with torch.inference_mode():
                        self.assertTrue(torch.isfinite(model(rgb, state, K, pose)).all())
