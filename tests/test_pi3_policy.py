"""Verify learned-map inference isolation and end-to-end backbone gradients."""
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

import numpy as np
import torch

from tsn.common.config import default_config, read_json
from tsn.evaluation.open_loop import predict_batch
from tsn.models.factory import make_maps, make_policy
from tsn.training.losses import predicted_map_loss
from tsn.evaluation.closed_loop import rollout
from tsn.features.maps import MapSettings
from tsn.data.hdf5_dataset import JOINT_NAMES


class Pi3PolicyTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Parallel Pi3 integration requires two GPUs')
    def test_parallel_inference_matches_single_gpu_with_cached_positions(self):
        config = read_json(default_config('model', 'pi3_small.json'))
        model = make_policy(config, initialize_backbone=False).cuda().eval()
        rgb = torch.randint(0, 256, (2, 24, 32, 3), dtype=torch.uint8, device='cuda')
        state = torch.randn(2, 16, device='cuda')
        K = torch.eye(3, device='cuda')[None].repeat(2, 1, 1)
        T = torch.eye(4, device='cuda')[None].repeat(2, 1, 1)
        with torch.inference_mode():
            expected = model(rgb, state, K, T)
            parallel = torch.nn.DataParallel(model, device_ids=[0, 1])
            actual = parallel(rgb, state, K, T)
        torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.03)

    def test_closed_loop_never_reads_expert_future_or_teacher_maps(self):
        class InitialStateOnly:
            def __init__(self, values):
                self.values = values
            def __getitem__(self, index):
                if index != slice(None, 1, None):
                    raise AssertionError('Closed loop read a held-out expert trajectory')
                return self.values[:1]
        handle = {'qpos': InitialStateOnly(np.zeros((2, 9), dtype=np.float32)),
                  'ee_pose': InitialStateOnly(np.array([[0., 0., 0., 1., 0., 0., 0.]] * 2)),
                  'intrinsics': np.eye(3), 'T_ee_camera_cv': np.eye(4),
                  'depth_m': np.ones((2, 24, 32)), 'joint_names': np.array(JOINT_NAMES),
                  'goal_pose_xyz_wxyz': np.array([1., 0., 0., 1., 0., 0., 0.], dtype=np.float32)}
        file = MagicMock()
        file.__enter__.return_value = handle
        state = {'qpos': np.zeros(9, dtype=np.float32), 'T_B_E': np.eye(4), 'T_B_C': np.eye(4)}
        simulation = MagicMock()
        simulation.__enter__.return_value = simulation
        simulation.snapshot.return_value = state
        simulation.render.return_value = (np.zeros((24, 32, 3), dtype=np.uint8), np.ones((24, 32)))
        simulation.step.return_value = (state, [], 0)
        class Policy(torch.nn.Module):
            uses_predicted_maps = True
            chunk_size = 30
            def forward(self, rgb, normalized, K, transform):
                return torch.zeros(1, 30, 7)
        maps = Mock(settings=MapSettings(), side_effect=AssertionError('Teacher maps reached rollout'))
        options = read_json(default_config('eval', 'pi3_small.json'))
        options.update(max_control_steps=16, render_videos=False)
        with tempfile.TemporaryDirectory() as temporary, \
             patch('tsn.evaluation.closed_loop.h5py.File', return_value=file), \
             patch('tsn.evaluation.closed_loop.validate_episode'), \
             patch('tsn.evaluation.closed_loop.read_json', return_value={}), \
             patch('tsn.evaluation.closed_loop.EpisodeSimulation', return_value=simulation):
            result = rollout('episode_001', 'direct', Path(temporary), Path(temporary),
                             Policy(), maps, torch.device('cpu'), options)
            self.assertEqual(result['control_steps'], 16)
            self.assertEqual(result['replans'], 2)
            self.assertFalse(result['privileged_action_map'])
            self.assertIsNone(result['expert_progress_method'])
            with np.load(Path(temporary) / 'episodes/episode_001/trajectory.npz') as trajectory:
                self.assertEqual(trajectory['reference_indices'].size, 0)
        maps.assert_not_called()

    def test_inference_does_not_read_teacher_depth_or_future(self):
        config = read_json(default_config('model', 'pi3_small.json'))
        maps = make_maps(config)
        maps.forward = Mock(side_effect=AssertionError('Teacher maps reached inference'))
        class Policy(torch.nn.Module):
            uses_predicted_maps = True
            def forward(self, rgb, state, K, transform):
                return state[:, :7, None].transpose(1, 2).expand(-1, 30, -1)
        batch = {'rgb': torch.zeros(2, 24, 32, 3, dtype=torch.uint8),
                 'qpos': torch.zeros(2, 9), 'goal_pose': torch.tensor([[.6, 0., .2, 1., 0., 0., 0.]]).repeat(2, 1),
                 'K': torch.eye(3).repeat(2, 1, 1), 'T_B_C': torch.eye(4).repeat(2, 1, 1)}
        model = Policy()
        first = predict_batch(model, maps, batch)
        batch.update(depth=torch.full((2, 24, 32), float('nan')),
                     future_ee=torch.full((2, 30, 3), float('nan')),
                     valid_future=torch.zeros(2, 30, dtype=torch.bool), route=torch.tensor([0, 2]))
        second = predict_batch(model, maps, batch)
        torch.testing.assert_close(first, second)
        maps.forward.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), 'Pi3 gradient integration requires CUDA')
    def test_small_pi3_full_finetuning_and_predicted_map_gradient(self):
        config = read_json(default_config('model', 'pi3_small.json'))
        model = make_policy(config, initialize_backbone=False).cuda().train()
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        rgb = torch.randint(0, 256, (1, 24, 32, 3), dtype=torch.uint8, device='cuda')
        state = torch.randn(1, 16, device='cuda')
        K = torch.eye(3, device='cuda')[None]
        T = torch.eye(4, device='cuda')[None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actions, predicted = model.forward_with_maps(rgb, state, K, T)
            predicted.retain_grad()
            auxiliary, _ = predicted_map_loss(predicted, torch.zeros_like(predicted))
            loss = actions.float().square().mean() + .25 * auxiliary
        self.assertEqual(tuple(actions.shape), (1, 30, 7))
        self.assertEqual(tuple(predicted.shape), (1, 6, 80, 80))
        loss.backward()
        self.assertGreater(float(predicted.grad.abs().sum()), 0.)
        for name, module in [('encoder', model.encoder), ('decoder', model.decoder),
                             ('point', model.point_head), ('goal', model.goal_head),
                             ('action_map', model.action_map_head), ('policy', model.action_policy)]:
            parameters = list(module.parameters())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), name)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in parameters), 0., name)


if __name__ == '__main__':
    unittest.main()
