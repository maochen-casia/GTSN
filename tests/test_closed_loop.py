"""Closed-loop input isolation: no held-out expert future reaches the policy."""
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

import numpy as np
import torch

from tsn.evaluation.closed_loop import rollout
from tsn.features.maps import MapSettings
from tsn.data.hdf5_dataset import JOINT_NAMES


class RolloutTests(unittest.TestCase):
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
        options = dict(execute_horizon=15, max_control_steps=16, video_stride=2,
                       goal_position_tolerance_m=.01, goal_orientation_tolerance_rad=.15,
                       contact_impulse_threshold_ns=1e-7, stop_on_collision=True,
                       goal_success_criterion='xyz', render_videos=False)
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

