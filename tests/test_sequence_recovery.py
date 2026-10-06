"""Verified teacher labels, causal masking, and training-only collection."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from tsn.training.sequence_recovery import training_subset, verify_targets, corrective_candidates, corrective_label
from tsn.training.corrective import SequenceCache, training_loss


class SequenceRecoveryTests(unittest.TestCase):
    def test_collection_is_stratified_disjoint_and_training_only(self):
        catalog = {f'episode_{i:03}': 'direct' if i < 200 else ('over' if i < 600 else 'side') for i in range(1000)}
        train = [ep for route, count in (('direct', 160), ('over', 320), ('side', 320))
                 for ep in [e for e in catalog if catalog[e] == route][:count]]
        shards = [training_subset({'train': train}, catalog, 200, i, 4, 42) for i in range(4)]
        combined = sum(shards, [])
        self.assertEqual(len(combined), len(set(combined)))
        self.assertTrue(set(combined) <= set(train))
        for shard in shards:
            self.assertEqual([sum(catalog[ep] == r for ep in shard) for r in ('direct', 'over', 'side')], [10, 20, 20])

    def test_trial_restores_pose_velocity_and_drives_even_after_collision(self):
        class Joint:
            def __init__(self): self.target, self.speed = 2., 3.
            def get_drive_target(self): return self.target
            def get_drive_velocity_target(self): return self.speed
            def set_drive_target(self, value): self.target = value
            def set_drive_velocity_target(self, value): self.speed = value
        class Simulation:
            def __init__(self):
                self.q, self.velocity = np.zeros(9), np.ones(9)
                self.joints = [Joint()]
                self.robot = SimpleNamespace(get_qvel=lambda: self.velocity.copy())
                self.scene = SimpleNamespace(physx_system=self)
                self.steps = 0
            def pack(self): return self.q.copy(), self.velocity.copy()
            def unpack(self, value): self.q, self.velocity = [v.copy() for v in value]
            def snapshot(self): return dict(qpos=self.q.copy())
            def step(self, target):
                self.q[:7] = target
                self.velocity[:] += 1
                self.joints[0].target = target[0]
                self.joints[0].speed = 99.
                self.steps += 1
                return self.snapshot(), ['collision'] if target[0] < 0 else [], 0
        simulation = Simulation()
        candidates = np.ones((2, 30, 7))
        candidates[0] = -1
        chosen, tried = verify_targets(simulation, candidates)
        self.assertEqual((chosen, tried, simulation.steps), (1, 2, 31))
        np.testing.assert_array_equal(simulation.q, np.zeros(9))
        np.testing.assert_array_equal(simulation.velocity, np.ones(9))
        self.assertEqual((simulation.joints[0].target, simulation.joints[0].speed), (2., 3.))
        with self.assertRaisesRegex(ValueError, 'exactly 30'):
            verify_targets(simulation, candidates[:, :15])

    def test_unverified_observation_has_no_action_gradient(self):
        prediction = torch.ones(1, 3, 6, 3, requires_grad=True)
        batch = dict(valid=torch.ones(1, 3, dtype=torch.bool),
                     supervision_valid=torch.tensor([[False, True, False]]),
                     waypoint=torch.zeros_like(prediction), teacher=torch.zeros(1, 3, 16, 3),
                     geometry_valid=torch.ones(1, 3, 16, dtype=torch.bool))
        training_loss(prediction, batch).backward()
        self.assertEqual(float(prediction.grad[:, [0, 2]].abs().sum()), 0.)
        self.assertGreater(float(prediction.grad[:, 1].abs().sum()), 0.)

    def test_recovery_sampling_preserves_source_and_route_mix(self):
        data = SequenceCache.__new__(SequenceCache)
        data.device = torch.device('cpu')
        data.sequences = [np.array([i]) for i in range(9)]
        data.arrays = dict(source=torch.tensor([0]*3+[1]*3+[2]*3), route=torch.tensor([0, 1, 2]*3),
                           supervision_valid=torch.ones(9, dtype=torch.bool))
        weights = data.sampling()
        torch.testing.assert_close(weights[:6], torch.zeros(6, dtype=torch.float64))
        torch.testing.assert_close(weights[6:], torch.tensor([.2, .4, .4], dtype=torch.float64))

    def test_corrective_candidates_fit_the_learned_residual_bound(self):
        route = torch.randn(6, 3)*.1
        candidates = corrective_candidates(route, torch.tensor([1., 0., 0.]))
        self.assertEqual(candidates.shape, (24, 6, 3))
        self.assertLessEqual(float((candidates-route).abs().max()), .035001)
        torch.testing.assert_close(candidates[0, -1]-route[-1], torch.tensor([0., 0., .015]))

    def test_corrective_teacher_preserves_safe_proposal_and_masks_failed_search(self):
        policy = SimpleNamespace(route_actions=Mock(return_value=torch.zeros(24, 30, 7)), servo_radius=.08)
        route, action = torch.ones(6, 3)*.1, torch.zeros(1, 30, 7)
        state, pose = torch.zeros(1, 16), torch.eye(4)[None]
        args = (policy, object(), route, action, state, pose, pose, action)
        with patch('tsn.training.sequence_recovery.verify_targets', return_value=(0, 1)):
            target, supervised, unsafe, attempts = corrective_label(*args)
            torch.testing.assert_close(target, route, rtol=0, atol=0)
            self.assertEqual((supervised, unsafe, attempts), (True, False, 1))
            policy.route_actions.assert_not_called()
        with patch('tsn.training.sequence_recovery.verify_targets', side_effect=[(None, 1), (2, 3)]):
            target, supervised, unsafe, attempts = corrective_label(*args)
            self.assertEqual((supervised, unsafe, attempts), (True, True, 4))
            self.assertGreater(float((target-route).abs().sum()), 0.)
            self.assertLessEqual(float((target-route).abs().max()), .035001)
        with patch('tsn.training.sequence_recovery.verify_targets', side_effect=[(None, 1), (None, 24)]):
            _, supervised, unsafe, attempts = corrective_label(*args)
            self.assertEqual((supervised, unsafe, attempts), (False, True, 25))
        policy.route_actions.reset_mock()
        state[:, 9:12] = torch.tensor([-.65/.55, 0., -.22/.5])
        with patch('tsn.training.sequence_recovery.verify_targets', return_value=(None, 1)):
            _, supervised, unsafe, attempts = corrective_label(*args)
            self.assertEqual((supervised, unsafe, attempts), (False, True, 1))
            policy.route_actions.assert_not_called()

    def test_action_weight_prioritizes_risk_without_unmasking_unverified_labels(self):
        prediction = torch.ones(1, 3, 6, 3, requires_grad=True)
        batch = dict(valid=torch.ones(1, 3, dtype=torch.bool),
                     supervision_valid=torch.tensor([[True, True, False]]),
                     action_weight=torch.tensor([[1., 8., 8.]]),
                     waypoint=torch.zeros_like(prediction), teacher=torch.zeros(1, 3, 16, 3),
                     geometry_valid=torch.ones(1, 3, 16, dtype=torch.bool))
        training_loss(prediction, batch).backward()
        torch.testing.assert_close(prediction.grad[:, 1], 8*prediction.grad[:, 0])
        self.assertEqual(float(prediction.grad[:, 2].abs().sum()), 0.)
