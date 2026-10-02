"""Regression checks; run with unittest inside the project Docker image."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tsn.data.hdf5_dataset import padded_future
from tsn.data.loaders import make_recovery_loader
from tsn.data.recovery_dataset import RecoveryDataset
from tsn.data.recovery_generation import _generate_episode
from tsn.training.losses import imitation_loss
from tsn.training.selection import selection_key, validation_episodes


class RecoveryFixTests(unittest.TestCase):
    def test_episode_without_safe_perturbations_is_recorded_as_rejected(self):
        handle = {
            "qpos": np.zeros((2, 9), dtype=np.float32), "ee_pose": np.zeros((2, 7)),
            "intrinsics": np.eye(3), "T_ee_camera_cv": np.eye(4),
            "goal_pose_xyz_wxyz": np.zeros(7), "joint_names": np.array(["unused"]),
            "depth_m": np.ones((2, 4, 4)),
        }
        file = MagicMock()
        file.__enter__.return_value = handle
        simulation = MagicMock()
        simulation.__enter__.return_value = simulation
        simulation.qlimits = np.tile([-10., 10.], (7, 1))
        simulation._contacts.return_value = []
        with tempfile.TemporaryDirectory() as temporary, \
             patch("tsn.data.recovery_generation.h5py.File", return_value=file), \
             patch("tsn.data.recovery_generation.validate_episode"), \
             patch("tsn.data.recovery_generation.read_json", return_value={"objects": []}), \
             patch("tsn.data.recovery_generation.EpisodeSimulation", return_value=simulation), \
             patch("tsn.data.recovery_generation._minimum_clearance", return_value=(-.01, "obstacle")):
            result = _generate_episode({"episode": "episode_301", "route": "over", "dataset_root": temporary,
                "output_dir": temporary, "eval_options": {}, "chunk_size": 30, "samples_per_episode": 4,
                "noise_std_rad": .035, "noise_max_rad": .1, "max_attempts": 2, "minimum_clearance_m": .002})
            self.assertEqual(result["accepted"], 0)
            self.assertIsNone(result["minimum_clearance_m"])
            self.assertFalse((Path(temporary) / "episode_301.npz").exists())

    def test_terminal_correction_has_loss_and_gradient_at_every_executed_horizon(self):
        q = np.zeros((2, 7), dtype=np.float32)
        future, valid = padded_future(q, 1, 30)
        target = torch.from_numpy(future - .426)[None]
        prediction = torch.zeros_like(target, requires_grad=True)
        loss = imitation_loss(prediction, target, torch.from_numpy(valid)[None], .02, 10., .25)
        loss.backward()
        self.assertGreater(float(loss), .4)
        self.assertTrue(torch.all(prediction.grad[0, :15].abs().sum(-1) > 0))

    def test_near_terminal_future_keeps_hold_targets(self):
        q = np.arange(28, dtype=np.float32).reshape(4, 7)
        future, valid = padded_future(q, 2, 30)
        np.testing.assert_array_equal(future, np.repeat(q[-1:], 30, axis=0))
        self.assertTrue(valid.all())

    def test_legacy_archive_terminal_mask_is_repaired_without_changing_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            target = np.full((1, 30, 7), -.426, dtype=np.float32)
            arrays = dict(depth=np.ones((1, 4, 4)), T_B_C=np.eye(4)[None], K=np.eye(3)[None],
                          future_ee=np.ones((1, 30, 3)), qpos=np.zeros((1, 9)), goal_pose=np.zeros((1, 7)),
                          target=target, valid_future=np.zeros((1, 30), dtype=bool),
                          frame_index=np.array([10]), route=np.array("over"), episode_id=np.array("episode_211"))
            np.savez(path / "episode_211.npz", **arrays)
            data = RecoveryDataset(path, 30)
            self.assertTrue(data[0]["valid_future"].all())
            np.testing.assert_array_equal(data[0]["target"].numpy(), target[0])
            data.close()
            arrays["target"][0, -1, 0] = 1
            np.savez(path / "episode_211.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "not terminal holds"):
                RecoveryDataset(path, 30)

    def test_source_and_route_sampling_weights(self):
        class Expert:
            ids = ["direct", "over", "side"]
            counts = [3, 5, 7]
            catalog = dict(zip(ids, ids))
            stride = 1
            def __len__(self):
                return 12
        class Recovery:
            route_indices = [0, 1, 1, 2, 2, 2]
            def __len__(self):
                return len(self.route_indices)
        loader = make_recovery_loader(Expert(), [Recovery(), Recovery()],
            {"batch_size": 2, "num_workers": 0, "device": "cpu"}, 1, [.35, .23],
            {route: 1 / 3 for route in Expert.ids})
        weights = loader.sampler.weights.numpy()
        for part, fraction, routes in [(weights[:12], .42, [0] * 2 + [1] * 4 + [2] * 6),
                                      (weights[12:18], .35, Recovery.route_indices),
                                      (weights[18:], .23, Recovery.route_indices)]:
            self.assertAlmostEqual(part.sum(), fraction)
            for route in range(3):
                self.assertAlmostEqual(part[np.array(routes) == route].sum(), fraction / 3)

    def test_checkpoint_rank_prefers_safety_over_prediction_rmse(self):
        def key(success, collision, recovery, expert):
            return selection_key({"overall": {"success_rate": success, "collision_rate": collision}},
                                  {"rmse_rad": recovery}, {"rmse_rad": expert})
        parent = key(.6, .1, .06, .05)
        self.assertGreater(parent, key(.5, 0., .01, .01))
        self.assertGreater(parent, key(.6, .2, .01, .01))
        self.assertGreater(key(.7, .1, .07, .06), parent)
        self.assertFalse(parent > parent)

    def test_selection_uses_only_fixed_validation_episodes(self):
        catalog = {f"{route}_{i}": route for route in ("direct", "over", "side") for i in range(10)}
        split = {"validation": sorted(catalog), "train": ["training"], "test": ["testing"]}
        counts = {"direct": 3, "over": 6, "side": 6}
        ids = validation_episodes(split, catalog, counts, 20261004)
        self.assertEqual(ids, validation_episodes(split, catalog, counts, 20261004))
        self.assertEqual(len(ids), 15)
        self.assertFalse(set(ids) & set(split["train"] + split["test"]))


if __name__ == "__main__":
    unittest.main()
