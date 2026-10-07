"""Terminal target semantics for expert and existing recovery data."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tsn.data.hdf5_dataset import padded_future
from tsn.data.recovery_dataset import RecoveryDataset
from tsn.data.history import build_history
from tsn.data.navigation import sampling_weights


class DataTests(unittest.TestCase):
    def test_histories_are_causal_episode_local_and_reset_for_perturbations(self):
        episode = np.array([0, 0, 0, 0, 0, 1, 1])
        frames = np.array([0, 15, 30, 45, 20, 0, 15])
        source = np.array([0, 0, 0, 0, 1, 0, 0])
        history, ages, mask = build_history(episode, frames, source)
        np.testing.assert_array_equal(history[3], [0, 1, 2, 3])
        np.testing.assert_array_equal(ages[3], [45, 30, 15, 0])
        self.assertEqual(mask[4].sum(), 1)
        self.assertTrue(np.all(history[4] == 4))
        self.assertTrue(np.all(episode[history] == episode[:, None]))
        self.assertTrue(np.all(frames[history] <= frames[:, None]))

    def test_source_and_route_mixtures_do_not_depend_on_group_sizes(self):
        routes = np.array([0, 1, 1, 2, 2, 2]*2)
        sources = np.repeat([0, 1], 6)
        weights = sampling_weights(routes, sources).numpy()
        self.assertAlmostEqual(weights[sources == 0].sum(), .65)
        self.assertAlmostEqual(weights[sources == 1].sum(), .35)
        for route, probability in enumerate((.2, .4, .4)):
            self.assertAlmostEqual(weights[routes == route].sum(), probability)
        with self.assertRaises(ValueError):sampling_weights(routes[:6], sources[:6])

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
