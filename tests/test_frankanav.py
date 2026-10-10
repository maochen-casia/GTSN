"""Real-data alignment and geometry must not inherit benchmark assumptions."""
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image

from tsn.data.frankanav import (MetadataUnpickler, causal_indices, interpolate_joints, link7_fk,
                               nearest_indices, payload_path, process_episode, scaled_intrinsics,
                               timestamp_ns, validate_processed)


class FrankaNavTests(unittest.TestCase):
    def test_causal_rgb_and_separate_depth_clock(self):
        np.testing.assert_array_equal(causal_indices([0, .035, .07, .105, .14], [0, .05, .1, .15]), [0, 1, 2, 4])
        with self.assertRaises(ValueError):
            causal_indices([.01, .05], [0])
        # Latest-depth snapshots lag RGB, but the next stored snapshot matches it.
        np.testing.assert_array_equal(nearest_indices([0, 0, .035, .07], [0, .035, .07]), [0, 2, 3])

    def test_nanosecond_timing_and_measured_state_interpolation(self):
        first = timestamp_ns(dict(sec=1785123818, nanosec=667357531))
        second = timestamp_ns(dict(sec=1785123818, nanosec=702357532))
        self.assertEqual(second - first, 35_000_001)
        q = np.repeat(np.array([0., 1., 1., 2.])[:, None], 7, axis=1)
        actual = interpolate_joints([0, 1, 1, 2], q, [.5, 1.5])
        np.testing.assert_allclose(actual, np.repeat([[.5], [1.5]], 7, axis=1))
        with self.assertRaises(ValueError):
            timestamp_ns(dict(sec=0, nanosec=0))

    def test_resize_preserves_pinhole_ray_at_corresponding_pixel_center(self):
        K = np.array([[400., 0, 315.7], [0, 401., 238.4], [0, 0, 1.]])
        resized = scaled_intrinsics(K, (480, 640), (192, 256))
        pixel = np.array([100., 70., 1.])
        source_pixel = np.array([(100.5) / .4 - .5, (70.5) / .4 - .5, 1.])
        np.testing.assert_allclose(np.linalg.inv(resized) @ pixel, np.linalg.inv(K) @ source_pixel)

    def test_pickle_globals_and_payload_paths_are_bounded(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            MetadataUnpickler(io.BytesIO(pickle.dumps(Path('/tmp')))).load()
        with self.assertRaises(ValueError):
            payload_path('/tmp', 'ep_00000', 'ep_00000/back/0000.jpg', 'front', '.jpg')
        with self.assertRaises(ValueError):
            payload_path('/tmp', 'ep_00000', '../front/0000.jpg', 'front', '.jpg')

    def test_conversion_keeps_real_gripper_frame_and_terminal_endpoint(self):
        calibration = json.loads((Path(__file__).resolve().parents[1] / 'configs/frankanav_calibration.json').read_text())
        calibration.update(image_hw=[8, 8], K=[8, 0, 3.5, 0, 8, 3.5, 0, 0, 1], distortion=[0] * 5)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'raw'
            episode = root / 'ep_00000'
            (episode / 'front').mkdir(parents=True)
            (episode / 'front_depth').mkdir()
            rows = []
            flange = np.eye(4)
            flange[2, 3] = .107
            for index in range(5):
                q = np.array([.1 * index, -.4, 0, -2, 0, 1.5, 0.])
                stamp = dict(sec=1785123818, nanosec=index * 35_000_000 + 10_000_000)
                joint_stamp = dict(sec=stamp['sec'], nanosec=stamp['nanosec'] + 1_000_000)
                paths = dict(front=f'ep_00000/front/{index:04d}.jpg')
                depths = dict(front_depth=f'ep_00000/front_depth/{index:04d}.npy')
                Image.fromarray(np.full((8, 8, 3), index * 20, dtype=np.uint8)).save(root / paths['front'])
                np.save(root / depths['front_depth'], np.full((8, 8), .3, dtype=np.float32))
                rows.append(dict(frame_id=index, language_instruction='', images=paths, depths=depths,
                                 timestamps=dict(trigger=stamp, images=dict(front=stamp), depths=dict(front_depth=stamp),
                                                 states={'/franka/joint_states': joint_stamp}),
                                 **{'/franka/joint_states': dict(position=q, velocity=np.zeros(7), effort=np.zeros(7)),
                                    '/factr_teleop/right/cmd_franka_pos': dict(position=q),
                                    '/franka/end_effector_pose': link7_fk(q, calibration) @ flange}))
            rows[-1]['timestamps']['depths']['front_depth'] = rows[-2]['timestamps']['depths']['front_depth']
            # The recorder can leave a final camera file without a metadata/state row.
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(episode / 'front/0005.jpg')
            np.save(episode / 'front_depth/0005.npy', np.ones((8, 8), dtype=np.float32))
            with (episode / 'data.pkl').open('wb') as stream:
                pickle.dump(rows, stream, protocol=5)
            output = Path(temporary) / 'processed'
            report = process_episode(root, 'ep_00000', output, calibration, (4, 4))
            self.assertTrue(report['audit']['passed'])
            self.assertEqual(report['unreferenced_rgb_files'], ['0005.jpg'])
            self.assertEqual(report['unreferenced_depth_files'], ['0005.npy'])
            self.assertEqual(report['depth_pairs_masked'], 1)
            with h5py.File(output / 'ep_00000/episode.h5') as f:
                self.assertEqual(f['qpos'].shape, (4, 8))
                self.assertEqual(f['joint_names'][-1], b'finger_joint')
                np.testing.assert_array_equal(f['qpos'][-1, :7], rows[-1]['/franka/joint_states']['position'].astype(np.float32))
                np.testing.assert_allclose(f['ee_pose'][-1], f['goal_pose_xyz_wxyz'][:])
                # Mount is relative to the flange, not blindly copied from link7.
                mount = np.asarray(calibration['T_link7_camera_cv']).reshape(4, 4)
                np.testing.assert_allclose(flange @ f['T_ee_camera_cv'][:], mount)
                self.assertFalse(f.attrs['training_ready'])
                self.assertFalse(f['depth_pair_valid'][-1])
                self.assertTrue(np.all(f['depth_m'][-1] == 0))
            path = output / 'ep_00000/episode.h5'
            with h5py.File(path, 'r+') as f:
                f['rgb_time_seconds'][1] = .06
            with self.assertRaisesRegex(ValueError, 'future'):
                validate_processed(path)


if __name__ == '__main__':
    unittest.main()
