"""Audit updated benchmark calibration, FK, and reconstructed depth in Docker."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

from tsn.common.config import read_json, write_json
from tsn.data.hdf5_dataset import validate_episode
from tsn.data.splits import make_splits, episode_catalog
from tsn.models.kinematics import PandaKinematics
from tsn.simulation.episode import EpisodeSimulation, quaternion_matrix


def check_views(job):
    root, episode, options = job
    with h5py.File(Path(root)/episode/'episode.h5', 'r') as f:
        q, ee = f['qpos'][:], f['ee_pose'][:]
        calibration, mount = f['intrinsics'][:], f['T_ee_camera_cv'][:]
        names = tuple(x.decode() for x in f['joint_names'][:])
        rows = []
        with EpisodeSimulation(read_json(Path(root)/episode/'scene.json'), calibration, mount,
                               f['depth_m'].shape[1:], q[0], ee[0], names, options) as sim:
            for frame in (0, len(q)//2, len(q)-1):
                sim.robot.set_qpos(q[frame])
                rgb, depth = sim.render()
                expected = f['depth_m'][frame]
                delta = abs(depth-expected)
                expected_visible = expected > 0
                rows.append(dict(episode=episode, frame=frame, source_hw=list(depth.shape),
                                 median_depth_error_m=float(np.median(delta[expected_visible])),
                                 fraction_within_2mm=float((delta[expected_visible] <= .00201).mean()),
                                 fraction_within_10mm=float((delta[expected_visible] <= .01001).mean()),
                                 missing_fraction=float(((depth == 0)&expected_visible).mean()),
                                 camera_pose_error=float(abs(sim.snapshot()['T_B_C']-f['T_base_camera_cv'][frame]).max())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read_json(args.config)
    root = Path(config['benchmark']['root'])
    catalog, splits = episode_catalog(root), make_splits(config['benchmark'])
    kin = PandaKinematics(config['model'].get('robot', 'panda'))
    worst_position, worst_rotation, worst_camera = 0., 0., 0.
    intrinsics, mounts, resolutions = set(), set(), set()
    for episode, route in catalog.items():
        with h5py.File(root/episode/'episode.h5', 'r') as f:
            validate_episode(f, route)
            q, ee = f['qpos'][:], f['ee_pose'][:]
            actual = kin(torch.as_tensor(q[:, :7], dtype=torch.float32)).detach().numpy()
            desired_rotation = np.stack([quaternion_matrix(p[3:]) for p in ee])
            worst_position = max(worst_position, float(np.linalg.norm(actual[:, :3, 3]-ee[:, :3], axis=-1).max()))
            worst_rotation = max(worst_rotation, float(abs(actual[:, :3, :3]-desired_rotation).max()))
            worst_camera = max(worst_camera, float(abs(actual@f['T_ee_camera_cv'][:]-f['T_base_camera_cv'][:]).max()))
            intrinsics.add(f['intrinsics'][:].tobytes()); mounts.add(f['T_ee_camera_cv'][:].tobytes())
            resolutions.add(tuple(f['depth_m'].shape[1:]))
    jobs = [(str(root), f'episode_{index:03d}', config['eval'])
            for index in (*range(5), *range(200, 205), *range(600, 605))]
    with ProcessPoolExecutor(args.workers) as pool:
        views = [row for group in pool.map(check_views, jobs) for row in group]
    report = dict(episodes=len(catalog), partitions={k:len(splits[k]) for k in ('train', 'validation', 'test')},
                  unique_intrinsics=len(intrinsics), unique_mounts=len(mounts), resolutions_hw=sorted(resolutions),
                  max_fk_position_error_m=worst_position, max_fk_rotation_matrix_error=worst_rotation,
                  max_camera_pose_matrix_error=worst_camera, reconstructed_views=views)
    assert worst_position < 2e-6 and worst_rotation < 2e-6 and worst_camera < 3e-6
    assert len(intrinsics) == len(mounts) == 1000
    write_json(args.output, report)
    print({k:v for k,v in report.items() if k != 'reconstructed_views'}, flush=True)
    print('Depth within 2/10 mm, worst views:', min(row['fraction_within_2mm'] for row in views),
          min(row['fraction_within_10mm'] for row in views), flush=True)


if __name__ == '__main__':
    main()
