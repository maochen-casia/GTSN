"""Add RGB at saved perturbation states, preserving original labels and archives."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time

import h5py
import numpy as np

from tsn.common.config import read_json, write_json
from tsn.data.recovery_generation import _set_simulation_state
from tsn.simulation.episode import EpisodeSimulation


def augment(job):
    source, destination, dataset, options = job
    path = destination / source.name
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if 'rgb' not in archive:
                raise ValueError(f'{path}: incomplete RGB archive')
            count = len(archive['rgb'])
        return {'episode': source.stem, 'samples': count, 'retained_validated_archive': True,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'rgb_archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with np.load(source, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    episode = str(arrays['episode_id'])
    with h5py.File(dataset / episode / 'episode.h5', 'r') as handle:
        names = tuple(v.decode() if isinstance(v, bytes) else str(v) for v in handle['joint_names'][:])
        arguments = (read_json(dataset / episode / 'scene.json'), handle['intrinsics'][:],
                     handle['T_ee_camera_cv'][:], handle['depth_m'].shape[1:],
                     handle['qpos'][0], handle['ee_pose'][0], names, options)
    rgb = []
    differences = []
    with EpisodeSimulation(*arguments) as simulation:
        for index, q in enumerate(arrays['qpos']):
            _set_simulation_state(simulation, q)
            measured = simulation.snapshot()['T_B_C']
            if not np.allclose(measured, arrays['T_B_C'][index], atol=1e-5):
                raise ValueError(f'{episode}: saved camera transform differs from measured state')
            color, depth = simulation.render()
            difference = np.abs(depth - arrays['depth'][index].astype(np.float32))
            if difference.mean() > .0005 or np.quantile(difference, .99) > .002:
                raise ValueError(f'{episode}: reconstructed depth disagrees with saved observation')
            differences.append(float(difference.mean()))
            rgb.append(color)
    arrays['rgb'] = np.stack(rgb)
    path = destination / source.name
    with path.with_suffix('.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix('.npz.tmp').replace(path)
    return {'episode': episode, 'samples': len(rgb), 'mean_depth_difference_m': float(np.mean(differences)),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'rgb_archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    source = Path('/run/user/1016/experiments/recovery_fixed_20261001/perturbation_data')
    dataset = Path('/run/user/1016/tsn-1k')
    options = read_json('/workspace/configs/eval/pi3_small.json')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.monotonic()
    results = []
    jobs = [(p, args.output_dir, dataset, options) for p in sorted(source.glob('episode_*.npz'))]
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(augment, job) for job in jobs]):
            results.append(future.result())
            if len(results) % 20 == 0 or len(results) == len(jobs):
                print(f'RGB recovery: {len(results)}/{len(jobs)} episodes', flush=True)
    manifest = read_json(source / 'manifest.json')
    manifest['rgb_augmentation'] = {'source': str(source), 'method': 'render at saved measured qpos',
                                   'depth_agreement_checked': True, 'labels_unchanged': True,
                                   'wall_seconds': time.monotonic() - start, 'results': results}
    write_json(args.output_dir / 'manifest.json', manifest)
    write_json(args.output_dir / 'source_split.json', read_json(source / 'source_split.json'))
    print(f'Prepared {sum(r["samples"] for r in results)} RGB perturbation samples', flush=True)


if __name__ == '__main__':
    main()
