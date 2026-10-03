"""Audit split isolation, matched training, and exact baseline rollout replay."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from tsn.common.config import create_output, read_json, write_json


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    create_output(args.output_dir)
    splits = read_json(args.training_root/'full/protocol.json')['splits']
    for part, count in [('train', 800), ('validation', 100), ('test', 100)]:
        if len(splits[part]) != count or len(set(splits[part])) != count:
            raise ValueError('Invalid episode split')
        actual = [sum(a <= int(ep.split('_')[1]) < b for ep in splits[part])
                  for a, b in [(0, 200), (200, 600), (600, 1000)]]
        if actual != [count//5, 2*count//5, 2*count//5]:
            raise ValueError('Incorrect route ratio')
    if any(set(splits[a]) & set(splits[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Overlapping splits')
    orders, epochs = [], {}
    for mode in ('full', 'current', 'deterministic', 'global', 'residual'):
        records = read_json(args.training_root/mode/'epochs.json')
        orders.append([x['sampling_sha256'] for x in records[1:]])
        epochs[mode] = read_json(args.training_root/mode/'complete.json')
    if not all(x == orders[0] for x in orders) or len(orders[0]) != 15:
        raise ValueError('Training was not matched')
    cache_checks = {}
    for part in ('train', 'validation'):
        root = args.reference/'cache'/part
        data = {key: np.load(root/f'{key}.npy', mmap_mode='r') for key in ('episode', 'frame', 'source', 'history', 'ages', 'mask')}
        h = data['history']
        checks = dict(same_episode=bool(np.all(data['episode'][h] == data['episode'][:, None])),
                      no_future=bool(np.all(data['frame'][h] <= data['frame'][:, None])),
                      recovery_singleton=bool(np.all(data['mask'][data['source'] == 1].sum(1) == 1)))
        if not all(checks.values()):
            raise ValueError('Invalid cached history')
        cache_checks[part] = checks
    replay = {}
    original = args.reference/'rollouts/validation/single_memory'
    current = args.root/'baseline'
    for ep in splits['validation']:
        old = np.load(original/'episodes'/ep/'trajectory.npz')
        new = np.load(current/'episodes'/ep/'trajectory.npz')
        checks = {key: np.array_equal(old[key], new[key]) for key in ('qpos', 'T_B_E', 'predicted_joint_targets')}
        old_result = read_json(original/'episodes'/ep/'metrics.json')
        new_result = read_json(current/'episodes'/ep/'metrics.json')
        checks['outcome'] = old_result['termination'] == new_result['termination']
        checks['steps'] = old_result['control_steps'] == new_result['control_steps']
        if not all(checks.values()):
            raise ValueError(f'Baseline replay differs: {ep}: {checks}')
        replay[ep] = True
    frozen = read_json(args.root/'protocol.json')['source_sha256']
    repository = Path(__file__).resolve().parents[3]
    numerical = ['src/tsn/models/compact_policy.py', 'src/tsn/models/clearance_policy.py',
                 'src/tsn/models/pi3_policy.py', 'src/tsn/models/kinematics.py',
                 'src/tsn/evaluation/closed_loop.py', 'src/tsn/simulation/episode.py',
                 'src/tsn/simulation/benchmark_scene.py']
    for name in numerical:
        if sha256(repository/name) != frozen[name]:
            raise ValueError(f'Numerical source changed during evaluation: {name}')
    checkpoint_hash = sha256(args.reference/'simplified.pt')
    if checkpoint_hash != 'fa3d77a605426da18d4bbb15a7bc982f3b1c9487b60d3e6fbc897850bf4f000b':
        raise ValueError('Baseline checkpoint changed')
    write_json(args.output_dir/'audit.json', dict(disjoint_splits=True, route_ratio=[2, 4, 4],
               matched_training=True, training_epochs=75, selected_heads=epochs,
               causal_cache=cache_checks, baseline_bitwise_replay_episodes=len(replay),
               unchanged_numerical_sources=numerical, unchanged_baseline_sha256=checkpoint_hash,
               selection=read_json(args.root/'selection.json')))
    print('Audit passed: 75 matched training epochs, causal cache, 100 exact baseline replays, unchanged model sources.', flush=True)


if __name__ == '__main__':
    main()
