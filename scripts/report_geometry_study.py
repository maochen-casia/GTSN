"""Audit selected learned-geometry checkpoints and paired full-partition results."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import read_json, write_json
from tsn.data.splits import episode_catalog, validate_splits
from run_geometry_study import digest, source_hashes, validate_trace


def paired_results(rows, baseline, seed=20261009):
    reference = {row['episode_id']: row for row in baseline}
    ordered = [reference[row['episode_id']] for row in rows]
    difference = np.asarray([int(row['success'])-int(old['success']) for row, old in zip(rows, ordered)])
    rng = np.random.default_rng(seed)
    draws = np.zeros(20000)
    for route in ('direct', 'over', 'side'):
        values = difference[[i for i, row in enumerate(rows) if row['route'] == route]]
        draws += values[rng.integers(0, len(values), (len(draws), len(values)))].sum(-1)
    return dict(gains=int((difference == 1).sum()), losses=int((difference == -1).sum()),
                gain_percentage_points=int(difference.sum()),
                bootstrap95_percentage_points=np.quantile(draws, [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    protocol = read_json(root/'protocol.json')
    summary = read_json(root/'summary.json')
    freeze = read_json(root/'test_freeze.json')
    if read_json(root/'status.json')['phase'] != 'complete':raise ValueError('The study is incomplete')
    if source_hashes(root/'source') != read_json(root/'source_sha256.json'):raise ValueError('Frozen study source changed')
    parent_path = Path(protocol['parent_checkpoint'])
    if digest(parent_path) != protocol['parent_sha256']:raise ValueError('Parent checkpoint changed')
    parent = load_checkpoint(parent_path)
    validate_splits(parent['splits'], episode_catalog(Path(parent['config']['benchmark']['root'])))
    manifest = read_json(root/'cache/manifest.json')
    if manifest['partition'] != 'train' or manifest['train_episodes'] != parent['splits']['train']:
        raise ValueError('Cache overlaps a held-out partition')
    if digest(root/'cache/examples.pt') != read_json(root/'cache_sha256.json')['examples']:
        raise ValueError('Training cache changed')
    baseline = {p: read_json(parent_path.parents[1]/p/'closed_loop.json')['results'] for p in ('validation', 'test')}
    audit = dict(models={}, partition_episodes={p: len(parent['splits'][p]) for p in ('train', 'validation', 'test')},
                 source_frozen=True, cache_frozen=True, parent_frozen=True, test_selection=False,
                 training_seed=protocol['seed'], reference_parity=read_json(root/'reference_parity.json'))
    runtime_files = ['c1_memory.py', 'c2_embodiment.py', 'c3_clearance.py', 'kinematics.py', 'policy.py']
    active = Path(__file__).resolve().parents[1]
    for name in runtime_files:
        relative = Path('src/tsn/models')/name
        if digest(active/relative) != digest(root/'source'/relative):raise ValueError(f'Active runtime differs: {name}')
    for name, record in summary['models'].items():
        checkpoint = Path(record['checkpoint'])
        if digest(checkpoint) != record['sha256'] or record['sha256'] != freeze['nominees'][name]['sha256']:
            raise ValueError(f'Selected checkpoint changed: {name}')
        updated = load_checkpoint(checkpoint)
        if any(not torch.isfinite(value).all() for value in updated['model'].values()):
            raise ValueError(f'Nonfinite selected weights: {name}')
        if any(not torch.equal(updated['model'][key], value) for key, value in parent['model'].items()):
            raise ValueError(f'A parent tensor changed: {name}')
        if updated['splits'] != parent['splits']:raise ValueError(f'Checkpoint split mismatch: {name}')
        if not updated['config']['model']['perception']['freeze_encoder']:raise ValueError('Encoder is not frozen')
        validation_condition = freeze['nominees'][name]['condition']
        partitions = {'validation': root/validation_condition/'validation', 'test': root/name/'test'}
        results = {}
        for p, directory in partitions.items():
            ids = parent['splits'][p]
            rows = read_json(directory/'closed_loop.json')['results']
            if [row['episode_id'] for row in rows] != ids:raise ValueError(f'Incomplete or reordered {name}/{p}')
            if read_json(directory/'complete.json')['episodes'] != len(ids):raise ValueError('Incomplete partition')
            for row in rows:
                if validate_trace(directory/'episodes'/row['episode_id'], row['episode_id']) != row:
                    raise ValueError('Trajectory summary mismatch')
            results[p] = dict(successes=sum(row['success'] for row in rows),
                collisions=sum(row['collision'] is not None for row in rows),
                xyz_orientation_successes=sum(row['xyz_orientation_success'] for row in rows),
                by_route={r: sum(row['success'] for row in rows if row['route'] == r) for r in ('direct', 'over', 'side')},
                paired=paired_results(rows, baseline[p]))
        audit['models'][name] = dict(parent_tensors_unchanged=True, finite_trajectories=True,
                                    frozen_nomination=True, partitions=results, accepted=record['accepted'])
    write_json(root/'audit.json', audit)
    print({name: row['partitions'] for name, row in audit['models'].items()})


if __name__ == '__main__':main()
