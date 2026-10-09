"""Audit saved rollout trajectories and report additional XYZ success tolerances."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil

import h5py
import numpy as np

from tsn.common.config import read_json, write_json

TOLERANCES_CM = (1, 3, 5, 10)
METHODS = ('gtsn', 'diffusion_policy', 'dp3', 'flowpolicy', 'carp')
LABELS = {'gtsn': 'GTSN, archived', 'diffusion_policy': 'Diffusion Policy adaptation',
          'dp3': 'DP3 adaptation, depth', 'flowpolicy': 'FlowPolicy adaptation, depth',
          'carp': 'CARP adaptation'}


def summarize_records(records):
    """Keep whole-rollout collision rejection separate from earlier-stop estimates."""
    count = len(records)
    collisions = sum(r['collision_step'] is not None for r in records)
    result = {'episodes': count, 'collisions': collisions,
              'collision_rate': collisions/count if count else None,
              'whole_rollout': {}, 'first_hit_stopping_estimates': {}}
    for cm in TOLERANCES_CM:
        key = str(cm)
        successes = sum(r['first_hit_steps'][key] is not None and r['collision_step'] is None
                        for r in records)
        earlier_successes = sum(r['first_hit_steps'][key] is not None and
            (r['collision_step'] is None or r['first_hit_steps'][key] < r['collision_step'])
            for r in records)
        earlier_collisions = sum(r['collision_step'] is not None and
            (r['first_hit_steps'][key] is None or r['collision_step'] <= r['first_hit_steps'][key])
            for r in records)
        result['whole_rollout'][key] = {'successes': successes,
            'success_rate': successes/count if count else None}
        result['first_hit_stopping_estimates'][key] = {'successes': earlier_successes,
            'success_rate': earlier_successes/count if count else None,
            'collisions': earlier_collisions, 'collision_rate': earlier_collisions/count if count else None,
            'timeouts': count-earlier_successes-earlier_collisions}
    return result


def percentage(rate):
    return f'{100*rate:.0f}%'


def whole_table(models, partition):
    lines = ['| Model | 1 cm | 3 cm | 5 cm | 10 cm | Collision rate |',
             '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        row = models[method][partition]['overall']
        values = [percentage(row['whole_rollout'][str(cm)]['success_rate']) for cm in TOLERANCES_CM]
        lines.append('| '+LABELS[method]+' | '+' | '.join(values)+
                     ' | '+percentage(row['collision_rate'])+' |')
    return '\n'.join(lines)


def report_section(result):
    lines = ['## Success at additional position tolerances', '',
        'These are post hoc scores of the original 1 cm-stopping rollouts, with 100 episodes per model '
        'per partition. Success requires a saved end-effector position within the stated XYZ tolerance '
        'and no collision anywhere in the original recorded episode. Orientation is not required. '
        'Collision rate is the fraction of original episodes with a detected collision, so it is unchanged '
        'across tolerance columns. No policies were retrained or rerun.', '',
        '### Test', '', whole_table(result['models'], 'test'), '',
        '### Validation', '', whole_table(result['models'], 'validation'), '',
        '### Estimates with earlier stopping', '',
        'For a different stopping rule, the saved trajectory prefix also supports the following estimates: '
        'stop at the first recorded control step within the tolerance. A collision at that same step or '
        'earlier remains a failure; collisions after an earlier reach are excluded. These are trajectory-prefix '
        'estimates, not new simulator runs. They reproduce the original 1 cm successes and collisions exactly. '
        'Collision detection retains the recorded 100 Hz physics checks; reaching is observed at 20 Hz control steps.', '',
        '| Model, test partition | 3 cm success | 3 cm collision | 5 cm success | 5 cm collision | 10 cm success | 10 cm collision |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        estimates = result['models'][method]['test']['overall']['first_hit_stopping_estimates']
        values = []
        for cm in (3, 5, 10):
            values.extend(percentage(estimates[str(cm)][key]) for key in ('success_rate', 'collision_rate'))
        lines.append('| '+LABELS[method]+' | '+' | '.join(values)+' |')
    lines += ['', 'The audit recomputed distances from all 1,000 saved end-effector trajectories and benchmark '
        'goal positions, verified their per-episode metrics, and exactly reproduced the original 1 cm '
        'success and collision counts. `success_thresholds.json` includes route breakdowns and both definitions; '
        '`threshold_reporting/` preserves per-episode first-hit steps, input hashes, source and provenance. '
        'Model capacity, pretraining and input differences remain as stated elsewhere in this report.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    summary = read_json(root/'summary.json')
    assert summary['audit']['passed'] and summary['audit']['baseline_rollouts'] == 800
    config = read_json(root/'base_config.json')
    assert config['eval']['goal_position_tolerance_m'] == .01
    assert config['eval']['goal_success_criterion'] == 'xyz' and config['eval']['stop_on_collision']
    reference = Path(summary['protocol']['main_root'])
    dataset = Path(config['benchmark']['root'])
    splits = read_json(root/'cache/splits.json')
    output = root/'threshold_reporting'
    output.mkdir(exist_ok=True)
    hashes, goal_cache, models, episodes = {}, {}, {}, {}

    def tracked_json(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_json(path)

    for method in METHODS:
        models[method], episodes[method] = {}, {}
        directory = reference/'full' if method == 'gtsn' else root/method
        for partition in ('validation', 'test'):
            rows = tracked_json(directory/partition/'closed_loop.json')['results']
            assert len(rows) == 100 and {r['episode_id'] for r in rows} == set(splits[partition])
            assert dict(Counter(r['route'] for r in rows)) == dict(direct=20, over=40, side=40)
            records = []
            for row in rows:
                episode = row['episode_id']
                episode_dir = directory/partition/'episodes'/episode
                assert tracked_json(episode_dir/'metrics.json') == row
                if episode not in goal_cache:
                    with h5py.File(dataset/episode/'episode.h5', 'r') as handle:
                        goal_cache[episode] = np.asarray(handle['goal_pose_xyz_wxyz'][:3], dtype=np.float32)
                    hashes[f'goal_xyz:{episode}'] = hashlib.sha256(goal_cache[episode].tobytes()).hexdigest()
                trajectory = episode_dir/'trajectory.npz'
                hashes[str(trajectory)] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
                with np.load(trajectory, allow_pickle=False) as trace:
                    transforms = trace['T_B_E']
                    assert transforms.shape == (row['control_steps']+1, 4, 4)
                    assert np.isfinite(transforms).all()
                    distances = np.linalg.norm(transforms[:, :3, 3]-goal_cache[episode], axis=1)
                assert abs(float(distances.min())-row['minimum_position_error_m']) < 1e-6
                assert abs(float(distances[-1])-row['final_position_error_m']) < 1e-6
                assert row['goal_success_criterion'] == 'xyz'
                collision_step = row['collision']['control_step'] if row['collision'] is not None else None
                if collision_step is not None:
                    assert collision_step == row['control_steps']
                hits = {}
                for cm in TOLERANCES_CM:
                    indices = np.flatnonzero(distances <= cm/100)
                    hits[str(cm)] = int(indices[0]) if len(indices) else None
                record = {'episode_id': episode, 'route': row['route'],
                          'minimum_position_error_m': float(distances.min()),
                          'collision_step': collision_step, 'first_hit_steps': hits}
                raw_success = hits['1'] is not None and collision_step is None
                assert bool(row['success']) == raw_success
                records.append(record)
            overall = summarize_records(records)
            assert overall['whole_rollout']['1']['successes'] == sum(r['success'] for r in rows)
            assert overall['first_hit_stopping_estimates']['1']['successes'] == overall['whole_rollout']['1']['successes']
            assert overall['first_hit_stopping_estimates']['1']['collisions'] == overall['collisions']
            assert all(overall['whole_rollout'][str(a)]['successes'] <= overall['whole_rollout'][str(b)]['successes']
                       for a, b in zip(TOLERANCES_CM, TOLERANCES_CM[1:]))
            models[method][partition] = {'overall': overall, 'by_route': {
                route: summarize_records([r for r in records if r['route'] == route])
                for route in ('direct', 'over', 'side')}}
            episodes[method][partition] = records
    result = {'tolerances_cm': list(TOLERANCES_CM),
        'primary_definition': 'Any saved XYZ position within tolerance and collision-free entire original rollout.',
        'original_rollout_stop_tolerance_cm': 1, 'orientation_required': False,
        'collision_rate_definition': 'Original episodes with a detected collision divided by all episodes.',
        'first_hit_estimates_definition': 'Stop at first recorded reach; a same-step or earlier collision fails.',
        'new_simulator_runs': False, 'models': models,
        'audit': {'passed': True, 'rollouts_verified': 1000, 'baseline_rollouts': 800,
                  'main_rollouts': 200, 'original_1cm_counts_reproduced': True}}
    write_json(root/'success_thresholds.json', result)
    write_json(output/'episode_metrics.json', episodes)
    write_json(output/'input_sha256.json', hashes)
    summary['success_thresholds'] = result
    write_json(root/'summary.json', summary)
    report = root/'RESULTS.md'
    text = report.read_text()
    marker = '\n## Success at additional position tolerances\n'
    next_marker = '\n## Capacity, reconstruction and execution scope\n'
    if marker in text:
        before, after = text.split(marker, 1)
        text = before + (next_marker+after.split(next_marker, 1)[1] if next_marker in after else '\n')
    section = '\n'+report_section(result)+'\n'
    if next_marker in text:
        text = text.replace(next_marker, section+next_marker, 1)
    else:
        text += section
    report.write_text(text)
    source = output/'report_success_thresholds.py'
    if source.resolve() != Path(__file__).resolve():
        shutil.copyfile(__file__, source)
    write_json(output/'provenance.json', {'script_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'input_manifest_sha256': hashlib.sha256((output/'input_sha256.json').read_bytes()).hexdigest(),
        'changed_model_weights': False, 'changed_raw_episode_metrics': False,
        'changed_frozen_source': False, 'audit_passed': True})
    print(whole_table(models, 'test'), flush=True)


if __name__ == '__main__':
    main()
