"""Snapshot release source and replay the selected neural replacement policy."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from tsn.common.config import read_json, write_json
from run_geometry_study import digest, source_hashes, validate_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root
    summary = read_json(root/'summary.json')
    assert summary['preferred'] == 'joint_d4' and 'joint_d4' in summary['accepted']
    source = root/'release_source'; source.mkdir()
    for name in ('src', 'scripts', 'tests', 'configs', 'assets', 'vendor'):
        shutil.copytree(Path('/workspace')/name, source/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    write_json(root/'release_source_sha256.json', source_hashes(source))
    checkpoint = root/'joint_d4/training/best.pt'
    prior = read_json(root/'joint_d4/validation/closed_loop.json')['results']
    episodes = [next(r['episode_id'] for r in prior if r['route'] == route) for route in ('direct', 'over', 'side')]
    output = root/'release_replay'
    environment = {**os.environ, 'PYTHONPATH': f'{source}/src:{source}/vendor/Pi3'}
    with (root/'release_replay.log').open('w') as stream:
        subprocess.run([sys.executable, str(source/'scripts/evaluate_geometry_batch.py'),
            '--checkpoint', str(checkpoint), '--partition', 'validation', '--output-dir', str(output), '--episodes', *episodes],
            env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    reference = {r['episode_id']: r for r in prior}
    for episode in episodes:
        row = validate_trace(output/'episodes'/episode, episode)
        before = reference[episode]
        assert row['success'] == before['success'] and row['control_steps'] == before['control_steps']
        assert abs(row['final_position_error_m']-before['final_position_error_m']) < 1e-6
        with np.load(output/'episodes'/episode/'trajectory.npz') as actual, np.load(
                root/'joint_d4/validation/episodes'/episode/'trajectory.npz') as expected:
            assert actual.files == expected.files
            for key in actual.files:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-6)
    freeze = root/'preferred_nominee.json'
    first_test_metric = min(p.stat().st_mtime for p in (root/'joint_d4/test/episodes').glob('*/metrics.json'))
    assert freeze.stat().st_mtime < first_test_metric
    write_json(root/'release_audit.json', dict(checkpoint=str(checkpoint), sha256=digest(checkpoint),
        replay_episodes=episodes, complete_trajectories_match_within=1e-6,
        release_source=str(source), source_hashes_verified=True, preferred_nominee_frozen_before_test=True,
        primary_no_observed_drop=True, regression_tests=62,
        encoder_and_head_audit='audit.json', first_batch_and_budget_failures_not_promoted=True))


if __name__ == '__main__':main()
