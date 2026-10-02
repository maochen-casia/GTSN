"""Run four matched diagnostic conditions on separate idle project GPUs."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from tsn.common.config import experiment_path, read_json, write_json
from tsn.evaluation.map_diagnostics import episode_bootstrap
from adapt_pi3_action_head import MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = experiment_path(args.root)
    assert read_json(root / 'status.json')['state'] == 'complete'
    assert read_json(root / 'training_cache_status.json')['state'] == 'complete'
    directory = root / 'logs'
    directory.mkdir(exist_ok=True)
    all_results = {}
    baseline = np.load(root / 'paired_episode_errors.npz')
    for seed in (20261002, 20261003):
        processes = []
        for gpu, mode in enumerate(MODES):
            path = root / 'adapted_heads' / f'seed_{seed}' / mode / 'results.json'
            if path.is_file():
                continue
            log = (directory / f'head_{seed}_{mode}.log').open('w')
            command = [sys.executable, '/workspace/scripts/adapt_pi3_action_head.py', '--root', str(root),
                       '--mode', mode, '--seed', str(seed)]
            process = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                       stdout=log, stderr=subprocess.STDOUT)
            processes.append((mode, process, log))
        while any(process.poll() is None for _, process, _ in processes):
            print(f'Adaptation seed {seed}: ' + ', '.join(f'{mode}={process.poll()}' for mode, process, _ in processes), flush=True)
            time.sleep(20)
        for mode, process, log in processes:
            log.close()
            if process.returncode:
                raise RuntimeError(f'{mode} failed; see {directory}/head_{seed}_{mode}.log')
        records, errors = {}, {}
        histories = {}
        initial_hashes = set()
        for mode in MODES:
            output = root / 'adapted_heads' / f'seed_{seed}' / mode
            records[mode] = read_json(output / 'results.json')
            paired = np.load(output / 'paired_episode_errors.npz')
            errors[mode] = paired['squared']
            assert np.array_equal(paired['counts'], baseline['counts'])
            histories[mode] = read_json(output / 'epochs.json')
            initial_hashes.add(read_json(output / 'protocol.json')['initialization_sha256'])
            assert records[mode]['completed_epochs'] == 10
            assert len(histories[mode]) == 10
        assert len(initial_hashes) == 1
        assert all([item['sampling_sha256'] for item in histories[mode]] ==
                   [item['sampling_sha256'] for item in histories['predicted']] for mode in MODES)
        reference = records['predicted']['selected_validation']['rmse_rad']
        for mode in MODES:
            records[mode]['relative_rmse_reduction_vs_adapted_predicted'] = 1-records[mode]['selected_validation']['rmse_rad']/reference
            records[mode]['paired_vs_adapted_predicted'] = episode_bootstrap(
                errors[mode][:, 0], baseline['counts'][:, 0], errors['predicted'][:, 0], baseline['episode_routes'])
            records[mode]['paired_first15_vs_adapted_predicted'] = episode_bootstrap(
                errors[mode][:, 1], baseline['counts'][:, 1], errors['predicted'][:, 1], baseline['episode_routes'])
        all_results[str(seed)] = records
        write_json(root / 'adapted_head_results.json', all_results)
    write_json(root / 'adapted_head_audit.json', {'passed': True, 'seeds': [20261002, 20261003],
               'conditions_per_seed': MODES, 'epochs_per_condition': 10,
               'identical_initial_action_head': True, 'identical_sample_order_within_seed': True,
               'updates_use_train_partition_only': True, 'test_set_used': False,
               'frozen_map_predictor': True,
               'source_sha256': {str(path.relative_to(Path('/workspace'))): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), Path('/workspace/scripts/adapt_pi3_action_head.py'),
                                              Path('/workspace/scripts/cache_pi3_training_maps.py')]}})
    write_json(root / 'diagnostic_status.json', {'state': 'complete'})
    print('All matched adaptation conditions completed and audited.', flush=True)


if __name__ == '__main__':
    main()
