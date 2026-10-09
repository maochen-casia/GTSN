"""Recover held-out rollouts in isolated batches using immutable selected weights.

Completed episodes are admitted only with matching metadata and finite complete
trajectories. Partial episodes are archived and rerun. Fresh subprocesses release
native simulator/renderer allocations every five episodes. Numerical policy,
simulation, seed, thresholds and checkpoint selection remain the frozen source's.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from tsn.common.config import read_json, write_json
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.metrics import summarize_rollouts

METHODS = ('diffusion_policy', 'dp3', 'flowpolicy', 'carp')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def valid_episode(directory, episode, method):
    try:
        row = read_json(directory/'metrics.json')
        if row['episode_id'] != episode or row['baseline_method'] != method:
            return None
        with np.load(directory/'trajectory.npz', allow_pickle=False) as trace:
            if (len(trace['qpos']) != row['control_steps']+1 or
                len(trace['predicted_joint_targets']) != row['control_steps'] or
                len(trace['reference_indices']) or not (trace['execution_horizons'] == 15).all() or
                not all(np.isfinite(trace[key]).all() for key in trace.files)):
                return None
        return row
    except (OSError, ValueError, KeyError, EOFError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu-slots', default='0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--script-root', type=Path, default=Path('/workspace/scripts'))
    args = parser.parse_args()
    root = args.root
    slots = args.gpu_slots.split(',')
    if len(slots) != 4 or len(set(slots)) != 4 or not 1 <= args.batch_size <= 5:
        parser.error('Use four distinct GPU slots and batches of at most five episodes')
    source = root/'source'
    environment = {**os.environ, 'PYTHONPATH': f'{source}/src:{source}/vendor/Pi3',
                   'PYTHONDONTWRITEBYTECODE': '1'}
    config = read_json(root/'base_config.json')
    splits = read_json(root/'cache/splits.json')
    validate_splits(splits, episode_catalog(Path(config['benchmark']['root'])))
    archive = root/'evaluation_recovery'
    archive.mkdir(exist_ok=False)
    shutil.copy2(root/'status.json', archive/'previous_status.json')
    shutil.copy2(root/'pipeline.log', archive/'previous_pipeline.log')
    partition_rows, pending, reuse, digests = {}, {}, {}, {}
    for method in METHODS:
        selected = root/method/'training/best.pt'
        digests[method] = sha256(selected)
        receipt = read_json(root/method/'training/selected_checkpoint.json')
        assert receipt['sha256'] == digests[method] and not receipt['test_used_for_selection']
        assert read_json(root/method/'training/complete.json')['epochs'] == 30
        pending[method] = []
        reuse[method] = {}
        for partition in ('validation', 'test'):
            output = root/method/partition
            output.mkdir(exist_ok=True)
            (output/'episodes').mkdir(exist_ok=True)
            settings = {'checkpoint': str(selected), 'checkpoint_sha256': digests[method],
                'partition': partition, 'episodes': splits[partition],
                'dataset_root': config['benchmark']['root'], 'eval': config['eval'],
                'full_partition': True, 'method': method, 'depth_input_at_inference': method in ('dp3', 'flowpolicy')}
            if (output/'config.json').exists():
                assert read_json(output/'config.json') == settings, ('Evaluation settings changed', method, partition)
            else:
                write_json(output/'config.json', settings)
                write_json(output/'splits.json', splits)
            rows = {}
            missing = []
            for episode in splits[partition]:
                directory = output/'episodes'/episode
                row = valid_episode(directory, episode, method)
                if row is not None:
                    rows[episode] = row
                else:
                    if directory.exists():
                        destination = archive/'incomplete'/method/partition/episode
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(directory), destination)
                    missing.append(episode)
            partition_rows[method, partition] = rows
            reuse[method][partition] = len(rows)
            for i in range(0, len(missing), args.batch_size):
                pending[method].append((partition, missing[i:i+args.batch_size]))
    write_json(archive/'protocol.json', {'reason': 'Bound native simulator memory with isolated evaluation processes',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'reused_episodes': reuse,
        'checkpoint_sha256': digests, 'batch_size': args.batch_size,
        'numerical_source': str(source), 'policy_or_simulation_changes': False,
        'checkpoint_reselected': False, 'test_used_for_tuning': False})
    active, counters = {}, {name: 0 for name in METHODS}

    def persist(method, partition):
        rows = [partition_rows[method, partition][episode] for episode in splits[partition]
                if episode in partition_rows[method, partition]]
        if rows:
            write_json(root/method/partition/'closed_loop.json', {**summarize_rollouts(rows), 'results': rows})
        if len(rows) == len(splits[partition]):
            write_json(root/method/partition/'complete.json', {'episodes': len(rows), 'partition': partition,
                'architecture': 'tsn_baseline', 'method': method})

    for method in METHODS:
        for partition in ('validation', 'test'):
            persist(method, partition)
    try:
        while any(pending.values()) or active:
            for method, slot in zip(METHODS, slots):
                if method in active or not pending[method]:
                    continue
                partition, ids = pending[method].pop(0)
                counters[method] += 1
                batch = root/'evaluation_batches'/method/partition/f'batch_{counters[method]:03d}'
                batch.parent.mkdir(parents=True, exist_ok=True)
                stream = (batch.parent/f'{batch.name}.log').open('w')
                command = [sys.executable, str(args.script_root/'evaluate_baseline_batch.py'),
                    '--checkpoint', str(root/method/'training/best.pt'), '--partition', partition,
                    '--output-dir', str(batch), '--episodes', *ids]
                process = subprocess.Popen(command, env={**environment, 'CUDA_VISIBLE_DEVICES': slot},
                    stdout=stream, stderr=subprocess.STDOUT)
                active[method] = (process, stream, partition, ids, batch)
            for method, (process, stream, partition, ids, batch) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                if code != 0:
                    raise RuntimeError(f'{method} {partition} batch failed ({code}): {batch}')
                batch_config = read_json(batch/'config.json')
                assert batch_config['checkpoint_sha256'] == digests[method]
                assert batch_config['episodes'] == ids and batch_config['eval'] == config['eval']
                for episode in ids:
                    directory = batch/'episodes'/episode
                    row = valid_episode(directory, episode, method)
                    if row is None:
                        raise RuntimeError(f'Incomplete recovered episode {method}/{partition}/{episode}')
                    destination = root/method/partition/'episodes'/episode
                    assert not destination.exists()
                    shutil.move(str(directory), destination)
                    partition_rows[method, partition][episode] = row
                persist(method, partition)
                del active[method]
            write_json(root/'status.json', {'phase': 'evaluating_isolated_batches',
                'updated_utc': datetime.now(timezone.utc).isoformat(),
                'completed_episodes': {method: {p: len(partition_rows[method, p]) for p in ('validation', 'test')}
                                       for method in METHODS}})
            if active:
                time.sleep(5)
        for method in METHODS:
            assert sha256(root/method/'training/best.pt') == digests[method]
            write_json(root/method/'status.json', {'phase': 'complete'})
        write_json(root/'status.json', {'phase': 'auditing', 'updated_utc': datetime.now(timezone.utc).isoformat()})
        with (root/'audit.log').open('w') as stream:
            subprocess.run([sys.executable, str(source/'scripts/report_baselines.py'), '--root', str(root)],
                env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        summary = read_json(root/'summary.json')
        write_json(root/'status.json', {'phase': 'complete', 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'evaluation_recovered': True, 'test_successes': {m: summary['models'][m]['test']['successes'] for m in METHODS}})
    except Exception as error:
        for process, stream, *_ in active.values():
            if process.poll() is None:
                process.terminate()
            process.wait()
            stream.close()
        write_json(root/'status.json', {'phase': 'failed', 'updated_utc': datetime.now(timezone.utc).isoformat(),
                                      'error': str(error)})
        raise


if __name__ == '__main__':
    main()
