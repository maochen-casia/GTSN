"""Run matched-control, independent and joint replacement trials on four GPUs."""
import argparse
import copy
import gc
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import read_json, write_json
from tsn.evaluation.metrics import summarize_rollouts
from run_geometry_study import digest, source_hashes, validate_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--recipe', choices=('original', 'calibrated'), default='original')
    args = parser.parse_args()
    root = args.root
    source = root/'source'
    source.mkdir()
    for name in ('src', 'scripts', 'tests', 'configs', 'assets', 'vendor'):
        shutil.copytree(Path('/workspace')/name, source/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    write_json(root/'source_sha256.json', source_hashes(source))
    environment = {**os.environ, 'PYTHONPATH': f'{source}/src:{source}/vendor/Pi3'}
    parent_root = Path('/run/user/1016/experiments/gtsn_cam_var_20261008')
    parent_path = parent_root/'full/training/best.pt'
    saved = load_checkpoint(parent_path)
    base, splits = saved['config'], saved['splits']
    del saved; gc.collect()
    reference = {p: read_json(parent_root/f'full/{p}/closed_loop.json') for p in ('validation', 'test')}
    baseline = {p: sum(r['success'] for r in reference[p]['results']) for p in reference}
    specifications = dict(control=(False, False, 2), c1_d2=(True, False, 2),
                          c2_d2=(False, True, 2), joint_d4=(True, True, 4))
    configurations = {}
    for name, (c1, c2, depth) in specifications.items():
        config = copy.deepcopy(base)
        config['model']['perception']['freeze_encoder'] = True
        config['model']['learned_geometry'] = dict(mode='replacement', c1=c1, c2=c2, width=64, depth=depth)
        config['train'].update(seed=20261009, epochs=3, batch_size=4, draws_per_epoch=1024,
            num_workers=2, learning_rate=1e-5, module_learning_rate=1e-4,
            initialize_from=str(parent_path), geometry_only=False, gpu_memory_fraction=.17)
        if args.recipe == 'calibrated':
            config['model']['learned_geometry'].update(query_source='recent', query_capacity=256, calibrated_risk=True)
            config['train'].update(epochs=2, learning_rate=1e-6, module_learning_rate=3e-5)
            config['train']['loss_weights']['trust'] = .001
        if c1:config['train']['loss_weights']['memory'] = .05
        if c2:config['train']['loss_weights']['embodiment'] = 1. if args.recipe == 'calibrated' else .2
        if c1 or c2:config['train']['module_initialization'] = str(root/name/'warmup/modules.pt')
        configurations[name] = config
        write_json(root/f'configs/{name}.json', config)
    write_json(root/'splits.json', splits)
    write_json(root/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        parent_checkpoint=str(parent_path), parent_sha256=digest(parent_path), image_id=args.image_id,
        trials=specifications, warmup_epochs=2, warmup_cache_partition='train',
        recipe=args.recipe, prior_study_test_inspected=args.recipe == 'calibrated',
        training_epochs=2 if args.recipe == 'calibrated' else 3,
        warmup_cache='/run/user/1016/experiments/learned_geometry_20261009/cache',
        fine_tuning='raw RGB, all existing heads and new modules; only Pi3 encoder frozen',
        epoch_selection='lowest validation waypoint RMSE', trial_selection='full validation closed-loop success',
        primary_metric='collision-free XYZ reaching within 10 mm', reference_successes=baseline,
        validation_gate='at least both parent and matched control', test_gate='at least parent and matched control',
        single_seed=True, benchmark_previously_inspected=True, test_used_for_selection=False))

    def status(phase, **fields):
        write_json(root/'status.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields))

    def spawn(command, log, slot):
        stream = log.open('w')
        return subprocess.Popen(command, env={**environment, 'CUDA_VISIBLE_DEVICES': str(slot)},
                                stdout=stream, stderr=subprocess.STDOUT), stream

    # Numerical tests and every training/evaluation job run inside this Docker.
    status('testing')
    with (root/'tests.log').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(source/'tests'), '-v'],
                       env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    # Check the actual control's memory footprint before adding parallel jobs.
    running, pending = {}, list(specifications)
    status('training')
    while pending or running:
        for slot in range(4):
            if slot in running or not pending:continue
            if running and 'control' in pending:continue
            if pending[0] != 'control' and not (root/'control/training/gradient_audit.json').exists():continue
            name = pending.pop(0)
            if name == 'control':
                stage = 'training'
                command = [sys.executable, '-m', 'tsn.cli.train', '--config', str(root/f'configs/{name}.json'),
                           '--output-dir', str(root/name/'training')]
            else:
                stage = 'warmup'
                command = [sys.executable, str(source/'scripts/warmup_geometry_replacement.py'),
                    '--config', str(root/f'configs/{name}.json'), '--cache',
                    '/run/user/1016/experiments/learned_geometry_20261009/cache',
                    '--epochs', '2', '--output-dir', str(root/name/'warmup')]
            process, stream = spawn(command, root/f'{name}_{stage}.log', slot)
            running[slot] = (process, stream, name, stage)
        for slot, (process, stream, name, stage) in list(running.items()):
            code = process.poll()
            if code is None:continue
            stream.close()
            if code:raise RuntimeError(f'{name} {stage} failed; inspect {name}_{stage}.log')
            if stage == 'warmup':
                process, stream = spawn([sys.executable, '-m', 'tsn.cli.train', '--config', str(root/f'configs/{name}.json'),
                    '--output-dir', str(root/name/'training')], root/f'{name}_training.log', slot)
                running[slot] = (process, stream, name, 'training')
            else:del running[slot]
        status('training', jobs={str(s): dict(trial=n, stage=t) for s, (_, _, n, t) in running.items()}, pending=pending)
        if running:time.sleep(3)

    def evaluate(jobs, partition):
        queue, active = [], {}
        results = {name: {} for name in jobs}
        for name, path in jobs.items():
            destination = root/name/partition; destination.mkdir(parents=True)
            write_json(destination/'config.json', dict(checkpoint=str(path), sha256=digest(path),
                        partition=partition, episodes=splits[partition], eval=base['eval']))
            for number, offset in enumerate(range(0, len(splits[partition]), 5)):
                queue.append((name, path, splits[partition][offset:offset+5], number))
        while queue or active:
            for slot in range(4):
                if slot in active or not queue:continue
                name, path, ids, number = queue.pop(0)
                output = root/name/'batches'/partition/f'{number:03d}'
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(source/'scripts/evaluate_geometry_batch.py'),
                    '--checkpoint', str(path), '--partition', partition, '--output-dir', str(output), '--episodes', *ids]
                process, stream = spawn(command, output.with_suffix('.log'), slot)
                active[slot] = (process, stream, name, ids, output)
            for slot, (process, stream, name, ids, output) in list(active.items()):
                code = process.poll()
                if code is None:continue
                stream.close()
                if code:raise RuntimeError(f'Evaluation failed: {output}.log')
                for episode in ids:
                    directory = output/'episodes'/episode
                    results[name][episode] = validate_trace(directory, episode)
                    destination = root/name/partition/'episodes'/episode
                    destination.parent.mkdir(exist_ok=True)
                    shutil.move(str(directory), destination)
                rows = [results[name][i] for i in splits[partition] if i in results[name]]
                write_json(root/name/partition/'closed_loop.json', {**summarize_rollouts(rows), 'results': rows})
                del active[slot]
            status(f'evaluating_{partition}', completed_episodes={k: len(v) for k, v in results.items()})
            if active:time.sleep(3)
        for name, rows in results.items():
            assert set(rows) == set(splits[partition])
            write_json(root/name/partition/'complete.json', dict(episodes=len(rows), partition=partition))
        return {name: read_json(root/name/partition/'closed_loop.json') for name in jobs}

    paths = {n: root/n/'training/best.pt' for n in specifications}
    validation = evaluate(paths, 'validation')
    counts = {n: sum(r['success'] for r in v['results']) for n, v in validation.items()}
    gate = max(baseline['validation'], counts['control'])
    nominees = {n: p for n, p in paths.items() if n != 'control' and counts[n] >= gate}
    preferred_nominee = max(nominees, key=lambda n: (specifications[n][0]+specifications[n][1], counts[n],
                                                   -specifications[n][2], n)) if nominees else None
    # Export a concrete nomination before reading any new test result.
    write_json(root/'test_freeze.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        nominees={n: dict(checkpoint=str(p), sha256=digest(p), validation_successes=counts[n]) for n, p in nominees.items()},
        preferred_nominee=preferred_nominee,
        test_used_for_selection=False))
    test = evaluate({'control': paths['control'], **nominees}, 'test')
    test_counts = {n: sum(r['success'] for r in v['results']) for n, v in test.items()}
    test_gate = max(baseline['test'], test_counts['control'])
    accepted = [n for n in nominees if test_counts[n] >= test_gate]
    preferred = preferred_nominee if preferred_nominee in accepted else None
    # Audit every trained model against the same parent, one checkpoint at a
    # time to keep CPU memory bounded. Check all encoder tensors exactly.
    status('auditing')
    parent = load_checkpoint(parent_path)['model']
    audits = {}
    for name, path in paths.items():
        candidate = load_checkpoint(path)['model']
        frozen = [k for k in parent if k.startswith('perception.encoder.')]
        changed = [k for k in parent if not torch.equal(parent[k], candidate[k])]
        prefixes = ('perception.decoder.', 'perception.point_head.', 'perception.goal_head.',
                    'perception.action_map_head.', 'perception.joint_head.', 'route.',
                    'clearance.error_head.', 'clearance.trust_head.')
        assert all(torch.equal(parent[k], candidate[k]) for k in frozen)
        assert all(any(k.startswith(prefix) for k in changed) for prefix in prefixes)
        initialization = read_json(root/name/'training/initialization.json')
        assert all(k.startswith('perception.encoder.') for k in initialization['frozen_parameter_names'])
        audits[name] = dict(encoder_tensors_unchanged=len(frozen), changed_parent_tensors=len(changed),
                           every_existing_head_changed=True, only_encoder_frozen=True,
                           trainable_parameters=initialization['trainable_parameters'], sha256=digest(path))
        del candidate; gc.collect()
    del parent; gc.collect()
    write_json(root/'audit.json', audits)
    report = dict(reference=baseline, validation_successes=counts, test_successes=test_counts,
                  accepted=accepted, preferred=preferred, validation_gate=gate, test_gate=test_gate,
                  test_used_for_selection=False, full_partitions=True, trials=specifications)
    write_json(root/'summary.json', report)
    for name in specifications:
        if name not in accepted and name != 'control':
            paths[name].unlink()
            report.setdefault('removed_failed_checkpoints', []).append(name)
    write_json(root/'summary.json', report)
    status('complete', **report)


if __name__ == '__main__':main()
