"""Freeze, train, select and audit independent C1/C2 learned-geometry updates."""
import argparse
import copy
import gc
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import read_json, write_json
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.metrics import summarize_rollouts


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def validate_trace(directory, episode):
    row = read_json(directory/'metrics.json')
    with np.load(directory/'trajectory.npz', allow_pickle=False) as trace:
        assert row['episode_id'] == episode
        assert len(trace['qpos']) == row['control_steps']+1
        assert len(trace['predicted_joint_targets']) == row['control_steps']
        assert not len(trace['reference_indices'])
        assert (trace['execution_horizons'] == 15).all()
        assert all(np.isfinite(trace[key]).all() for key in trace.files)
    assert not row['depth_input_at_inference'] and row['expert_progress_method'] is None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--reuse-cache', action='store_true')
    parser.add_argument('--main-root', type=Path, default=Path('/run/user/1016/experiments/gtsn_cam_var_20261008'))
    args = parser.parse_args()
    root = args.root
    if any(p.name not in ('pipeline.log', 'cache', 'cache.log') for p in root.iterdir()):
        parser.error('Existing study artifacts cannot be overwritten')
    source = root/'source'; source.mkdir()
    for name in ('src', 'scripts', 'tests', 'configs', 'assets', 'vendor'):
        shutil.copytree(Path('/workspace')/name, source/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in ('pyproject.toml', 'Dockerfile'):
        shutil.copy2(Path('/workspace')/name, source/name)
    write_json(root/'source_sha256.json', source_hashes(source))
    environment = {**os.environ, 'PYTHONPATH': f'{source}/src:{source}/vendor/Pi3'}
    checkpoint = args.main_root/'full/training/best.pt'
    parent = load_checkpoint(checkpoint)
    parent_digest = digest(checkpoint)
    splits = parent['splits']
    validate_splits(splits, episode_catalog(Path(parent['config']['benchmark']['root'])))
    reference = {p: read_json(args.main_root/f'full/{p}/closed_loop.json') for p in ('validation', 'test')}
    baseline = {p: sum(row['success'] for row in reference[p]['results']) for p in reference}
    for p in reference:
        assert {r['episode_id'] for r in reference[p]['results']} == set(splits[p])
        assert read_json(args.main_root/f'full/{p}/complete.json')['episodes'] == 100
    # The archived experiment is a descriptive reference on an already inspected
    # benchmark. Source/command parity and five replayed validation traces bind it.
    for name in ('src/tsn/models/perception.py', 'src/tsn/models/route.py',
                 'src/tsn/simulation/episode.py', 'src/tsn/simulation/robot.py',
                 'src/tsn/simulation/benchmark_scene.py', 'src/tsn/features/camera.py'):
        assert digest(source/name) == digest(args.main_root/'source'/name), name
    strengths = (.05, .2, .5)
    protocol = dict(parent_checkpoint=str(checkpoint), parent_sha256=parent_digest, image_id=args.image_id,
        seed=20261009, adapter_epochs=4, cached_training_draws=2048, strengths=strengths,
        components=['c1', 'c2'], parent_frozen=True, encoder_frozen=True,
        selection='highest full validation success; ties choose smaller strength',
        validation_gate='at least parent success count', test_gate='at least parent success count',
        combined_scales=[.5, 1.], combined_selection='same validation rule, tie chooses smaller scale',
        test_used_for_selection=False, primary_metric='collision-free XYZ reaching within 10 mm',
        reference_successes=baseline, benchmark_already_inspected=True,
        created_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'protocol.json', protocol)
    write_json(root/'splits.json', splits)
    del parent['model']

    def status(phase, **fields):
        write_json(root/'status.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields))

    def run(command, log, slot='0'):
        with (root/log).open('w') as stream:
            subprocess.run(command, env={**environment, 'CUDA_VISIBLE_DEVICES': slot}, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    def evaluate(jobs, partition):
        """Four GPU workers; fresh simulator processes every five episodes."""
        queue = []
        results = {name: {} for name in jobs}
        for name, (weights, strength, ids) in jobs.items():
            destination = root/name/partition
            destination.mkdir(parents=True)
            for i in range(0, len(ids), 5):
                queue.append((name, weights, strength, ids[i:i+5], i//5))
            write_json(destination/'config.json', dict(checkpoint=str(weights), checkpoint_sha256=digest(weights),
                partition=partition, episodes=ids, full_partition=set(ids) == set(splits[partition]),
                strength_override=strength, eval=parent['config']['eval']))
        active = {}
        try:
            while queue or active:
                for slot in ('0', '1', '2', '3'):
                    if slot in active or not queue:continue
                    name, weights, strength, ids, number = queue.pop(0)
                    output = root/name/'batches'/partition/f'{number:03d}'
                    output.parent.mkdir(parents=True, exist_ok=True)
                    stream = output.with_suffix('.log').open('w')
                    command = [sys.executable, str(source/'scripts/evaluate_geometry_batch.py'),
                        '--checkpoint', str(weights), '--partition', partition, '--output-dir', str(output), '--episodes', *ids]
                    if strength is not None:command += ['--strength', str(strength)]
                    process = subprocess.Popen(command, env={**environment, 'CUDA_VISIBLE_DEVICES': slot},
                                               stdout=stream, stderr=subprocess.STDOUT)
                    active[slot] = (process, stream, name, ids, output)
                for slot, (process, stream, name, ids, output) in list(active.items()):
                    code = process.poll()
                    if code is None:continue
                    stream.close()
                    if code:raise RuntimeError(f'Evaluation failed ({code}): {output}.log')
                    for episode in ids:
                        directory = output/'episodes'/episode
                        results[name][episode] = validate_trace(directory, episode)
                        target = root/name/partition/'episodes'/episode
                        target.parent.mkdir(exist_ok=True)
                        shutil.move(str(directory), target)
                    ordered = [results[name][i] for i in jobs[name][2] if i in results[name]]
                    write_json(root/name/partition/'closed_loop.json', {**summarize_rollouts(ordered), 'results': ordered})
                    del active[slot]
                status(f'evaluating_{partition}', completed_episodes={n: len(v) for n, v in results.items()})
                if active:time.sleep(2)
            summaries = {}
            for name, rows in results.items():
                assert set(rows) == set(jobs[name][2])
                summaries[name] = read_json(root/name/partition/'closed_loop.json')
                write_json(root/name/partition/'complete.json', dict(episodes=len(rows), partition=partition))
            return summaries
        finally:
            for process, stream, *_ in active.values():
                if process.poll() is None:process.terminate()
                process.wait(); stream.close()

    processes = []
    try:
        status('testing')
        run([sys.executable, '-m', 'unittest', 'discover', '-s', str(source/'tests'), '-v'], 'tests.log')
        if not args.reuse_cache:
            status('caching_training')
            run([sys.executable, str(source/'scripts/geometry_update.py'), 'cache', '--checkpoint', str(checkpoint),
                 '--output-dir', str(root/'cache')], 'cache.log')
        manifest = read_json(root/'cache/manifest.json')
        assert manifest['partition'] == 'train' and manifest['train_episodes'] == splits['train']
        assert manifest['parent_checkpoint'] == str(checkpoint) and manifest['examples'] == 2048
        write_json(root/'cache_sha256.json', dict(examples=digest(root/'cache/examples.pt'), parent=parent_digest))
        status('training_independent_modules')
        for component, slot in (('c1', '0'), ('c2', '1')):
            stream = (root/f'{component}_training.log').open('w')
            command = [sys.executable, str(source/'scripts/geometry_update.py'), 'train', '--checkpoint', str(checkpoint),
                '--cache', str(root/'cache'), '--component', component, '--output-dir', str(root/component/'training')]
            process = subprocess.Popen(command, env={**environment, 'CUDA_VISIBLE_DEVICES': slot},
                                       stdout=stream, stderr=subprocess.STDOUT)
            processes.append((process, stream, component))
        for process, stream, component in processes:
            if process.wait():raise RuntimeError(f'{component} training failed')
            stream.close()
        processes = []
        # Check replay before admitting archived reference scores.
        by_route = {route: [r['episode_id'] for r in reference['validation']['results'] if r['route'] == route]
                    for route in ('direct', 'over', 'side')}
        replay_ids = by_route['direct'][:1]+by_route['over'][:2]+by_route['side'][:2]
        replay = evaluate({'parent_replay': (checkpoint, None, replay_ids)}, 'validation')['parent_replay']
        original = {r['episode_id']: r for r in reference['validation']['results']}
        for row in replay['results']:
            before = original[row['episode_id']]
            assert row['success'] == before['success'] and row['control_steps'] == before['control_steps']
            assert abs(row['final_position_error_m']-before['final_position_error_m']) < 1e-6
        write_json(root/'reference_parity.json', dict(episodes=replay_ids, passed=True))
        jobs = {f'{c}/strength_{s:g}': (root/c/'training/trained.pt', s, splits['validation'])
                for c in ('c1', 'c2') for s in strengths}
        panel = evaluate(jobs, 'validation')
        nominees = {}
        for component in ('c1', 'c2'):
            best_strength = max(strengths, key=lambda s: (sum(r['success'] for r in panel[f'{component}/strength_{s:g}']['results']), -s))
            condition = f'{component}/strength_{best_strength:g}'
            successes = sum(r['success'] for r in panel[condition]['results'])
            record = dict(strength=best_strength, condition=condition, validation_successes=successes,
                          passes_validation_gate=successes >= baseline['validation'])
            if record['passes_validation_gate']:
                trained = load_checkpoint(root/component/'training/trained.pt')
                trained['config']['model']['learned_geometry'][component+'_strength'] = best_strength
                trained['selected_by'] = 'full validation closed-loop success'
                trained['validation_successes'] = successes
                path = root/component/'selected.pt'; save_checkpoint(path, trained)
                record.update(checkpoint=str(path), sha256=digest(path))
                nominees[component] = record
                del trained
            write_json(root/component/'selection.json', record)
        joint = {}
        if len(nominees) == 2:
            combined = load_checkpoint(root/'c1/selected.pt')
            c2 = load_checkpoint(root/'c2/selected.pt')
            combined['config']['model']['learned_geometry'].update(c2=True, c2_strength=nominees['c2']['strength'])
            for name, value in c2['model'].items():
                if name.startswith('embodiment.'):combined['model'][name] = value
            combined['component'] = 'c1_c2'
            combined['geometry_update']['independent_adapters'] = nominees
            (root/'combined').mkdir()
            for scale in (.5, 1.):
                candidate = copy.deepcopy(combined)
                for c in ('c1', 'c2'):candidate['config']['model']['learned_geometry'][c+'_strength'] *= scale
                save_checkpoint(root/f'combined/scale_{scale:g}.pt', candidate)
            del combined, c2, candidate
            gc.collect()
            joint_panel = evaluate({f'combined/scale_{s:g}': (root/f'combined/scale_{s:g}.pt', None, splits['validation'])
                                    for s in (.5, 1.)}, 'validation')
            scale = max((.5, 1.), key=lambda s: (sum(r['success'] for r in joint_panel[f'combined/scale_{s:g}']['results']), -s))
            condition = f'combined/scale_{scale:g}'
            successes = sum(r['success'] for r in joint_panel[condition]['results'])
            joint = dict(scale=scale, validation_successes=successes, condition=condition,
                         passes_validation_gate=successes >= baseline['validation'])
            if joint['passes_validation_gate']:
                path = root/'combined/selected.pt'
                run([sys.executable, str(source/'scripts/export_geometry_selection.py'),
                     '--input', str(root/f'combined/scale_{scale:g}.pt'), '--output', str(path),
                     '--validation-successes', str(successes)], 'combined/export_selection.log')
                joint.update(checkpoint=str(path), sha256=digest(path))
                nominees['combined'] = joint
            write_json(root/'combined/selection.json', joint)
            for s in (.5, 1.):(root/f'combined/scale_{s:g}.pt').unlink()
        # Immutable nomination precedes ALL updated-policy test rollouts.
        write_json(root/'test_freeze.json', dict(nominees=nominees, test_used_for_selection=False,
                                               created_utc=datetime.now(timezone.utc).isoformat()))
        tested = evaluate({c: (Path(n['checkpoint']), None, splits['test']) for c, n in nominees.items()}, 'test') if nominees else {}
        report = dict(reference=baseline, models={}, validation_panel={
            name: sum(r['success'] for r in result['results']) for name, result in panel.items()},
            test_used_for_selection=False, full_partitions=True, single_training_seed=True)
        for c, result in tested.items():
            n = nominees[c]
            successes = sum(r['success'] for r in result['results'])
            accepted = successes >= baseline['test'] and n['passes_validation_gate']
            report['models'][c] = dict(validation_successes=n['validation_successes'], test_successes=successes,
                accepted=accepted, checkpoint=n['checkpoint'], sha256=n['sha256'], strengths=n,
                test_summary={k: v for k, v in result.items() if k != 'results'},
                paired_test_gains=sum(r['success'] and not b['success'] for r, b in zip(result['results'], reference['test']['results'])),
                paired_test_losses=sum(b['success'] and not r['success'] for r, b in zip(result['results'], reference['test']['results'])))
            assert digest(n['checkpoint']) == n['sha256']
        assert digest(checkpoint) == parent_digest
        assert source_hashes(source) == read_json(root/'source_sha256.json')
        for c in ('c1', 'c2'):
            # Keep small, reproducible adapter weights and the frozen parent
            # reference; remove redundant complete trial checkpoints.
            run([sys.executable, str(source/'scripts/export_geometry_selection.py'),
                 '--input', str(root/c/'training/trained.pt'), '--parent', str(checkpoint),
                 '--output', str(root/c/'training/adapter.pt'), '--compact'], f'{c}/export_adapter.log')
            (root/c/'training/trained.pt').unlink()
        write_json(root/'summary.json', report)
        lines = ['# Learned C1/C2 geometry update', '',
            'All parent policy tensors, including the previously trained Pi3 backbone, were frozen.',
            'C1 and C2 were trained independently on 2,048 causal training-only expert/perturbation examples.',
            'Full validation success selected strength before any updated-policy test rollout.', '',
            '| Model | Validation /100 | Test /100 | Accepted without observed success drop |',
            '|---|---:|---:|---|', f'| Frozen parent | {baseline["validation"]} | {baseline["test"]} | Reference |']
        for c, row in report['models'].items():
            lines.append(f'| {c} | {row["validation_successes"]} | {row["test_successes"]} | {row["accepted"]} |')
        lines += ['', 'Acceptance compares observed collision-free XYZ success counts on these existing 100-scene partitions.',
                  'This single-seed study does not establish population noninferiority or real-robot transfer.',
                  'All rollouts use RGB and measured state; depth and expert futures appear only in training.',
                  'Protocol, frozen source, cache provenance, nomination, logs and finite trajectories are retained.', '']
        (root/'RESULTS.md').write_text('\n'.join(lines))
        status('complete', accepted_models=[c for c, r in report['models'].items() if r['accepted']])
    except Exception as error:
        for process, stream, _ in processes:
            if process.poll() is None:process.terminate()
            process.wait(); stream.close()
        status('failed', error=str(error))
        raise


if __name__ == '__main__':main()
