"""Resume immutable trained trials with more GPU rollout workers."""
import argparse
import gc
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
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(); root = args.root; source = root/'source'
    protocol = read_json(root/'protocol.json'); splits = read_json(root/'splits.json')
    specs = protocol['trials']; paths = {n: root/n/'training/best.pt' for n in specs}
    for name in specs:
        assert read_json(root/name/'training/complete.json')['epochs'] == protocol.get('training_epochs', 3)
    support = root/'completion_source'; support.mkdir(exist_ok=True)
    for name in ('resume_replacement_study.py', 'report_replacement_study.py', 'diagnose_geometry_replacement.py'):
        shutil.copy2(Path('/workspace/scripts')/name, support/name)
    write_json(root/'completion_source_sha256.json', source_hashes(support))
    saved = load_checkpoint(protocol['parent_checkpoint']); options = saved['config']['eval']
    del saved; gc.collect()
    environment = {**os.environ, 'PYTHONPATH': f'{source}/src:{source}/vendor/Pi3'}

    def status(phase, **fields):
        write_json(root/'status.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields))

    def evaluate(jobs, partition):
        results, queues, active = {}, {}, {}
        for name, path in jobs.items():
            destination = root/name/partition; destination.mkdir(parents=True, exist_ok=True)
            # Recover every complete episode, including a partially finished
            # five-episode process. Validate each trace before reuse.
            for folder in (root/name/'batches'/partition).glob('*/episodes/*'):
                if not (folder/'metrics.json').exists() or not (folder/'trajectory.npz').exists():continue
                try:validate_trace(folder, folder.name)
                except (AssertionError, ValueError, OSError):continue
                target = destination/'episodes'/folder.name
                if not target.exists():
                    target.parent.mkdir(exist_ok=True); shutil.move(str(folder), target)
            rows = {}
            for folder in (destination/'episodes').glob('*'):
                assert folder.name in splits[partition]
                rows[folder.name] = validate_trace(folder, folder.name)
            results[name] = rows
            missing = [i for i in splits[partition] if i not in rows]
            queues[name] = [missing[i:i+5] for i in range(0, len(missing), 5)]
            write_json(destination/'config.json', dict(checkpoint=str(path), sha256=digest(path), partition=partition,
                episodes=splits[partition], eval=options, resumed=True))
        number = 0
        while any(queues.values()) or active:
            for slot in range(args.workers):
                if slot in active or not any(queues.values()):continue
                # Fairly interleave conditions so every trial advances.
                name = min((n for n in queues if queues[n]), key=lambda n: (len(results[n]), n))
                ids = queues[name].pop(0); number += 1
                output = root/name/'batches'/partition/f'resumed_{number:04d}'
                output.parent.mkdir(parents=True, exist_ok=True)
                stream = output.with_suffix('.log').open('w')
                command = [sys.executable, str(source/'scripts/evaluate_geometry_batch.py'),
                    '--checkpoint', str(jobs[name]), '--partition', partition, '--output-dir', str(output), '--episodes', *ids]
                process = subprocess.Popen(command, env={**environment, 'CUDA_VISIBLE_DEVICES': str(slot)},
                    stdout=stream, stderr=subprocess.STDOUT)
                active[slot] = (process, stream, name, ids, output)
            for slot, (process, stream, name, ids, output) in list(active.items()):
                code = process.poll()
                if code is None:continue
                stream.close()
                if code:raise RuntimeError(f'Evaluation failed: {output}.log')
                for episode in ids:
                    folder = output/'episodes'/episode
                    results[name][episode] = validate_trace(folder, episode)
                    target = root/name/partition/'episodes'/episode
                    target.parent.mkdir(exist_ok=True); shutil.move(str(folder), target)
                del active[slot]
            for name, rows in results.items():
                ordered = [rows[i] for i in splits[partition] if i in rows]
                if ordered:write_json(root/name/partition/'closed_loop.json', {**summarize_rollouts(ordered), 'results': ordered})
            status(f'evaluating_{partition}', completed_episodes={n: len(v) for n, v in results.items()}, workers=args.workers)
            if active:time.sleep(3)
        for name, rows in results.items():
            assert set(rows) == set(splits[partition])
            write_json(root/name/partition/'complete.json', dict(episodes=len(rows), partition=partition))
        return {n: read_json(root/n/partition/'closed_loop.json') for n in jobs}

    write_json(root/'evaluation_resume.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        workers=args.workers, image_id=args.image_id, unchanged_checkpoints={n: digest(p) for n, p in paths.items()},
        numerical_source='source/', reuse='validated complete trajectories only'))
    validation = evaluate(paths, 'validation')
    counts = {n: sum(r['success'] for r in v['results']) for n, v in validation.items()}
    baseline = protocol['reference_successes']; gate = max(baseline['validation'], counts['control'])
    nominees = {n: p for n, p in paths.items() if n != 'control' and counts[n] >= gate}
    preferred = max(nominees, key=lambda n: (specs[n][0]+specs[n][1], counts[n], -specs[n][2], n)) if nominees else None
    write_json(root/'test_freeze.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        nominees={n: dict(checkpoint=str(p), sha256=digest(p), validation_successes=counts[n]) for n, p in nominees.items()},
        preferred_nominee=preferred, test_used_for_selection=False))
    write_json(root/'preferred_nominee.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), preferred=preferred,
        chosen_by='validation and requested joint-module coverage', test_used_for_selection=False))
    test = evaluate({'control': paths['control'], **nominees}, 'test')
    test_counts = {n: sum(r['success'] for r in v['results']) for n, v in test.items()}
    test_gate = max(baseline['test'], test_counts['control'])
    accepted = [n for n in nominees if test_counts[n] >= test_gate]
    status('auditing'); parent = load_checkpoint(protocol['parent_checkpoint'])['model']; audits = {}
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
    del parent; gc.collect(); write_json(root/'audit.json', audits)
    report = dict(reference=baseline, validation_successes=counts, test_successes=test_counts,
        accepted=accepted, preferred=preferred if preferred in accepted else None, preferred_nominee=preferred,
        validation_gate=gate, test_gate=test_gate, test_used_for_selection=False, full_partitions=True, trials=specs)
    for name in specs:
        if name not in accepted and name != 'control':
            paths[name].unlink(); report.setdefault('removed_failed_checkpoints', []).append(name)
    write_json(root/'summary.json', report)
    subprocess.run([sys.executable, str(support/'report_replacement_study.py'), '--root', str(root)], check=True)
    status('complete', **report)


if __name__ == '__main__':main()
