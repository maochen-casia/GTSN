"""Resume an interrupted geometry study from audited trajectories and fixed weights.

The supervisor holds no model tensors while two isolated GPU workers evaluate
five-episode batches. Numerical code comes from the original frozen source.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tsn.common.config import read_json, write_json
from tsn.evaluation.metrics import summarize_rollouts
from run_geometry_study import digest, source_hashes, validate_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu-slots', default='0,1')
    args = parser.parse_args()
    root, source = args.root, args.root/'source'
    protocol = read_json(root/'protocol.json')
    if source_hashes(source) != read_json(root/'source_sha256.json'):raise ValueError('Frozen source changed')
    checkpoint = Path(protocol['parent_checkpoint'])
    if digest(checkpoint) != protocol['parent_sha256']:raise ValueError('Frozen parent changed')
    splits = read_json(root/'splits.json')
    config = read_json(checkpoint.parent/'config.json')
    slots = args.gpu_slots.split(',')
    if len(slots) != 2 or len(set(slots)) != 2:parser.error('Use two distinct GPU slots')
    environment = {**os.environ, 'PYTHONPATH': f'{source}/src:{source}/vendor/Pi3'}
    recovery = root/'evaluation_recovery'; recovery.mkdir()
    shutil.copy2(root/'status.json', recovery/'previous_status.json')
    write_json(recovery/'protocol.json', dict(reason='Original container exceeded its RAM limit',
        numerical_source=str(source), numerical_policy_changed=False, adapters_retrained=False,
        workers=2, batch_size=5, parent_sha256=protocol['parent_sha256'],
        created_utc=datetime.now(timezone.utc).isoformat(), test_used_for_selection=False))

    def status(phase, **fields):
        write_json(root/'status.json', dict(phase=phase, recovery=True,
            updated_utc=datetime.now(timezone.utc).isoformat(), **fields))

    def evaluate(jobs, partition):
        pending, rows, active = [], {}, {}
        for name, weights in jobs.items():
            destination = root/name/partition
            destination.mkdir(parents=True, exist_ok=True)
            episodes = destination/'episodes'; episodes.mkdir(exist_ok=True)
            expected = dict(checkpoint=str(weights), checkpoint_sha256=digest(weights), partition=partition,
                episodes=splits[partition], full_partition=True, strength_override=None, eval=config['eval'])
            if (destination/'config.json').exists():
                if read_json(destination/'config.json') != expected:raise ValueError(f'Evaluation settings changed: {name}')
            else:write_json(destination/'config.json', expected)
            rows[name] = {}
            directories = list(episodes.iterdir())
            for batch in (root/name/'batches'/partition).glob('*'):
                if batch.is_dir():directories += list((batch/'episodes').glob('*'))
            for directory in directories:
                episode = directory.name
                if episode not in splits[partition]:raise ValueError('Recovered episode is outside partition')
                try:row = validate_trace(directory, episode)
                except (OSError, ValueError, KeyError, AssertionError):
                    shutil.rmtree(directory)
                    continue
                target = episodes/episode
                if episode in rows[name]:
                    if row != rows[name][episode]:raise ValueError('Duplicate recovered results disagree')
                    if directory != target:shutil.rmtree(directory)
                    continue
                if directory != target:shutil.move(str(directory), target)
                rows[name][episode] = row
            missing = [episode for episode in splits[partition] if episode not in rows[name]]
            write_json(recovery/(name.replace('/', '_')+'_'+partition+'.json'),
                       dict(reused=len(rows[name]), remaining=len(missing)))
            for i in range(0, len(missing), 5):pending.append((name, weights, missing[i:i+5], i//5))
        try:
            while pending or active:
                for slot in slots:
                    if slot in active or not pending:continue
                    name, weights, ids, number = pending.pop(0)
                    batch = root/'evaluation_recovery_batches'/name/partition/f'{number:03d}'
                    batch.parent.mkdir(parents=True, exist_ok=True)
                    stream = batch.with_suffix('.log').open('w')
                    command = [sys.executable, str(source/'scripts/evaluate_geometry_batch.py'),
                        '--checkpoint', str(weights), '--partition', partition, '--output-dir', str(batch), '--episodes', *ids]
                    process = subprocess.Popen(command, env={**environment, 'CUDA_VISIBLE_DEVICES': slot},
                                               stdout=stream, stderr=subprocess.STDOUT)
                    active[slot] = (process, stream, name, ids, batch)
                for slot, (process, stream, name, ids, batch) in list(active.items()):
                    code = process.poll()
                    if code is None:continue
                    stream.close()
                    if code:raise RuntimeError(f'Failed recovered batch ({code}): {batch}.log')
                    for episode in ids:
                        directory = batch/'episodes'/episode
                        rows[name][episode] = validate_trace(directory, episode)
                        shutil.move(str(directory), root/name/partition/'episodes'/episode)
                    ordered = [rows[name][episode] for episode in splits[partition] if episode in rows[name]]
                    write_json(root/name/partition/'closed_loop.json', {**summarize_rollouts(ordered), 'results': ordered})
                    del active[slot]
                status(f'recovering_{partition}', completed_episodes={name: len(values) for name, values in rows.items()})
                if active:time.sleep(2)
            for name, values in rows.items():
                ordered = [values[episode] for episode in splits[partition]]
                write_json(root/name/partition/'closed_loop.json', {**summarize_rollouts(ordered), 'results': ordered})
                write_json(root/name/partition/'complete.json', dict(episodes=len(ordered), partition=partition))
            return {name: read_json(root/name/partition/'closed_loop.json') for name in jobs}
        finally:
            for process, stream, *_ in active.values():
                if process.poll() is None:process.terminate()
                process.wait(); stream.close()

    try:
        nominees = {c: read_json(root/c/'selection.json') for c in ('c1', 'c2')}
        for c, n in nominees.items():
            if not n['passes_validation_gate'] or digest(n['checkpoint']) != n['sha256']:
                raise ValueError('Independent validation nomination changed')
        panel = evaluate({f'combined/scale_{s:g}': root/f'combined/scale_{s:g}.pt' for s in (.5, 1.)}, 'validation')
        scale = max((.5, 1.), key=lambda s: (sum(r['success'] for r in panel[f'combined/scale_{s:g}']['results']), -s))
        condition = f'combined/scale_{scale:g}'
        successes = sum(r['success'] for r in panel[condition]['results'])
        joint = dict(scale=scale, condition=condition, validation_successes=successes,
                     passes_validation_gate=successes >= protocol['reference_successes']['validation'])
        if joint['passes_validation_gate']:
            selected = root/'combined/selected.pt'
            # Export in a short child process; no full-model tensors remain in
            # the supervisor when evaluation workers launch.
            command = [sys.executable, '/workspace/scripts/export_geometry_selection.py',
                '--input', str(root/f'combined/scale_{scale:g}.pt'), '--output', str(selected),
                '--validation-successes', str(successes)]
            subprocess.run(command, env=environment, check=True)
            joint.update(checkpoint=str(selected), sha256=digest(selected))
            nominees['combined'] = joint
        write_json(root/'combined/selection.json', joint)
        for s in (.5, 1.):(root/f'combined/scale_{s:g}.pt').unlink()
        write_json(root/'test_freeze.json', dict(nominees=nominees, test_used_for_selection=False,
                                               created_utc=datetime.now(timezone.utc).isoformat()))
        tested = evaluate({c: Path(n['checkpoint']) for c, n in nominees.items()}, 'test')
        reference = {p: read_json(checkpoint.parents[1]/p/'closed_loop.json')['results'] for p in ('validation', 'test')}
        summary = dict(reference=protocol['reference_successes'], models={}, validation_panel={},
                       test_used_for_selection=False, full_partitions=True, single_training_seed=True)
        for p in root.glob('c*/strength_*/validation/closed_loop.json'):
            summary['validation_panel'][str(p.parent.parent.relative_to(root))] = sum(r['success'] for r in read_json(p)['results'])
        for c, result in tested.items():
            n = nominees[c]
            successes = sum(r['success'] for r in result['results'])
            summary['models'][c] = dict(validation_successes=n['validation_successes'], test_successes=successes,
                accepted=successes >= protocol['reference_successes']['test'], checkpoint=n['checkpoint'],
                sha256=n['sha256'], strengths=n, test_summary={k: v for k, v in result.items() if k != 'results'})
            if digest(n['checkpoint']) != n['sha256']:raise ValueError('Frozen test nominee changed')
        if digest(checkpoint) != protocol['parent_sha256']:raise ValueError('Frozen parent changed')
        for c in ('c1', 'c2'):
            subprocess.run([sys.executable, '/workspace/scripts/export_geometry_selection.py',
                '--input', str(root/c/'training/trained.pt'), '--parent', str(checkpoint),
                '--output', str(root/c/'training/adapter.pt'), '--compact'], env=environment, check=True)
            (root/c/'training/trained.pt').unlink()
        write_json(root/'summary.json', summary)
        lines = ['# Learned C1/C2 geometry update', '',
            'Only new adapters were trained; Pi3 and every parent policy tensor stayed frozen.',
            'Strengths were selected on full validation before any updated-policy test rollout.', '',
            '| Model | Validation /100 | Test /100 | Accepted without observed success drop |',
            '|---|---:|---:|---|', '| Frozen parent | 68 | 69 | Reference |']
        for c, row in summary['models'].items():
            lines.append(f'| {c} | {row["validation_successes"]} | {row["test_successes"]} | {row["accepted"]} |')
        lines += ['', 'These descriptive single-seed results use the existing 100-scene partitions.',
            'Observed no-drop acceptance does not establish population noninferiority.',
            'The container RAM interruption was recovered using unchanged frozen source and audited completed episodes.', '']
        (root/'RESULTS.md').write_text('\n'.join(lines))
        status('complete', accepted_models=[c for c, r in summary['models'].items() if r['accepted']])
    except Exception as error:
        status('failed_recovery', error=str(error))
        raise


if __name__ == '__main__':main()
