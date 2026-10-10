"""Validation-only hard-query budget comparison for already trained replacements."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import read_json, write_json
from tsn.evaluation.metrics import summarize_rollouts
from run_geometry_study import digest, validate_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    args = parser.parse_args(); root, base = args.root, args.base_root; source = base/'source'
    splits = read_json(base/'splits.json'); protocol = read_json(base/'protocol.json')
    names = ('c1_d2', 'joint_d4'); paths = {}
    provenance = {}
    for name in names:
        original = base/name/'training/best.pt'
        model = load_checkpoint(original)
        provenance[name] = dict(original_checkpoint=str(original), sha256=digest(original),
            trained_query_budget=256, candidate_query_budget=512, parameters_unchanged=True)
        model['config']['model']['learned_geometry']['query_capacity'] = 512
        model['query_budget_selection'] = provenance[name]
        (root/name).mkdir(); path = root/name/'candidate.pt'
        save_checkpoint(path, model); paths[name] = path
        del model; gc.collect()
    write_json(root/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), image_id=args.image_id,
        base_study=str(base), numerical_source=str(source), query_budgets=[256, 512],
        provenance=provenance, selection='highest full validation success; ties prefer budget 256',
        validation_gate='at least parent and matched control', test_used_for_selection=False,
        prior_batch_test_inspected=True, weights_not_retrained=True))
    environment = {**os.environ, 'PYTHONPATH': f'{source}/src:{source}/vendor/Pi3'}

    def status(phase, **fields):
        write_json(root/'status.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields))

    def evaluate(jobs, partition):
        queue, active, results = [], {}, {n: {} for n in jobs}
        for name in jobs:
            (root/name/partition).mkdir(parents=True)
            for number, offset in enumerate(range(0, len(splits[partition]), 5)):
                queue.append((name, splits[partition][offset:offset+5], number))
        while queue or active:
            for slot in range(4):
                if slot in active or not queue:continue
                name, ids, number = queue.pop(0)
                output = root/name/'batches'/partition/f'{number:03d}'; output.parent.mkdir(parents=True, exist_ok=True)
                stream = output.with_suffix('.log').open('w')
                process = subprocess.Popen([sys.executable, str(source/'scripts/evaluate_geometry_batch.py'),
                    '--checkpoint', str(jobs[name]), '--partition', partition, '--output-dir', str(output), '--episodes', *ids],
                    env={**environment, 'CUDA_VISIBLE_DEVICES': str(slot)}, stdout=stream, stderr=subprocess.STDOUT)
                active[slot] = (process, stream, name, ids, output)
            for slot, (process, stream, name, ids, output) in list(active.items()):
                code = process.poll()
                if code is None:continue
                stream.close()
                if code:raise RuntimeError(f'Evaluation failed: {output}.log')
                for episode in ids:
                    results[name][episode] = validate_trace(output/'episodes'/episode, episode)
                rows = [results[name][i] for i in splits[partition] if i in results[name]]
                write_json(root/name/partition/'closed_loop.json', {**summarize_rollouts(rows), 'results': rows})
                del active[slot]
            status('evaluating_'+partition, completed_episodes={n: len(v) for n, v in results.items()})
            if active:time.sleep(3)
        for name, rows in results.items():
            assert set(rows) == set(splits[partition])
            write_json(root/name/partition/'complete.json', dict(episodes=len(rows), partition=partition))
        return {n: sum(r['success'] for r in v.values()) for n, v in results.items()}

    counts = evaluate(paths, 'validation')
    while not all((base/n/'validation/complete.json').exists() for n in ('control', *names)):
        status('waiting_for_reference_validation'); time.sleep(3)
    original_counts = {n: sum(r['success'] for r in read_json(base/n/'validation/closed_loop.json')['results'])
                       for n in ('control', *names)}
    gate = max(68, original_counts['control'])
    selected = {n: (512 if counts[n] > original_counts[n] else 256) for n in names}
    nominees = {n: paths[n] for n in names if selected[n] == 512 and counts[n] >= gate}
    preferred = 'joint_d4' if 'joint_d4' in nominees else 'c1_d2' if 'c1_d2' in nominees else None
    write_json(root/'test_freeze.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        nominees={n: dict(checkpoint=str(p), sha256=digest(p), validation_successes=counts[n]) for n, p in nominees.items()},
        preferred=preferred, selected_budgets=selected, validation_256=original_counts, validation_512=counts,
        test_used_for_selection=False))
    tested = evaluate(nominees, 'test') if nominees else {}
    while not (base/'control/test/complete.json').exists():
        status('waiting_for_reference_test'); time.sleep(3)
    control_test = sum(r['success'] for r in read_json(base/'control/test/closed_loop.json')['results'])
    test_gate = max(69, control_test); accepted = [n for n in nominees if tested[n] >= test_gate]
    report = dict(validation_256=original_counts, validation_512=counts, selected_budgets=selected,
        test_successes=tested, accepted=accepted, preferred=preferred if preferred in accepted else None,
        validation_gate=gate, test_gate=test_gate, control_test=control_test, test_used_for_selection=False,
        checkpoints={n: str(paths[n]) for n in accepted})
    for name in names:
        if name not in accepted:paths[name].unlink()
    write_json(root/'summary.json', report); status('complete', **report)


if __name__ == '__main__':main()
