"""Train one compact corrective head, lock it, and evaluate both fixed splits.

Model operations run in Docker. Validation and test evaluate the same final
checkpoint on separate GPUs; neither result changes the checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--training-gpu', default='0')
    parser.add_argument('--validation-gpu', default='0')
    parser.add_argument('--test-gpu', default='1')
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--resume-evaluation', action='store_true',
                        help='Evaluate an already trained, locked run after a launch failure')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    storage = Path('/run/user/1016/experiments')
    retained = storage/'gtsn_spatial_v7_corrective_20261005'
    base = storage/'gtsn_simplification_20261003/simplified.pt'
    cache = storage/'gtsn_persistent_cache_20261005'
    recoveries = [retained/f'collection/shard_{i}' for i in range(4)]
    output = args.output_dir.resolve()
    if (output.exists() and not args.resume_evaluation) or not output.is_relative_to(storage) or output == storage:
        parser.error('Use a fresh directory under /run/user/1016/experiments')
    for p in (base, cache, *recoveries):
        if not p.exists():
            parser.error(f'Missing input: {p}')
    snapshot = output/'source'
    if args.resume_evaluation:
        if not (output/'training/complete.json').is_file() or not (output/'locked_checkpoint.json').is_file():
            parser.error('Resume requires completed training and a locked checkpoint')
        locked = json.loads((output/'locked_checkpoint.json').read_text())
        if file_sha256(output/'training/best.pt') != locked['sha256']:
            parser.error('Locked checkpoint changed')
        for name, digest in json.loads((output/'source_sha256.json').read_text()).items():
            if file_sha256(snapshot/name) != digest:
                parser.error(f'Snapshot source changed: {name}')
        evaluation = output/'evaluation'
        if evaluation.exists():
            if any(p.is_file() for p in evaluation.rglob('*')):
                parser.error('Resume refuses to overwrite evaluation artifacts')
            evaluation.rename(output/f'evaluation_failed_{time.time_ns()}')
        for partition in ('validation', 'test'):
            log = output/f'{partition}.log'
            if log.exists():
                log.rename(output/f'{partition}_failed_{time.time_ns()}.log')
    else:
        output.mkdir(parents=True)
        snapshot.mkdir()
        for name in ('src', 'scripts', 'tests', 'configs'):
            shutil.copytree(project/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('README.md', 'Dockerfile', 'pyproject.toml'):
            shutil.copy2(project/name, snapshot/name)
        write(output/'source_sha256.json', {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in snapshot.rglob('*') if p.is_file()})
        write(output/'protocol.json', dict(head='compact', checkpoint=str(base), cache=str(cache),
            recovery_caches=[str(p) for p in recoveries], epochs=20, draws=4096, batch_size=32,
            seed=20261005, learning_rate=args.learning_rate, checkpoint_selection='fixed final epoch',
            source_mix=[0, 0, 1], route_mix=[.2, .4, .4], controller='compact', clearance='disabled',
            validation_gpu=args.validation_gpu, test_gpu=args.test_gpu,
            current_view_reference=str(retained/'followup'), original_compact_reference=str(retained/'reference')))
    launcher = [sys.executable, str(project/'scripts/docker_run.py'), '--source-snapshot', str(snapshot),
                '--image', 'gtsn-persistent:20261005-compact-only']
    children = []
    try:
        if not args.resume_evaluation:
            write(output/'status.json', dict(stage='training'))
            command = launcher+['--gpu', args.training_gpu, 'corrective', '--stage', 'train', '--head', 'compact',
                '--checkpoint', str(base), '--cache', str(cache), '--output-dir', str(output/'training'),
                '--epochs', '20', '--draws', '4096', '--batch-size', '32', '--seed', '20261005',
                '--learning-rate', str(args.learning_rate)]
            for p in recoveries:
                command += ['--recovery-cache', str(p)]
            with (output/'training.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        checkpoint = output/'training/best.pt'
        digest = file_sha256(checkpoint)
        if not args.resume_evaluation:
            write(output/'locked_checkpoint.json', dict(path=str(checkpoint), sha256=digest,
                                                      selection='epoch 20 before any closed-loop evaluation'))
        for partition, gpu in [('validation', args.validation_gpu), ('test', args.test_gpu)]:
            directory = output/'evaluation'/partition/'compact_corrective'
            log = (output/f'{partition}.log').open('w')
            command = launcher+['--gpu', gpu, 'evaluate', '--controller', 'compact',
                '--checkpoint', str(checkpoint), '--partition', partition, '--no-render-videos',
                '--output-dir', str(directory)]
            children.append((partition, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
        write(output/'status.json', dict(stage='evaluation', partitions=['validation', 'test']))
        while any(process.poll() is None for _, process, _ in children):
            failed = [(name, process.returncode) for name, process, _ in children
                      if process.poll() is not None and process.returncode != 0]
            if failed:
                raise RuntimeError(f'Evaluation failed: {failed}')
            time.sleep(5)
        for name, process, log in children:
            log.close()
            if process.returncode:
                raise RuntimeError(f'{name} failed: exit {process.returncode}')
        results = {}
        for partition in ('validation', 'test'):
            directory = output/'evaluation'/partition/'compact_corrective'
            results[partition] = json.loads((directory/'closed_loop.json').read_text())['overall']
            for label, reference in [('current_view', retained/f'followup/{partition}/spatial_history_corrective_current'),
                                      ('original_compact', retained/f'reference/compact_{partition}')]:
                report = output/'reports'/partition/label
                with (output/f'{partition}_{label}_report.log').open('w') as log:
                    subprocess.run(launcher+['--cpu', 'report', '--root', str(directory.parent),
                        '--reference', str(reference), '--candidate', 'compact_corrective',
                        '--output-dir', str(report)], stdout=log, stderr=subprocess.STDOUT, check=True)
        if file_sha256(checkpoint) != digest:
            raise RuntimeError('Locked checkpoint changed')
        write(output/'complete.json', dict(checkpoint_sha256=digest, results=results,
                                           selection='fixed final epoch', clearance='disabled'))
        write(output/'status.json', dict(stage='complete', results=results))
        print(json.dumps(results), flush=True)
    except BaseException as exc:
        for _, process, log in children:
            if process.poll() is None:
                process.terminate()
            log.close()
        write(output/'status.json', dict(stage='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    main()
