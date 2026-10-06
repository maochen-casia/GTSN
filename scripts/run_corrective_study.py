"""Train the current-view residual and evaluate it from a saved source snapshot.

Host operations use only the standard library. All model work runs in Docker.
Test evaluation is opt-in and never selects or changes the trained checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'initial-head', 'initial-residual', 'cache', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--recovery-cache', type=Path, action='append', required=True)
    parser.add_argument('--baseline-validation', type=Path)
    parser.add_argument('--baseline-test', type=Path)
    parser.add_argument('--test', action='store_true', help='Also evaluate the fixed test split')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--image', default='gtsn-persistent:20261005-compact-only')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--draws', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261005)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(Path('/run/user/1016/experiments')):
        parser.error('Use a fresh directory under /run/user/1016/experiments')
    if min(args.epochs, args.draws, args.batch_size) < 1:
        parser.error('Training sizes must be positive')
    for path in (args.checkpoint, args.initial_head, args.initial_residual, args.cache, *args.recovery_cache):
        if not path.exists():
            parser.error(f'Missing input: {path}')
    output.mkdir(parents=True)
    snapshot = output/'source'
    snapshot.mkdir()
    for name in ('src', 'scripts', 'tests', 'configs'):
        shutil.copytree(root/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('README.md', 'Dockerfile', 'pyproject.toml'):
        shutil.copy2(root/name, snapshot/name)
    write(output/'source_sha256.json', {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in snapshot.rglob('*') if p.is_file()})
    write(output/'protocol.json', {k: ([str(p.resolve()) for p in v] if isinstance(v, list) else
                                     str(v.resolve()) if isinstance(v, Path) else v)
                                   for k, v in vars(args).items()})
    launcher = [sys.executable, str(root/'scripts/docker_run.py'), '--image', args.image,
                '--source-snapshot', str(snapshot)]

    def run(arguments, stage):
        write(output/'status.json', dict(stage=stage))
        with (output/f'{stage}.log').open('w') as log:
            subprocess.run(launcher+arguments, stdout=log, stderr=subprocess.STDOUT, check=True)

    training = ['--gpu', args.gpu, 'corrective', '--stage', 'train', '--output-dir', str(output/'training')]
    for name in ('checkpoint', 'initial_head', 'initial_residual', 'cache'):
        training += ['--'+name.replace('_', '-'), str(getattr(args, name).resolve())]
    for name in ('epochs', 'draws', 'batch_size', 'seed'):
        training += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    for path in args.recovery_cache:
        training += ['--recovery-cache', str(path.resolve())]
    run(training, 'training')
    checkpoint = output/'training/best.pt'
    hasher = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    write(output/'locked_checkpoint.json', dict(path=str(checkpoint), sha256=digest))
    results = {}
    for partition in (('validation', 'test') if args.test else ('validation',)):
        directory = output/'evaluation'/partition/'current_view'
        run(['--gpu', args.gpu, 'evaluate', '--controller', 'compact', '--checkpoint', str(checkpoint),
             '--partition', partition, '--no-render-videos', '--output-dir', str(directory)], partition)
        results[partition] = json.loads((directory/'closed_loop.json').read_text())['overall']
        reference = getattr(args, 'baseline_'+partition)
        if reference:
            run(['--cpu', 'report', '--root', str(directory.parent), '--reference', str(reference.resolve()),
                 '--candidate', 'current_view', '--output-dir', str(output/'reports'/partition)], partition+'_report')
    write(output/'complete.json', dict(checkpoint_sha256=digest, results=results,
                                      selection='fixed final epoch', clearance='disabled'))


if __name__ == '__main__':
    main()
