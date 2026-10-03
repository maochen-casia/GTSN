"""Snapshot and verify the compact deployment after experiment completion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/run/user/1016/experiments/gtsn_simplification_20261003'))
    args = parser.parse_args()
    if args.root.parent != Path('/run/user/1016/experiments'):
        parser.error('Expected an experiment under /run/user/1016/experiments')
    assert json.loads((args.root/'status.json').read_text())['state'] == 'complete'
    source = args.root/'deployment_source'
    source.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    for directory in ('src', 'scripts', 'configs', 'tests', 'vendor/Pi3'):
        shutil.copytree(repo/directory, source/directory,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.rglob('*')) if p.is_file()}
    image = json.loads((args.root/'launch.json').read_text())['image_id']
    command = ['docker', 'run', '--rm', '--init', '--network', 'none', '--read-only',
               '--user', f'{os.getuid()}:{os.getgid()}', '--gpus', 'device=4',
               '--shm-size', '2g', '--tmpfs', '/tmp:rw,exec,size=2g']
    for value in ('OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'LP_NUM_THREADS=1',
                  'MPLCONFIGDIR=/tmp/matplotlib', 'XDG_CACHE_HOME=/tmp/cache', 'PYTHONDONTWRITEBYTECODE=1'):
        command += ['--env', value]
    baseline = Path('/run/user/1016/experiments/gtsn_consensus_20261003_fresh_interim_20261003_141728')
    for host, target, readonly in ((source, Path('/workspace'), True), (args.root, args.root, False),
                                  (baseline, baseline, True),
                                  (Path('/run/user/1016/tsn-1k'), Path('/run/user/1016/tsn-1k'), True)):
        command += ['--mount', f'type=bind,source={host},target={target}'+(',readonly' if readonly else '')]
    command += [image, 'python', '/workspace/scripts/audit_simplification.py', '--root', str(args.root)]
    (args.root/'deployment_provenance.json').write_text(json.dumps(dict(command=command, source_sha256=hashes), indent=2)+'\n')
    with (args.root/'export_audit.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print((args.root/'export_audit.json').read_text())
    with (args.root/'run.log').open('w') as log:
        subprocess.run(['docker', 'logs', args.root.name.replace('_', '-')], stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
