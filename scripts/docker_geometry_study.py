"""Launch the learned-geometry study in the existing offline project image."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,3,6')
    parser.add_argument('--image', default='gtsn-experiment:20261009-clean')
    parser.add_argument('--reuse-cache', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    storage = Path('/home/datasets_v2/chenmao')
    root = args.root.resolve()
    if not root.is_relative_to(storage/'experiments') or root == storage/'experiments':
        parser.error('Use a directory under the requested experiment storage')
    if root.exists() and not args.reuse_cache:parser.error('Use a new experiment directory')
    if args.reuse_cache and not (root/'cache/complete.json').is_file():parser.error('The reusable cache is incomplete')
    ids = args.gpus.split(',')
    if len(ids) != 4 or len(set(ids)) != 4 or not all(i.isdigit() for i in ids):parser.error('Use four distinct GPU IDs')
    project = Path(__file__).resolve().parents[1]
    image_id = args.image if args.print_command else subprocess.check_output(
        ['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    command = ['docker', 'run', '--name', 'gtsn-'+root.name.replace('_', '-'), '--init',
        '--network', 'none', '--read-only', '--user', f'{os.getuid()}:{os.getgid()}',
        '--cpus', '12', '--memory', '40g', '--shm-size', '4g', '--tmpfs', '/tmp:rw,exec,size=2g',
        '--gpus', f'"device={args.gpus}"']
    if args.detach:command.append('-d')
    for value in ('PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=/workspace/src:/workspace/vendor/Pi3',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'LP_NUM_THREADS=1',
        'XDG_CACHE_HOME=/tmp/cache', 'MPLCONFIGDIR=/tmp/matplotlib'):
        command += ['--env', value]
    for source, target, readonly in ((project, Path('/workspace'), True), (project, project, True),
        (storage, storage, True), (storage, Path('/run/user/1016'), True), (root, root, False)):
        command += ['--mount', f'type=bind,source={source},target={target}'+(',readonly' if readonly else '')]
    invocation = ['python', '/workspace/scripts/run_geometry_study.py', '--root', str(root), '--image-id', image_id]
    if args.reuse_cache:invocation.append('--reuse-cache')
    command += [args.image, 'bash', '-c', shlex.join(invocation)+' > '+shlex.quote(str(root/'pipeline.log'))+' 2>&1']
    print(shlex.join(command), flush=True)
    if not args.print_command:
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':main()
