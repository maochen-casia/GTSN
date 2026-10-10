"""Launch parallel attention replacement trials in the existing offline image."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,2,5,7')
    parser.add_argument('--image', default='gtsn-experiment:20261009-clean')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--recipe', choices=('original', 'calibrated'), default='original')
    parser.add_argument('--query-study-from', type=Path)
    args = parser.parse_args()
    storage = Path('/home/datasets_v2/chenmao'); root = args.root.resolve()
    if (root.exists() and not args.resume) or not root.is_relative_to(storage/'experiments') or root == storage/'experiments':
        parser.error('Use a new experiment directory, or explicitly resume an existing study')
    if args.resume and not (root/'protocol.json').exists():parser.error('Resume requires the original protocol')
    ids = args.gpus.split(',')
    if len(ids) != len(set(ids)) or not all(i.isdigit() for i in ids):parser.error('Use distinct GPU IDs')
    if not args.resume and len(ids) != 4:parser.error('Training requires four distinct GPUs')
    project = Path(__file__).resolve().parents[1]
    image_id = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    root.mkdir(parents=True, exist_ok=args.resume)
    container = 'gtsn-'+root.name.replace('_', '-')+('-resume' if args.resume else '')
    command = ['docker', 'run', '-d', '--name', container, '--init',
        '--network', 'none', '--read-only', '--user', f'{os.getuid()}:{os.getgid()}',
        '--cpus', '24' if args.resume else '16', '--memory', '64g' if args.resume else '48g',
        '--shm-size', '4g', '--tmpfs', '/tmp:rw,exec,size=2g',
        '--gpus', f'"device={args.gpus}"']
    for value in ('PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=/workspace/src:/workspace/vendor/Pi3',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'LP_NUM_THREADS=1',
        'XDG_CACHE_HOME=/tmp/cache', 'MPLCONFIGDIR=/tmp/matplotlib'):
        command += ['--env', value]
    for source, target, readonly in ((project, Path('/workspace'), True), (project, project, True),
        (storage, storage, True), (storage, Path('/run/user/1016'), True), (root, root, False)):
        command += ['--mount', f'type=bind,source={source},target={target}'+(',readonly' if readonly else '')]
    script = ('run_query_budget_study.py' if args.query_study_from else
              'resume_replacement_study.py' if args.resume else 'run_replacement_study.py')
    invocation = ['python', '/workspace/scripts/'+script, '--root', str(root), '--image-id', image_id]
    if args.query_study_from:invocation += ['--base-root', str(args.query_study_from.resolve())]
    elif args.resume:invocation += ['--workers', str(len(ids))]
    else:invocation += ['--recipe', args.recipe]
    log = root/('resume.log' if args.resume else 'pipeline.log')
    command += [args.image, 'bash', '-c', shlex.join(invocation)+' > '+shlex.quote(str(log))+' 2>&1']
    print(shlex.join(command), flush=True); subprocess.run(command, check=True)


if __name__ == '__main__':main()
