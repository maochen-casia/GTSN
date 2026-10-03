"""Launch a source-snapshotted fresh run with no historical checkpoints mounted."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='gtsn-consensus-fresh-20261003')
    parser.add_argument('--root', type=Path,
                        default=Path('/run/user/1016/experiments/gtsn_consensus_20261003_fresh'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    if not args.root.is_absolute() or args.root.parent != Path('/run/user/1016/experiments'):
        parser.error('Use a new direct child of /run/user/1016/experiments')
    image = 'gtsn-pi3:20261002'
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
    args.root.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    snapshot = args.root/'source'
    for directory in ('src', 'scripts', 'configs', 'tests', 'vendor/Pi3'):
        shutil.copytree(repo/directory, snapshot/directory,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    command = ['docker', 'run', '--init', '--network', 'none', '--read-only',
               '--user', f'{os.getuid()}:{os.getgid()}', '--gpus', '"device=0,1,2,3"',
               '--shm-size', '8g', '--tmpfs', '/tmp:rw,exec,size=4g',
               '--env', 'OMP_NUM_THREADS=1', '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'MKL_NUM_THREADS=1',
               '--env', 'LP_NUM_THREADS=1', '--env', 'MPLCONFIGDIR=/tmp/matplotlib',
               '--env', 'XDG_CACHE_HOME=/tmp/cache', '--env', 'PYTHONDONTWRITEBYTECODE=1',
               '--env', f'GTSN_DOCKER_IMAGE_ID={image_id}',
               '--mount', f'type=bind,source={snapshot},target=/workspace,readonly',
               '--mount', f'type=bind,source={args.root},target={args.root}']
    for path in ('/run/user/1016/tsn-1k', '/run/user/1016/pi3',
                 '/run/user/1016/experiments/pi3_rgb_perturbations_20261002'):
        command += ['--mount', f'type=bind,source={path},target={path},readonly']
    if args.test:
        command += ['--rm', image_id, 'python', '-m', 'unittest', 'discover', '-s', '/workspace/tests', '-v']
    else:
        command += ['--name', args.name, '-d', image_id, 'python', '/workspace/scripts/run_fresh_consensus.py',
                    '--root', str(args.root), '--batch-size', str(args.batch_size), '--epochs', str(args.epochs)]
        if args.smoke:
            command += ['--smoke']
    (args.root/'launch.json').write_text(json.dumps({'command': command, 'image_id': image_id}, indent=2)+'\n')
    subprocess.run(command, check=True)
    print(f'Artifacts: {args.root}', flush=True)


if __name__ == '__main__':
    main()
