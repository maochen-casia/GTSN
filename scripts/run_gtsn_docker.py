"""Launch GTSN numerical work in the existing project image; no host dependencies."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='gtsn-pilot-20261002')
    parser.add_argument('--root', default='/run/user/1016/experiments/gtsn_pilot_20261002')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--test', action='store_true')
    actions.add_argument('--report', action='store_true')
    args = parser.parse_args()
    image = 'gtsn-pi3:20261002'
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
    command = ['docker', 'run', '--init', '--network', 'none', '--read-only',
               '--user', f'{os.getuid()}:{os.getgid()}', '--gpus', '"device=0,1,2,3"',
               '--shm-size', '8g', '--tmpfs', '/tmp:rw,exec,size=4g',
               '--env', 'OMP_NUM_THREADS=1', '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'MKL_NUM_THREADS=1',
               '--env', 'LP_NUM_THREADS=1', '--env', 'MPLCONFIGDIR=/tmp/matplotlib',
               '--env', 'XDG_CACHE_HOME=/tmp/cache', '--env', 'PYTHONDONTWRITEBYTECODE=1',
               '--env', f'GTSN_DOCKER_IMAGE_ID={image_id}',
               '--mount', f'type=bind,source={Path(__file__).resolve().parents[1]},target=/workspace,readonly',
               '--mount', 'type=bind,source=/run/user/1016/tsn-1k,target=/run/user/1016/tsn-1k,readonly',
               '--mount', 'type=bind,source=/run/user/1016/experiments,target=/run/user/1016/experiments']
    if args.test:
        command += ['--rm', image, 'python', '-m', 'unittest', 'discover', '-s', '/workspace/tests', '-v']
    elif args.report:
        command += ['--rm', image, 'python', '/workspace/scripts/report_gtsn.py', '--root', args.root]
    else:
        command += ['--name', args.name, '-d', image, 'python', '/workspace/scripts/run_gtsn.py', '--root', args.root, 'all']
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
