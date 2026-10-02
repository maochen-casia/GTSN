"""Launch the complete Pi3 experiment using the project Docker image."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='gtsn-pi3:20261002')
    parser.add_argument('--name', default='gtsn-pi3-small-perturbation10-20261002')
    parser.add_argument('--output-dir', default='/run/user/1016/experiments/pi3_small_perturbation10_20261002')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    image_id = 'inspect-at-launch'
    if not args.print_command:
        image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image], text=True).strip()
    command = ['docker', 'run', '--name', args.name, '--init', '--network', 'none', '--read-only',
               '--user', f'{os.getuid()}:{os.getgid()}', '--gpus', '"device=0,1,2,3"',
               '--shm-size', '8g', '--tmpfs', '/tmp:rw,exec,size=4g',
               '--env', 'OMP_NUM_THREADS=1', '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'MKL_NUM_THREADS=1',
               '--env', 'LP_NUM_THREADS=1', '--env', 'MPLCONFIGDIR=/tmp/matplotlib',
               '--env', 'XDG_CACHE_HOME=/tmp/cache', '--env', f'GTSN_DOCKER_IMAGE={args.image}',
               '--env', f'GTSN_DOCKER_IMAGE_ID={image_id}',
               '--mount', f'type=bind,source={root},target=/workspace,readonly',
               '--mount', 'type=bind,source=/run/user/1016/tsn-1k,target=/run/user/1016/tsn-1k,readonly',
               '--mount', 'type=bind,source=/run/user/1016/pi3,target=/run/user/1016/pi3,readonly',
               '--mount', 'type=bind,source=/run/user/1016/experiments,target=/run/user/1016/experiments']
    if not args.foreground:
        command.append('-d')
    command += [args.image, 'python', '/workspace/scripts/run_pi3_perturbation.py', '--output-dir', args.output_dir]
    print(shlex.join(command), flush=True)
    if not args.print_command:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
