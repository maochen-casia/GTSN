"""Snapshot and launch the simplification study using only the idle GPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/run/user/1016/experiments/gtsn_simplification_20261003'))
    parser.add_argument('--gpu', type=int, default=4)
    args = parser.parse_args()
    if args.root.parent != Path('/run/user/1016/experiments'):
        parser.error('Use a new direct child of /run/user/1016/experiments')
    args.root.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    for directory in ('src', 'scripts', 'configs', 'tests', 'vendor/Pi3'):
        shutil.copytree(repo/directory, args.root/'source'/directory,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    baseline = Path('/run/user/1016/experiments/gtsn_consensus_20261003_fresh_interim_20261003_141728/train/best.pt')
    shutil.copy2(baseline, args.root/'baseline.pt')
    image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', 'gtsn-pi3:20261002'], text=True).strip()
    command = ['docker', 'run', '--init', '--network', 'none', '--read-only',
        '--user', f'{os.getuid()}:{os.getgid()}', '--gpus', f'device={args.gpu}', '--shm-size', '8g',
        '--tmpfs', '/tmp:rw,exec,size=4g']
    for value in ('OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'LP_NUM_THREADS=1',
                  'MPLCONFIGDIR=/tmp/matplotlib', 'XDG_CACHE_HOME=/tmp/cache', 'PYTHONDONTWRITEBYTECODE=1',
                  f'GTSN_DOCKER_IMAGE_ID={image}'):
        command += ['--env', value]
    for source, target, readonly in [(args.root/'source', Path('/workspace'), True), (args.root, args.root, False),
            (args.root/'baseline.pt', args.root/'baseline.pt', True),
            (Path('/run/user/1016/tsn-1k'), Path('/run/user/1016/tsn-1k'), True),
            (Path('/run/user/1016/experiments/pi3_rgb_perturbations_20261002'),
             Path('/run/user/1016/experiments/pi3_rgb_perturbations_20261002'), True)]:
        command += ['--mount', f'type=bind,source={source},target={target}'+(',readonly' if readonly else '')]
    tests = subprocess.run(command+['--rm', image, 'python', '-m', 'unittest', 'discover',
                                    '-s', '/workspace/tests', '-p', 'test_simplified_consensus.py', '-v'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (args.root/'tests.log').write_text(tests.stdout)
    print(tests.stdout, flush=True)
    tests.check_returncode()
    command += ['--name', args.root.name.replace('_', '-'), '-d', image, 'python',
                '/workspace/scripts/run_simplification.py', 'all', '--root', str(args.root), '--epochs', '5']
    (args.root/'launch.json').write_text(json.dumps(dict(command=command, image_id=image, gpu=args.gpu), indent=2)+'\n')
    subprocess.run(command, check=True)
    print(args.root, flush=True)


if __name__ == '__main__':
    main()
