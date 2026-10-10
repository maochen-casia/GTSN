"""Launch official-Pi3-only joint training in an isolated offline container."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--image', default='gtsn-experiment:20261009-clean')
    parser.add_argument('--config', type=Path, help='Fresh joint configuration, optionally with adaptive surface nodes')
    parser.add_argument('--seed', type=int,
                        help='Override only the training seed; preserve the benchmark split seed')
    args = parser.parse_args()
    storage = Path('/home/datasets_v2/chenmao')
    root, project = args.root.resolve(), Path(__file__).resolve().parents[1]
    ids = args.gpus.split(',')
    if len(ids) != len(set(ids)) or not all(i.isdigit() for i in ids):
        parser.error('Use distinct physical GPU IDs')
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error('Seed must lie in [0, 2**32)')
    if root.exists() or not root.is_relative_to(storage/'experiments') or root == storage/'experiments':
        parser.error('Use a new directory inside the experiment storage')
    config = json.loads((args.config or project/'configs/attention_joint_fresh.json').read_text())
    if args.seed is not None:
        config['train']['seed'] = args.seed
    if config['train']['batch_size'] % len(ids):
        parser.error('Global batch size must be divisible by the GPU count')
    image_id = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    root.mkdir(parents=True)
    source = root/'source'
    for name in ('src', 'scripts', 'tests', 'configs', 'assets', 'vendor/Pi3'):
        shutil.copytree(project/name, source/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in ('Dockerfile', 'pyproject.toml'):
        shutil.copy2(project/name, source/name)
    (root/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    name = 'gtsn-'+root.name.replace('_', '-')
    command = ['docker', 'run', '-d', '--name', name, '--init', '--network', 'none', '--read-only',
        '--user', f'{os.getuid()}:{os.getgid()}', '--cpus', str(4*len(ids)),
        '--memory', f'{max(32, 12*len(ids))}g',
        '--shm-size', '16g', '--tmpfs', '/tmp:rw,exec,size=2g', '--gpus', f'"device={args.gpus}"']
    for setting in ('PYTHONDONTWRITEBYTECODE=1', f'PYTHONPATH={source}/src:{source}/vendor/Pi3',
                    'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'LP_NUM_THREADS=1',
                    'XDG_CACHE_HOME=/tmp/cache', 'MPLCONFIGDIR=/tmp/matplotlib', 'NCCL_SOCKET_IFNAME=lo'):
        command += ['--env', setting]
    # Previous experiment weights and cached model predictions are inaccessible.
    mounts = [(source, Path('/workspace'), True), (root, root, False), (source, source, True),
              (storage/'tsn-1k-var', Path('/run/user/1016/tsn-1k-var'), True),
              (storage/'pi3', Path('/run/user/1016/pi3'), True),
              (storage/'experiments/gtsn_cam_var_20261008/recovery', Path('/run/user/1016/joint-recovery'), True)]
    for origin, target, readonly in mounts:
        command += ['--mount', f'type=bind,source={origin},target={target}'+(',readonly' if readonly else '')]
    (root/'runtime.json').write_text(json.dumps(dict(image_id=image_id, gpu_ids=ids, container=name,
        training_seed=config['train']['seed'], benchmark_seed=config['benchmark']['seed'],
        distributed_processes=len(ids), global_batch_size=config['train']['batch_size'],
        local_batch_size=config['train']['batch_size']//len(ids),
        mounts=[dict(source=str(a), target=str(b), readonly=c) for a,b,c in mounts]), indent=2)+'\n')
    invocation = ['python', str(source/'scripts/run_fresh_joint.py'), '--root', str(root), '--workers', str(len(ids))]
    command += [args.image, 'bash', '-c', shlex.join(invocation)+' > '+shlex.quote(str(root/'pipeline.log'))+' 2>&1']
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
