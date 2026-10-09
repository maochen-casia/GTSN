"""Frozen-source Docker pipeline for four matched TSN baseline experiments.

This script runs INSIDE Docker. The host mounts project/storage read-only, only
the new experiment root writable, and passes four available GPU IDs. It creates
configs, verifies algorithms, caches shared training inputs, runs four workers,
then audits all 800 closed-loop rollouts against the existing main experiment.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tsn.common.config import read_json, write_json, output_path
from tsn.baselines.policies import METHODS


def files_digest(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('/workspace/configs/cam_var.json'))
    parser.add_argument('--main-root', type=Path, default=Path('/run/user/1016/experiments/gtsn_cam_var_20261008'))
    parser.add_argument('--gpu-slots', default='0,1,2,3')
    parser.add_argument('--image-id', required=True)
    args = parser.parse_args()
    root = output_path(args.root)
    if root.exists() and any(p.name != 'pipeline.log' for p in root.iterdir()):
        parser.error('Use a fresh empty experiment directory')
    root.mkdir(parents=True, exist_ok=True)
    slots = args.gpu_slots.split(',')
    if len(slots) != 4 or len(set(slots)) != 4:
        parser.error('Provide four distinct GPU slots visible inside the container')
    source = root/'source'
    source.mkdir()
    project = Path('/workspace')
    for name in ('src', 'scripts', 'tests', 'configs', 'assets', 'vendor'):
        shutil.copytree(project/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in ('pyproject.toml', 'Dockerfile'):
        shutil.copy2(project/name, source/name)
    write_json(root/'source_sha256.json', files_digest(source))
    config = read_json(args.config)
    reference = read_json(args.main_root/'configs/full.json')
    if config != reference:
        raise ValueError('The baseline base config must match the archived main model')
    write_json(root/'base_config.json', config)
    (root/'configs').mkdir()
    for method in METHODS:
        adapted = copy.deepcopy(config)
        adapted['baseline'] = {'method': method, 'inference_steps': 20,
            'tokenizer_epochs': 30, 'observation_slots': 2, 'rgb_hw': [84, 112], 'points': 512,
            'unet_widths': [64, 128, 256], 'point_sampling': 'deterministic uniform visible raster samples',
            'carp_scales': [1, 2, 4, 8], 'carp_vocab': 256,
            'flow_segments': 2, 'flow_delta': .01, 'flow_alpha': 1e-5, 'flow_boundary': 1., 'flow_eps': .01}
        write_json(root/'configs'/f'{method}.json', adapted)
    environment = {**os.environ, 'PYTHONPATH': f'{source}/src:{source}/vendor/Pi3', 'PYTHONDONTWRITEBYTECODE': '1'}
    protocol = {'methods': list(METHODS), 'main_root': str(args.main_root), 'image_id': args.image_id,
        'seed': config['train']['seed'], 'epochs': config['train']['epochs'],
        'draws_per_epoch': config['train']['draws_per_epoch'], 'batch_size': config['train']['batch_size'],
        'learning_rate': config['train']['learning_rate'], 'source_mixture': {'expert': .65, 'perturbation': .35},
        'checkpoint_selection': 'minimum validation waypoint RMSE, fixed seeded sampling every epoch',
        'test_used_for_selection': False, 'prediction_horizon': 30, 'execute_horizon': 15,
        'carp_additional_tokenizer_epochs': 30, 'main_pretraining': 'published Pi3 encoder',
        'baseline_pretraining': None, 'matched_simulation': True, 'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(root/'protocol.json', protocol)
    import importlib.metadata
    import torch
    write_json(root/'environment.json', {'python': sys.version, 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'image_id': args.image_id,
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'packages': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'h5py', 'sapien', 'mani-skill', 'pyrender', 'matplotlib')},
        'network': 'none', 'host_package_installations': 0})
    # Compare numerical simulator source and bind the main results/checkpoint.
    simulation_files = [p for p in (source/'src/tsn/simulation').glob('*.py')]
    for path in simulation_files:
        relative = path.relative_to(source)
        if path.read_bytes() != (args.main_root/'source'/relative).read_bytes():
            raise ValueError(f'Simulator differs from archived main: {relative}')
    shutil.copy2(args.main_root/'summary.json', root/'main_summary.json')
    shutil.copy2(args.main_root/'benchmark_sha256.json', root/'benchmark_sha256.json')
    write_json(root/'main_reference_sha256.json', {name: hashlib.sha256((args.main_root/name).read_bytes()).hexdigest()
        for name in ('summary.json', 'full/training/best.pt', 'full/validation/closed_loop.json', 'full/test/closed_loop.json')})

    def status(phase, **fields):
        write_json(root/'status.json', {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), **fields})

    def run(command, log, env=environment):
        with (root/log).open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    processes, streams = {}, []
    try:
        status('testing')
        run([sys.executable, '-m', 'unittest', 'discover', '-s', str(source/'tests'), '-v'], 'tests.log')
        write_json(root/'tests_complete.json', {'passed': True})
        status('caching_shared_inputs')
        run([sys.executable, str(source/'scripts/baseline.py'), 'cache', '--config', str(root/'base_config.json'),
             '--output-dir', str(root/'cache')], 'cache.log')
        status('checking_gpu_and_live_inputs')
        run([sys.executable, str(source/'scripts/smoke_baselines.py'), '--root', str(root)], 'smoke.log',
            {**environment, 'CUDA_VISIBLE_DEVICES': slots[0]})
        status('training_and_evaluating')
        for method, slot in zip(METHODS, slots):
            stream = (root/f'{method}.log').open('w')
            streams.append(stream)
            worker_env = {**environment, 'CUDA_VISIBLE_DEVICES': slot}
            command = [sys.executable, str(source/'scripts/baseline_worker.py'), '--root', str(root), '--method', method]
            processes[method] = subprocess.Popen(command, env=worker_env, stdout=stream, stderr=subprocess.STDOUT)
        while processes:
            for name, process in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(f'{name} failed with exit code {code}; see {name}.log')
                del processes[name]
            status('training_and_evaluating', active_methods=list(processes))
            if processes:
                time.sleep(10)
        status('evaluating_isolated_batches')
        run([sys.executable, str(source/'scripts/diagnose_baseline_validation.py'), '--root', str(root)],
            'validation_diagnostics.log', {**environment, 'CUDA_VISIBLE_DEVICES': slots[0]})
        run([sys.executable, str(source/'scripts/resume_baseline_evaluations.py'), '--root', str(root),
             '--gpu-slots', args.gpu_slots, '--script-root', str(source/'scripts')], 'evaluation.log')
        run([sys.executable, str(source/'scripts/finalize_baseline_report.py'), '--root', str(root)],
            'report_finalization.log')
        run([sys.executable, str(source/'scripts/report_success_thresholds.py'), '--root', str(root)],
            'success_thresholds.log')
        summary = read_json(root/'summary.json')
        status('complete', test_successes={name: row['test']['successes'] for name, row in summary['models'].items()})
    except Exception as error:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            process.wait()
        status('failed', error=str(error))
        raise
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
