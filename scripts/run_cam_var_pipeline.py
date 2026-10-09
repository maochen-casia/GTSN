"""Freeze and complete the Panda camera-variation experiment inside Docker."""
import argparse
import copy
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tsn.common.config import PROJECT_ROOT, read_json, write_json
from report_sim2real import sha256, audit_recovery
from tsn.data.splits import make_splits

VARIANTS = {'full': None, 'without_c1': 'c1', 'without_c2': 'c2', 'without_c3': 'c3'}


def files_digest(directory):
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob('*')) if path.is_file() and '__pycache__' not in path.parts}


def gpu_bindings(protocol):
    """Use GPU UUIDs when requested; otherwise use container CUDA ordinals."""
    bindings = protocol.get('gpu_uuids') or list(map(str, range(len(protocol['variants']))))
    if len(bindings) != len(protocol['variants']) or len(set(bindings)) != len(bindings):
        raise ValueError('Assign one distinct GPU to each variant')
    return bindings


def prepare(root, seed=None, variants=None, reference_root=None, gpu_ids=None,
            gpu_uuids=None, comparison_roots=None):
    if any(root.iterdir()):
        raise FileExistsError('Preparation requires an empty experiment directory')
    source = root/'source'
    origin = reference_root/'source' if reference_root else PROJECT_ROOT
    variants = list(VARIANTS) if variants is None else variants
    gpu_ids = list(range(len(variants))) if gpu_ids is None else gpu_ids
    if len(gpu_ids) != len(variants) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Assign one distinct physical GPU to each variant')
    gpu_bindings(dict(variants=variants, gpu_uuids=gpu_uuids))
    for name in ('src', 'scripts', 'tests', 'configs', 'assets', 'vendor/Pi3'):
        location = PROJECT_ROOT if name == 'tests' else origin
        shutil.copytree(location/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in ('Dockerfile', 'pyproject.toml', 'instructions/cam_var.md'):
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin/name, target)
    if reference_root:
        # Reuse the exact numerical implementation, adding current orchestration
        # and reporting scripts without modifying model/data/training code.
        for name in ('run_cam_var_pipeline.py', 'report_cam_var.py', 'smoke_cam_var.py',
                     'compare_cam_var_seeds.py'):
            shutil.copy2(PROJECT_ROOT/'scripts'/name, source/'scripts'/name)
        base = read_json(reference_root/'configs/full.json')
        (root/'recovery').symlink_to(Path(base['train']['recovery_root']).resolve(), target_is_directory=True)
        for name in ('benchmark_sha256.json', 'benchmark_audit.json', 'runtime_image.json'):
            shutil.copy2(reference_root/name, root/name)
        shutil.copy2(reference_root/'summary.json', root/'previous_seed_summary.json')
        write_json(root/'recovery_sha256.json', files_digest(root/'recovery'))
    else:
        base = read_json(source/'configs/cam_var.json')
        base['train']['recovery_root'] = str(root/'recovery')
    if seed is not None:
        base['train']['seed'] = seed
    for variant in variants:
        removed = VARIANTS[variant]
        config = copy.deepcopy(base)
        config['model']['contributions'] = {key: key != removed for key in ('c1', 'c2', 'c3')}
        write_json(root/'configs'/f'{variant}.json', config)
    write_json(root/'source_sha256.json', files_digest(source))
    protocol = dict(benchmark=base['benchmark']['root'], robot='panda',
        geometry_mode='camera_ray_depth', split_counts=[800, 100, 100],
        route_counts=dict(direct=200, over=400, side=400), seed=base['train']['seed'],
        epochs=base['train']['epochs'], draws_per_epoch=base['train']['draws_per_epoch'],
        batch_size=base['train']['batch_size'], learning_rate=base['train']['learning_rate'],
        initialization='fresh navigation weights; only official Pi3 image encoder initialized',
        encoder_frozen=False, source_mixture=dict(expert=.65, independent_perturbations=.35),
        checkpoint_selection='minimum validation waypoint RMSE across all 30 epochs',
        test_used_for_selection=False, prediction_horizon=30, execute_horizon=15,
        goal_position_tolerance_m=.01, variants={name:description for name,description in {
            'full': 'C1/C2/C3', 'without_c1': 'current RGB only; no history or persistent surfaces',
            'without_c2': 'TCP point only; no hand/wrist/camera scoring or body self filtering',
            'without_c3': 'no uncertainty or trust training; no clearance refinement'}.items() if name in variants},
        runtime_image='gtsn-sim2real:20261008', gpu_ids=gpu_ids, gpu_uuids=gpu_uuids,
        comparison_roots=[str(path) for path in [*comparison_roots, root]] if comparison_roots else [],
        reference_root=str(reference_root) if reference_root else None,
        previous_seed=read_json(reference_root/'protocol.json')['seed'] if reference_root else None,
        previous_seed_summary_sha256=sha256(root/'previous_seed_summary.json') if reference_root else None,
        shared_recovery=bool(reference_root),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(root/'protocol.json', protocol)
    if reference_root:
        runtime = read_json(root/'runtime_image.json')
        runtime['gpu_ids'] = protocol['gpu_ids']
        runtime['gpu_uuids'] = gpu_uuids
        write_json(root/'runtime_image.json', runtime)
    print(protocol, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--variants', nargs='+', choices=tuple(VARIANTS))
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--gpu-ids', nargs='+', type=int)
    parser.add_argument('--gpu-uuids', nargs='+')
    parser.add_argument('--comparison-roots', nargs='+', type=Path)
    args = parser.parse_args()
    root = args.root
    if args.prepare_only:
        if args.variants and ('full' not in args.variants or len(args.variants) != len(set(args.variants))):
            parser.error('Variants must include full and contain no duplicates')
        prepare(root, args.seed, args.variants, args.reference_root, args.gpu_ids,
                args.gpu_uuids, args.comparison_roots)
        return
    source = root/'source'
    base = read_json(root/'configs/full.json')
    protocol = read_json(root/'protocol.json')
    variants = list(protocol['variants'])
    bindings = gpu_bindings(protocol)

    def status(phase, **details):
        value = dict(phase=phase, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details)
        write_json(root/'status.json', value)
        print(value, flush=True)

    def run_many(commands, phase):
        processes, handles = {}, []
        try:
            for name, (command, env) in commands.items():
                log = root/f'{name}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open('w'); handles.append(stream)
                processes[name] = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            while any(process.poll() is None for process in processes.values()):
                details = {}
                for name, process in processes.items():
                    if process.poll() not in (None, 0):
                        raise RuntimeError(f'{name} failed with exit {process.returncode}; see {name}.log')
                    progress = root/name/'progress.json'
                    rollouts = root/name/'closed_loop.json'
                    if progress.exists():
                        details[name] = read_json(progress)
                    elif rollouts.exists():
                        details[name] = dict(episodes=len(read_json(rollouts)['results']))
                    else:
                        details[name] = dict(running=process.poll() is None)
                status(phase, jobs=details)
                time.sleep(30)
            for name, process in processes.items():
                if process.returncode:
                    raise RuntimeError(f'{name} failed; see {name}.log')
        finally:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            for stream in handles:
                stream.close()

    python = sys.executable
    environment = dict(os.environ)
    try:
        expected = read_json(root/'source_sha256.json')
        assert files_digest(source) == expected
        if protocol.get('gpu_uuids'):
            available = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                                  '--format=csv,noheader'], text=True).splitlines()
            assert set(bindings) <= set(available), ('Requested GPUs are not exposed', bindings, available)
            write_json(root/'gpu_assignment.json', {variant:dict(physical_cuda_index=gpu,
                uuid=binding, process_cuda_index=0) for variant,gpu,binding in
                zip(variants, protocol['gpu_ids'], bindings)})
        if protocol.get('reference_root'):
            status('verifying_identical_reference_data_and_code')
            reference = Path(protocol['reference_root'])
            original = read_json(reference/'source_sha256.json')
            for name, digest in original.items():
                if name.startswith(('src/', 'vendor/', 'assets/')) or name in ('Dockerfile', 'pyproject.toml'):
                    assert expected[name] == digest, ('numerical implementation changed', name)
            for variant in variants:
                compared = read_json(root/'configs'/f'{variant}.json')
                previous = read_json(reference/'configs'/f'{variant}.json')
                compared['train']['seed'] = previous['train']['seed']
                assert compared == previous, ('Only training seed may change', variant)
            dataset = Path(base['benchmark']['root'])
            for name, digest in read_json(root/'benchmark_sha256.json').items():
                assert sha256(dataset/name) == digest, ('benchmark changed', name)
            assert files_digest(root/'recovery') == read_json(root/'recovery_sha256.json')
        # Identify actual episode contents, rather than just ID/route membership.
        if not (root/'benchmark_sha256.json').exists():
            status('fingerprinting_benchmark')
            dataset = Path(base['benchmark']['root'])
            paths = [dataset/'manifest.json', *sorted(dataset.glob('episode_*/episode.h5')),
                     *sorted(dataset.glob('episode_*/scene.json'))]
            write_json(root/'benchmark_sha256.json', {str(p.relative_to(dataset)):sha256(p) for p in paths})
        commands = {}
        if not (root/'tests_complete.json').exists():
            commands['tests'] = ([python, '-m', 'unittest', 'discover', '-s', str(source/'tests'), '-v'], environment)
        if not (root/'benchmark_audit.json').exists():
            commands['benchmark_audit'] = ([python, str(source/'scripts/check_sim2real.py'),
                '--config', str(root/'configs/full.json'), '--output', str(root/'benchmark_audit.json'),
                '--workers', str(args.workers)], environment)
        if not (root/'recovery/manifest.json').exists():
            commands['recovery'] = ([python, '-m', 'tsn.cli.generate_recovery', '--config',
                str(root/'configs/full.json'), '--output-dir', str(root/'recovery'), '--workers', str(args.workers)], environment)
        if commands:
            run_many(commands, 'checking_benchmark_and_generating_recovery')
            if 'tests' in commands:
                write_json(root/'tests_complete.json', dict(passed=True, source_sha256=sha256(root/'source_sha256.json')))
        calibration = read_json(root/'benchmark_audit.json')
        assert min(row['fraction_within_2mm'] for row in calibration['reconstructed_views']) > .999
        assert max(row['camera_pose_error'] for row in calibration['reconstructed_views']) < 3e-6
        splits = make_splits(base['benchmark'])
        write_json(root/'recovery_audit.json', audit_recovery(root, base, splits))
        if not (root/'smoke_complete.json').exists():
            run_many({'smoke': ([python, str(source/'scripts/smoke_cam_var.py'), '--root', str(root)],
                               {**environment, 'CUDA_VISIBLE_DEVICES': bindings[0]})}, 'checking_real_perception_gradients')
        commands = {}
        for binding, variant in zip(bindings, variants):
            output = root/variant/'training'
            if not (output/'complete.json').exists():
                commands[f'{variant}/training'] = ([python, '-m', 'tsn.cli.train', '--config',
                    str(root/'configs'/f'{variant}.json'), '--output-dir', str(output)],
                    {**environment, 'CUDA_VISIBLE_DEVICES': binding})
        if commands:
            run_many(commands, 'training')
        # Freeze every selected model before the first test rollout is launched.
        for variant in variants:
            directory = root/variant
            records = read_json(directory/'training/epochs.json')
            selected = min(records, key=lambda row:row['validation_rmse_m'])
            receipt = dict(
                checkpoint=str(directory/'training/best.pt'), sha256=sha256(directory/'training/best.pt'),
                selected_epoch=selected['epoch'], validation_rmse_m=selected['validation_rmse_m'],
                frozen_utc_before_test=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                test_used_for_selection=False)
            if (directory/'selected_checkpoint.json').exists():
                previous_receipt = read_json(directory/'selected_checkpoint.json')
                assert all(receipt[key] == previous_receipt[key] for key in
                           ('checkpoint', 'sha256', 'selected_epoch', 'validation_rmse_m', 'test_used_for_selection'))
            else:
                write_json(directory/'selected_checkpoint.json', receipt)
        # One rollout per GPU; keep all three contribution ablations paired.
        for partition in ('validation', 'test'):
            commands = {}
            for binding, variant in zip(bindings, variants):
                output = root/variant/partition
                if not (output/'complete.json').exists():
                    commands[f'{variant}/{partition}'] = ([python, '-m', 'tsn.cli.evaluate',
                        '--checkpoint', str(root/variant/'training/best.pt'), '--partition', partition,
                        '--output-dir', str(output), '--no-render-videos'],
                        {**environment, 'CUDA_VISIBLE_DEVICES': binding})
            if commands:
                run_many(commands, f'evaluating_{partition}')
        run_many({'audit': ([python, str(source/'scripts/report_cam_var.py'), '--root', str(root)], environment)}, 'auditing_results')
        summary = read_json(root/'summary.json')
        comparison_roots = [Path(path) for path in protocol.get('comparison_roots', [])]
        if comparison_roots:
            while True:
                pending = []
                for other in comparison_roots:
                    if other == root:
                        continue
                    other_status = read_json(other/'status.json') if (other/'status.json').exists() else {}
                    if other_status.get('phase') == 'failed':
                        raise RuntimeError(f'Seed comparison requires failed experiment {other}')
                    if other_status.get('phase') != 'complete':
                        pending.append(str(other))
                if not pending:
                    break
                status('waiting_for_seed_comparison', pending=pending, training_and_evaluation_complete=True)
                time.sleep(30)
            run_many({'seed_comparison': ([python, str(source/'scripts/compare_cam_var_seeds.py'),
                '--roots', *map(str, comparison_roots), '--output-dir', str(root)], environment)},
                'comparing_training_seeds')
        status('complete', test_success_rates={name: row['partitions']['test']['success_rate']
                                               for name, row in summary['models'].items()})
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
