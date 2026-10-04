"""Geometry-grounding interventions and validation-selected memory update.

The host uses stdlib orchestration only; all numerical execution is in Docker.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from queue import Queue
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4')
    parser.add_argument('--stage', choices=('visibility', 'calibration'), default='visibility')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--previous', type=Path, default=Path(
        '/run/user/1016/experiments/gtsn_route_preserving_20261003'))
    parser.add_argument('--checkpoint', default=
        '/run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt')
    args = parser.parse_args()
    if (args.stage == 'calibration') != (args.calibration is not None):
        parser.error('Calibration stage requires --calibration; visibility stage does not use it')
    root, source = args.root.resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or not root.is_relative_to('/run/user/1016/experiments'):
        parser.error('Use a fresh experiment directory under /run/user/1016/experiments')
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        parser.error('GPU IDs must be distinct integers')
    previous = args.previous.resolve()
    for split in ('validation', 'test'):
        for name in ('baseline', 'method'):
            if not (previous/split/name/'complete.json').is_file():
                parser.error('Completed previous baseline/method runs are required')
    root.mkdir(parents=True)
    (root/'logs').mkdir()
    for folder in ('src', 'scripts', 'tests', 'configs'):
        shutil.copytree(source/folder, root/'source'/folder)

    def write(name, data):
        (root/name).write_text(json.dumps(data, indent=2)+'\n')

    variants = [('visibility_004', 'visibility', .04), ('visibility_008', 'visibility', .08),
                ('shift_pos', 'shift_pos', .04), ('shift_neg', 'shift_neg', .04)]
    if args.stage == 'calibration':
        variants = [('calibrated', 'calibrated', .04), ('pooled_mean', 'pooled_mean', .04)]
    write('protocol.json', dict(
        frozen_at=datetime.now(timezone.utc).isoformat(), checkpoint=args.checkpoint,
        stage=args.stage,
        calibration=str(args.calibration.resolve()) if args.calibration else None,
        calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest() if args.calibration else None,
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        method=dict(mode='deterministic', margin=.04, penalty=.08, uncertainty=0.),
        update=(dict(cone_degrees=3, visibility_tolerances_m=[.04, .08],
                    rule='Remove historical surfaces contradicted by aligned current RGB-predicted surface rays; retain occluded or unmatched history.')
                if args.stage == 'visibility' else
                dict(variants=['train-only metric translation', 'existing head pooled mean correction'],
                     rule='Correct each fresh point set once, then recompute original workspace mask; no proposal change.')),
        interventions=(dict(axis='robot base x', shifts_m=[.10, -.10],
                           scope='Only clearance surfaces; frozen proposal, point identities, initial ROI masks, and schedule unchanged.')
                       if args.stage == 'visibility' else None),
        selection='Update must strictly improve 85/100 validation success. Ties among improvements: fewer collisions, then larger tolerance, then listed candidate order. Otherwise retain original union memory.',
        test='Both fixed alignment interventions in visibility stage; only validation-selected update if it improves. No further tuning.',
        prior_test_seen=True, previous_archive=str(previous),
        source_sha256={str(p.relative_to(root/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (root/'source').rglob('*') if p.is_file()}))
    for split in ('validation', 'test'):
        (root/split).mkdir()
        for name in ('baseline', 'method'):
            (root/split/name).symlink_to((previous/split/name).resolve(), target_is_directory=True)
    gpu_pool = Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)

    def run(split, name, update, tolerance):
        gpu = gpu_pool.get()
        try:
            command = [sys.executable, str(source/'scripts/docker_run.py'),
                '--source-snapshot', str(root/'source'), '--gpu', gpu, 'evaluate',
                '--checkpoint', args.checkpoint, '--partition', split,
                '--clearance-mode', 'deterministic', '--clearance-margin', '.04',
                '--clearance-uncertainty', '0', '--clearance-penalty', '.08',
                '--geometry-update', update, '--visibility-tolerance', str(tolerance),
                '--output-dir', str(root/split/name)]
            if update == 'calibrated':
                command += ['--geometry-calibration', str(args.calibration.resolve())]
            with (root/'logs'/f'{split}_{name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'Completed {split}/{name}', flush=True)
        finally:
            gpu_pool.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        validations = [executor.submit(run, 'validation', *variant) for variant in variants]
        tests = [executor.submit(run, 'test', *variant) for variant in variants[2:]]
        for future in validations:
            future.result()
        def outcome(variant):
            data = json.loads((root/'validation'/variant[0]/'closed_loop.json').read_text())['overall']
            return data['success_rate'], -data['collision_rate'], variant[2]
        selected = max(variants[:2], key=outcome)
        reference = json.loads((root/'validation/method/closed_loop.json').read_text())['overall']['success_rate']
        improved = outcome(selected)[0] > reference
        write('selection.json', dict(selected=selected[0] if improved else 'method',
              criterion='strict validation success improvement over existing method',
              original_validation_success=reference,
              candidate_results={variant[0]:outcome(variant) for variant in variants[:2]},
              selected_update_test_started=False, known_test_used_in_this_selection=False))
        if improved:
            tests.append(executor.submit(run, 'test', *selected))
        for future in tests:
            future.result()
    for split in ('validation', 'test'):
        command = [sys.executable, str(source/'scripts/docker_run.py'),
            '--source-snapshot', str(root/'source'), '--cpu', 'report',
            '--root', str(root/split), '--reference', str(root/split/'baseline'),
            '--output-dir', str(root/f'report_{split}')]
        with (root/'logs'/f'report_{split}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    write('complete.json', dict(new_validation_rollouts=100*len(variants),
          new_test_rollouts=100*(len(variants[2:])+int(improved)),
          selected=selected[0] if improved else 'method',
          finished_at=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
