"""Reproduce validation selection, matched ablations, and fixed test in Docker.

Host code uses only the standard library. Every numerical process runs through
docker_run.py; only the new experiment directory is writable in each container.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4')
    parser.add_argument('--checkpoint', default='/run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt')
    args = parser.parse_args()
    root = args.root.resolve()
    source = Path(__file__).resolve().parents[1]
    allowed = (source/'runs', Path('/run/user/1016/experiments'))
    if not any(root.is_relative_to(p) and root != p for p in allowed) or root.exists():
        parser.error('Use a new directory within project runs/ or experiment storage')
    gpus = args.gpus.split(',')
    root.mkdir(parents=True)
    (root/'logs').mkdir()
    for folder in ('src', 'scripts', 'tests', 'configs'):
        shutil.copytree(source/folder, root/'source'/folder)
    def write(name, value):
        (root/name).write_text(json.dumps(value, indent=2)+'\n')
    write('protocol.json', dict(checkpoint=args.checkpoint, penalties=[.03, .08, .15],
          selection='full-validation success, then fewer collisions, then larger correction penalty',
          test_used_for_selection=False, historical_test_already_known=True,
          source_sha256={str(p.relative_to(root/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (root/'source').rglob('*') if p.is_file()}))
    subprocess.run([sys.executable, str(source/'scripts/docker_run.py'), '--source-snapshot', str(root/'source'), '--cpu', 'calibrate_clearance',
                    '--checkpoint', args.checkpoint, '--cache', str(Path(args.checkpoint).parent/'cache'),
                    '--output-dir', str(root/'calibration')], check=True)
    constant_radius = json.loads((root/'calibration/calibration.json').read_text())['constant_radius_m']

    def run(spec):
        index, partition, name, mode, penalty = spec
        command = [sys.executable, str(source/'scripts/docker_run.py'), '--source-snapshot', str(root/'source'), '--gpu', gpus[index % len(gpus)],
                   'evaluate', '--checkpoint', args.checkpoint, '--partition', partition,
                   '--output-dir', str(root/partition/name)]
        if mode:
            command += ['--clearance-mode', 'deterministic' if mode == 'constant' else mode,
                        '--clearance-penalty', str(penalty)]
            if mode == 'constant':
                command += ['--clearance-margin', str(constant_radius)]
        with (root/'logs'/f'{partition}_{name}.log').open('w') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        return name

    def batch(specs):
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            for name in executor.map(run, specs):
                print('Completed', name, flush=True)

    penalties = [.03, .08, .15]
    batch([(i, 'validation', f'full_p{int(p*1000):03}', 'full', p) for i, p in enumerate(penalties)]+
          [(3, 'validation', 'baseline', None, 0)])
    def outcome(p):
        data = json.loads((root/'validation'/f'full_p{int(p*1000):03}'/'closed_loop.json').read_text())['overall']
        return data['success_rate'], -data['collision_rate'], p
    selected = max(penalties, key=outcome)
    batch([(i, 'validation', mode, mode, selected) for i, mode in enumerate(('deterministic', 'current', 'tcp', 'constant'))])
    write('selection.json', dict(penalty=selected, mode='full', criterion='validation only', test_started=False))
    batch([(i, 'test', mode, None if mode == 'baseline' else mode, selected)
           for i, mode in enumerate(('full', 'deterministic', 'current', 'tcp', 'constant', 'baseline'))])
    for partition in ('validation', 'test'):
        subprocess.run([sys.executable, str(source/'scripts/docker_run.py'), '--source-snapshot', str(root/'source'), '--cpu', 'report',
                        '--root', str(root/partition), '--reference', str(root/partition/'baseline'),
                        '--output-dir', str(root/f'report_{partition}')], check=True)
    write('complete.json', dict(penalty=selected, validation_episodes=800, test_episodes=600))


if __name__ == '__main__':
    main()
