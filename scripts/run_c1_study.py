"""Run the C1 persistent-geometry comparison with immutable source and inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('validation', 'test', 'report'), required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('/run/user/1016/experiments/gtsn_geometric_energy_20261007/training/energy.pt'))
    parser.add_argument('--conditions', default='current,recent,persistent,unconfirmed,persistent_visual')
    parser.add_argument('--candidate', default='unconfirmed')
    parser.add_argument('--gpus', default='0,1,2,3,4')
    parser.add_argument('--image', default='gtsn-persistent:20261005-compact-only')
    parser.add_argument('--reuse-current', type=Path, help='Reuse a compatible completed strict current-frame control')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path('/run/user/1016/experiments')):
        parser.error('Use designated experiment storage')
    root.mkdir(parents=True, exist_ok=True)
    conditions = args.conditions.split(',')
    checkpoint = args.checkpoint.resolve()
    protocol = root/'protocol.json'
    snapshot = root/'source'
    if not protocol.exists():
        if args.stage != 'validation':
            parser.error('Complete validation before test')
        snapshot.mkdir()
        for folder in ('src', 'scripts', 'tests', 'configs'):
            shutil.copytree(ROOT/folder, snapshot/folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        write(root/'source_sha256.json', {str(p.relative_to(snapshot)): digest(p)
            for p in sorted(snapshot.rglob('*')) if p.is_file()})
        write(protocol, dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), image=args.image,
            image_id=subprocess.check_output(['docker', 'image', 'inspect', args.image,
                                             '--format', '{{.Id}}'], text=True).strip(),
            validation_conditions=conditions, splits='Existing 800/100/100; 2:4:4 per split',
            target='Persistent C1 success >= current-frame-only success + 3 percentage points on both validation and test',
            current='One RGB frame, one surface cloud, no earlier visual features, no map',
            training='Unchanged expert + independent perturbation-only route-trust checkpoint',
            control='Same wrist proposal, 14 candidates, swept-hand queries, trust, IK, servo, fixed 15-step execution',
            selection='Validation success; prefer geometry-only change over longer visual history in a tie',
            test='Freeze selected architecture and matched controls before test; historically reused test is exploratory'))
    if args.reuse_current and args.stage in ('validation', 'test'):
        source = args.reuse_current.resolve()/args.stage/'current'
        destination = root/args.stage/'current'
        if not (source/'complete.json').exists():
            parser.error('Reused current-frame control must be complete')
        original_protocol = json.loads((args.reuse_current.resolve()/'protocol.json').read_text())
        if original_protocol['checkpoint_sha256'] != digest(checkpoint) or original_protocol['image'] != args.image:
            parser.error('Reused control checkpoint or image differs')
        config = json.loads((source/'config.json').read_text())
        if (config['controller'] != 'adaptive_geometry' or
                config['clearance']['memory_mode'] != 'current' or
                config['clearance']['memory']['route_feature_frames'] != 1 or
                config['checkpoint'] != str(checkpoint)):
            parser.error('Reused control is incompatible')
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source, target_is_directory=True)
        write(root/('reused_current_'+args.stage+'.json'), dict(source=str(source),
            checkpoint_sha256=digest(checkpoint), summary_sha256=digest(source/'closed_loop.json'),
            source_manifest_sha256=digest(args.reuse_current.resolve()/'source_sha256.json'),
            compatibility='Current mode has identical forward behavior; only additional geometry modes were added'))
    saved = json.loads(protocol.read_text())
    if saved['checkpoint'] != str(checkpoint) or saved['checkpoint_sha256'] != digest(checkpoint):
        parser.error('Frozen checkpoint mismatch')
    for name, expected in json.loads((root/'source_sha256.json').read_text()).items():
        if digest(snapshot/name) != expected:
            parser.error('Source snapshot mismatch: '+name)
    if args.stage == 'test':
        if not all((root/'validation'/c/'complete.json').exists() for c in saved['validation_conditions']):
            parser.error('Finish every nominated validation condition first')
        baseline = json.loads((root/'validation/current/closed_loop.json').read_text())['overall']['success_rate']
        candidate = json.loads((root/'validation'/args.candidate/'closed_loop.json').read_text())['overall']['success_rate']
        if candidate < .70 or candidate-baseline < .03-1e-8:
            parser.error('Candidate failed the validation target; diagnose before test')
        freeze = dict(candidate=args.candidate, conditions=conditions,
            checkpoint_sha256=digest(checkpoint), source_manifest_sha256=digest(root/'source_sha256.json'),
            validation_current=baseline, validation_candidate=candidate,
            frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), no_test_selection=True)
        freeze_path = root/'test_freeze.json'
        if freeze_path.exists():
            previous = json.loads(freeze_path.read_text())
            for key in ('candidate', 'conditions', 'checkpoint_sha256', 'source_manifest_sha256'):
                if previous[key] != freeze[key]:
                    parser.error('Test nomination changed')
        else:
            write(freeze_path, freeze)
    if args.stage == 'report':
        for partition in ('validation', 'test'):
            if not (root/partition/args.candidate/'complete.json').exists():
                continue
            expected = ('results.json', 'audit.json', 'paired_comparisons.json',
                        'closed_loop.pdf', 'paired_effects.pdf')
            if all((root/'reports'/partition/name).exists() for name in expected):
                continue
            subprocess.run([sys.executable, str(ROOT/'scripts/docker_run.py'), '--image', args.image,
                '--source-snapshot', str(snapshot), '--cpu', 'report', '--root', str(root/partition),
                '--reference', str(root/partition/'current'), '--candidate', args.candidate,
                '--output-dir', str(root/'reports'/partition)], check=True)
        return
    gpus = args.gpus.split(',')
    def run(item):
        index, condition = item
        output = root/args.stage/condition
        if (output/'complete.json').exists():
            return dict(condition=condition, already_complete=True)
        with (root/(args.stage+'_'+condition+'.log')).open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/docker_run.py'), '--image', args.image,
                '--source-snapshot', str(snapshot), '--gpu', gpus[index % len(gpus)], 'evaluate',
                '--controller', 'adaptive_geometry', '--memory-mode', condition,
                '--checkpoint', str(checkpoint), '--partition', args.stage, '--no-render-videos',
                '--output-dir', str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
        return dict(condition=condition,
            overall=json.loads((output/'closed_loop.json').read_text())['overall'])
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for result in pool.map(run, enumerate(conditions)):
            print(json.dumps(result), flush=True)
    completed = sorted(p.parent.name for p in (root/args.stage).glob('*/complete.json'))
    write(root/(args.stage+'_complete.json'), dict(requested_conditions=conditions,
        completed_conditions=completed, requested_rollouts=100*len(conditions),
        full_validation_panel_complete=all((root/'validation'/c/'complete.json').exists()
                                          for c in saved['validation_conditions'])))


if __name__ == '__main__':
    main()
