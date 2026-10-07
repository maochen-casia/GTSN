"""Run a fixed, auditable geometric-energy ablation panel in project Docker."""
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
CONDITIONS = ('full', 'no_history', 'tcp_only', 'no_trust', 'fixed_trust')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('validation', 'test', 'report'), required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--gpus', default='0,1,2,3,4')
    p.add_argument('--image', default='gtsn-persistent:20261005-compact-only')
    args = p.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path('/run/user/1016/experiments')):
        p.error('Experiment root must be within designated experiment storage')
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root/'protocol.json'
    checkpoint = (args.checkpoint or root/'training/energy.pt').resolve()
    if not checkpoint.is_file():
        p.error('Training checkpoint is required')
    if not protocol_path.exists():
        if args.stage != 'validation':
            p.error('Run the validation panel before accessing test')
        snapshot = root/'source'
        snapshot.mkdir()
        for folder in ('src', 'scripts', 'tests', 'configs'):
            shutil.copytree(ROOT/folder, snapshot/folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        hashes = {str(x.relative_to(snapshot)): digest(x) for x in sorted(snapshot.rglob('*')) if x.is_file()}
        write(root/'source_sha256.json', hashes)
        write(protocol_path, dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), image=args.image,
            conditions=CONDITIONS, selection='Fixed final training epoch; no rollout-based checkpoint tuning',
            splits='Existing 800/100/100; train 160/320/320; each evaluation 20/40/40',
            test_rule='Freeze checkpoint and source before test; evaluate all prespecified controls',
            contribution_rule='Report success differences separately on validation and test; never declare a tie an improvement',
            no_history='Only current surface cloud; four-frame proposal history stays intact',
            tcp_only='One TCP query replaces three axial hand probes; all other factors fixed',
            no_trust='Set route correction cost to zero; keep bounded candidates and goal fade',
            fixed_trust='Use constant .08; isolates whether learned context improves the existing clearance baseline',
            test_interpretation='Exploratory historically reused test split; freezing this run does not restore blindness'))
    protocol = json.loads(protocol_path.read_text())
    if str(checkpoint) != protocol['checkpoint'] or digest(checkpoint) != protocol['checkpoint_sha256']:
        p.error('Checkpoint identity differs from frozen protocol')
    snapshot = root/'source'
    for filename, expected in json.loads((root/'source_sha256.json').read_text()).items():
        if digest(snapshot/filename) != expected:
            p.error('Source snapshot changed: '+filename)
    if args.stage == 'test':
        if not all((root/'validation'/c/'complete.json').is_file() for c in CONDITIONS):
            p.error('Every prespecified validation condition must finish before test')
        summary = root/'validation/full/closed_loop.json'
        if not (root/'validation/full/complete.json').is_file():
            p.error('Complete validation is required before test')
        data = json.loads(summary.read_text())
        if data['overall']['success_rate'] < .70:
            p.error('Full method failed the validation threshold; diagnose before test')
        freeze = root/'test_freeze.json'
        if not freeze.exists():
            write(freeze, dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                checkpoint_sha256=digest(checkpoint), validation_success=data['overall']['success_rate'],
                source_manifest_sha256=digest(root/'source_sha256.json'), conditions=CONDITIONS,
                no_test_selection=True))
    if args.stage == 'report':
        for partition in ('validation', 'test'):
            expected = ('results.json', 'audit.json', 'paired_comparisons.json',
                        'closed_loop.pdf', 'paired_effects.pdf', 'paired_cases.pdf')
            if all((root/'reports'/partition/name).is_file() for name in expected):
                continue
            subprocess.run([sys.executable, str(ROOT/'scripts/docker_run.py'), '--image', args.image,
                '--source-snapshot', str(snapshot), '--cpu', 'report', '--root', str(root/partition),
                '--reference', str(root/partition/'fixed_trust'), '--candidate', 'full',
                '--output-dir', str(root/'reports'/partition)], check=True)
        return
    gpus = args.gpus.split(',')
    def evaluate(item):
        n, condition = item
        output = root/args.stage/condition
        if (output/'complete.json').exists():
            return condition+' already complete'
        log_path = root/(args.stage+'_'+condition+'.log')
        with log_path.open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/docker_run.py'), '--image', args.image,
                '--source-snapshot', str(snapshot), '--gpu', gpus[n % len(gpus)], 'evaluate',
                '--controller', 'geometric_energy', '--energy-ablation', condition,
                '--checkpoint', str(checkpoint), '--partition', args.stage, '--no-render-videos',
                '--output-dir', str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
        results = json.loads((output/'closed_loop.json').read_text())
        return dict(condition=condition, partition=args.stage, overall=results['overall'])
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for result in pool.map(evaluate, enumerate(CONDITIONS)):
            print(result, flush=True)
    write(root/(args.stage+'_complete.json'), dict(conditions=CONDITIONS, rollouts=500))


if __name__ == '__main__':
    main()
