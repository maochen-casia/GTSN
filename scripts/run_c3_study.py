"""Frozen C3 uncertainty-refinement panel with a matched route-only control."""
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
CONDITIONS = dict(none=('none', 0.), fixed=('fixed', 0.),
                  adaptive10=('adaptive', .01), adaptive20=('adaptive', .02),
                  adaptive30=('adaptive', .03), uniform20=('uniform', .02),
                  uniform30=('uniform', .03))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('validation', 'test', 'report'), required=True)
    parser.add_argument('--conditions', default=','.join(CONDITIONS))
    parser.add_argument('--candidate', default='adaptive20')
    parser.add_argument('--gpus', default='0,1,2,3,4')
    parser.add_argument('--image', default='gtsn-persistent:20261005-compact-only')
    parser.add_argument('--reuse-fixed', type=Path, default=Path('/run/user/1016/experiments/gtsn_c1_consensus_20261007'))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path('/run/user/1016/experiments')):
        parser.error('Use designated experiment storage')
    conditions = args.conditions.split(',')
    if len(conditions) != len(set(conditions)) or set(conditions)-CONDITIONS.keys():
        parser.error('Unknown or repeated condition')
    checkpoint = root/'training/uncertainty.pt'
    if not checkpoint.exists():
        parser.error('Complete fixed-epoch uncertainty training first')
    protocol_path = root/'protocol.json'
    snapshot = root/'source'
    if not protocol_path.exists():
        if args.stage != 'validation':
            parser.error('Validation must precede test')
        snapshot.mkdir()
        for folder in ('src','scripts','tests','configs'):
            shutil.copytree(ROOT/folder, snapshot/folder, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write(root/'source_sha256.json', {str(p.relative_to(snapshot)):digest(p) for p in snapshot.rglob('*') if p.is_file()})
        write(protocol_path, dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), image=args.image,
            image_id=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip(),
            validation_conditions=conditions, condition_config=CONDITIONS,
            primary='Uncertainty-aware refinement >= no refinement +3 percentage points on both full validation and test splits',
            controls='none keeps four visual frames and the same C1 map/proposal/IK/servo; fixed is selected C1 without inflation; uniform matches mean training padding factor',
            selection='Highest validation success among adaptive variants; smaller padding breaks ties',
            test_rule='Freeze selected candidate and all test controls before test; do not choose from test outcomes',
            splits='Original 800/100/100 with 160/320/320 and 20/40/40 route counts',
            data='Existing expert+perturbation-only observations; original depth labels supervise uncertainty only',
            test_interpretation='Historically reused exploratory benchmark'))
    protocol = json.loads(protocol_path.read_text())
    if digest(checkpoint) != protocol['checkpoint_sha256']:
        parser.error('Frozen uncertainty checkpoint changed')
    for filename, expected in json.loads((root/'source_sha256.json').read_text()).items():
        if digest(snapshot/filename) != expected:
            parser.error('Frozen source changed: '+filename)
    if args.stage in ('validation','test') and 'fixed' in conditions:
        original = args.reuse_fixed.resolve()/args.stage/'unconfirmed'
        target = root/args.stage/'fixed'
        saved = json.loads((args.reuse_fixed/'protocol.json').read_text())
        if not (original/'complete.json').exists() or saved['image'] != args.image:
            parser.error('Compatible completed C1 reference required')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(original, target_is_directory=True)
        write(root/('reused_fixed_'+args.stage+'.json'), dict(source=str(original),
            summary_sha256=digest(original/'closed_loop.json'), original_checkpoint_sha256=saved['checkpoint_sha256'],
            source_manifest_sha256=digest(args.reuse_fixed/'source_sha256.json'),
            parity='Fixed C3 mode delegates to C1 refinement; complete command equality checked in Docker regression tests'))
    if args.stage == 'test':
        if not all((root/'validation'/c/'complete.json').exists() for c in protocol['validation_conditions']):
            parser.error('Complete every validation condition before test')
        adaptive = [c for c in protocol['validation_conditions'] if CONDITIONS[c][0] == 'adaptive']
        nominee = max(adaptive, key=lambda c:(json.loads((root/'validation'/c/'closed_loop.json').read_text())['overall']['success_rate'], -CONDITIONS[c][1]))
        if args.candidate != nominee:
            parser.error('Candidate differs from the validation selection rule: '+nominee)
        baseline = json.loads((root/'validation/none/closed_loop.json').read_text())['overall']['success_rate']
        chosen = json.loads((root/'validation'/nominee/'closed_loop.json').read_text())['overall']['success_rate']
        if chosen < .70 or chosen-baseline < .03-1e-8:
            parser.error('Adaptive nominee failed the validation target')
        freeze_path = root/'test_freeze.json'
        freeze = dict(candidate=nominee, conditions=conditions, validation_candidate=chosen,
                      condition_config={c: CONDITIONS[c] for c in conditions},
                      validation_none=baseline, checkpoint_sha256=digest(checkpoint),
                      source_manifest_sha256=digest(root/'source_sha256.json'),
                      frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), no_test_selection=True)
        if freeze_path.exists():
            old = json.loads(freeze_path.read_text())
            if any(old[k] != freeze[k] for k in ('candidate','conditions','checkpoint_sha256','source_manifest_sha256')):
                parser.error('Test nomination changed')
        else:
            write(freeze_path, freeze)
    if args.stage == 'report':
        for partition in ('validation','test'):
            if not (root/partition/args.candidate/'complete.json').exists():
                continue
            expected = ('results.json','audit.json','paired_comparisons.json','closed_loop.pdf','paired_effects.pdf')
            if all((root/'reports'/partition/name).exists() for name in expected):
                continue
            subprocess.run([sys.executable,str(ROOT/'scripts/docker_run.py'),'--image',args.image,
                '--source-snapshot',str(snapshot),'--cpu','report','--root',str(root/partition),
                '--reference',str(root/partition/'none'),'--candidate',args.candidate,
                '--output-dir',str(root/'reports'/partition)],check=True)
        return
    pending = [c for c in conditions if not (root/args.stage/c/'complete.json').exists()]
    gpus = args.gpus.split(',')
    def run(item):
        index, condition = item
        mode, padding = CONDITIONS[condition]
        output = root/args.stage/condition
        with (root/(args.stage+'_'+condition+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/docker_run.py'),'--image',args.image,
                '--source-snapshot',str(snapshot),'--gpu',gpus[index % len(gpus)],'evaluate',
                '--controller','uncertain_clearance','--refinement-mode',mode,'--uncertainty-padding',str(padding),
                '--checkpoint',str(checkpoint),'--partition',args.stage,'--no-render-videos',
                '--output-dir',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
        return dict(condition=condition,overall=json.loads((output/'closed_loop.json').read_text())['overall'])
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for result in pool.map(run, enumerate(pending)):
            print(json.dumps(result),flush=True)
    write(root/(args.stage+'_complete.json'), dict(conditions=conditions,new_rollouts=100*len(pending)))


if __name__ == '__main__':
    main()
