"""Frozen hand/tool-volume ablations against strict TCP-only clearance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
C3 = Path('/run/user/1016/experiments/gtsn_c3_uncertainty_20261007')
CONDITIONS = dict(tcp=('tcp',0.),axial=('axial',1.),hand35=('hand',.35),
                  hand70=('hand',.7),tool35=('tool',.35),tool70=('tool',.7),
                  hand_mask35=('hand',.35,True,'fixed'),tool_mask35=('tool',.35,True,'fixed'),
                  hand_roll35=('hand',.35,True,'roll'),tool_roll35=('tool',.35,True,'wide_roll'),
                  tool_roll70=('tool',.7,True,'wide_roll'),
                  tcp_signed=('tcp',0.,False,'fixed','signed'),
                  hand_signed35=('hand',.35,True,'fixed','signed'),
                  hand_signed70=('hand',.7,True,'fixed','signed'),
                  tool_signed35=('tool',.35,True,'fixed','signed'),
                  tool_signed70=('tool',.7,True,'fixed','signed'),
                  tool_pose_signed35=('tool',.35,True,'wide_roll','signed'))
CONDITIONS.update(hand_parts35=('hand',.35,True,'fixed','gaussian','parts'),
                  hand_parts70=('hand',.7,True,'fixed','gaussian','parts'),
                  tool_parts35=('tool',.35,True,'fixed','gaussian','parts'),
                  tool_parts70=('tool',.7,True,'fixed','gaussian','parts'),
                  tool_parts100=('tool',1.,True,'fixed','gaussian','parts'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('validation','test','report'),required=True)
    parser.add_argument('--conditions',default=','.join(CONDITIONS))
    parser.add_argument('--candidate')
    parser.add_argument('--reference')
    parser.add_argument('--checkpoint',type=Path,default=C3/'training/uncertainty.pt')
    parser.add_argument('--reuse-axial',type=Path,default=C3)
    parser.add_argument('--reuse-tcp',type=Path)
    parser.add_argument('--gpus',default='0,1,2,3,4')
    parser.add_argument('--image',default='gtsn-persistent:20261005-compact-only')
    args = parser.parse_args()
    root,checkpoint = args.root.resolve(),args.checkpoint.resolve()
    if not root.is_relative_to(Path('/run/user/1016/experiments')):
        parser.error('Use authorized experiment storage')
    conditions = args.conditions.split(',')
    if len(set(conditions))!=len(conditions) or set(conditions)-CONDITIONS.keys():
        parser.error('Unknown or repeated condition')
    root.mkdir(parents=True,exist_ok=True)
    source = root/'source'
    if not (root/'protocol.json').exists():
        if args.stage!='validation':parser.error('Validation must precede test')
        reference = args.reference or 'tcp'
        if reference not in conditions or CONDITIONS[reference][0]!='tcp':
            parser.error('Validation must include the matching TCP-only reference')
        source.mkdir()
        for name in ('src','scripts','tests','configs'):
            shutil.copytree(PROJECT/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write(root/'source_sha256.json',{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()})
        write(root/'protocol.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),image=args.image,
            image_id=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip(),
            validation_conditions=conditions,condition_config=CONDITIONS,reference=reference,
            primary='Hand or tool geometry >= strict TCP-only clearance +3 percentage points on both full validation and test',
            controls='TCP queries only one reference point and never evaluates hand/tool volume; axial reproduces completed C3',
            selection='Highest validation success among hand/tool modes; smaller body weight, simpler posture, then hand mode break ties',
            test_rule='Freeze selected candidate and all test controls before reading test outcomes',
            data='Unchanged expert+independent perturbation checkpoints; no additional training or data',
            shared='C1 persistent map; C3 learned uncertainty with 30 mm maximum; same route/trust, lattice, IK, servo and fixed15',
            splits='Original 800/100/100; 160/320/320 training and 20/40/40 evaluation route counts',
            geometry='URDF collision convex support planes and oriented boxes; measured prismatic finger positions',
            test_interpretation='Historically reused exploratory benchmark'))
    protocol = json.loads((root/'protocol.json').read_text())
    reference = protocol.get('reference','tcp')
    if args.reference and args.reference!=reference:parser.error('Frozen reference changed')
    if digest(checkpoint)!=protocol['checkpoint_sha256']:parser.error('Checkpoint changed')
    for name,h in json.loads((root/'source_sha256.json').read_text()).items():
        if digest(source/name)!=h:parser.error('Frozen source changed: '+name)
    if args.reuse_tcp and 'tcp' in conditions and args.stage in ('validation','test'):
        original = args.reuse_tcp.resolve()/args.stage/'tcp'
        saved = json.loads((args.reuse_tcp/'protocol.json').read_text())
        if not (original/'complete.json').exists() or saved['checkpoint_sha256']!=protocol['checkpoint_sha256']:
            parser.error('Compatible completed TCP control required')
        target = root/args.stage/'tcp'
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(original,target_is_directory=True)
        write(root/('reused_tcp_'+args.stage+'.json'),dict(source=str(original),summary_sha256=digest(original/'closed_loop.json'),
            parity='TCP skips self filtering and prunes dominated posture alternatives; complete command equality tested in Docker'))
    if 'axial' in conditions and args.stage in ('validation','test'):
        original = args.reuse_axial.resolve()/args.stage/'adaptive30'
        cfg = json.loads((original/'config.json').read_text())
        saved = json.loads((args.reuse_axial/'protocol.json').read_text())
        if (not (original/'complete.json').exists() or saved['image']!=args.image or
                saved['checkpoint_sha256']!=protocol['checkpoint_sha256'] or
                cfg['clearance']['refinement']['max_padding_m']!=.03):
            parser.error('Compatible completed C3 axial reference required')
        target = root/args.stage/'axial'
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(original,target_is_directory=True)
        write(root/('reused_axial_'+args.stage+'.json'),dict(source=str(original),summary_sha256=digest(original/'closed_loop.json'),
            parity='Exact complete-command equality of axial mode and C3 adaptive30 verified in Docker tests'))
    if args.stage=='test':
        if not all((root/'validation'/c/'complete.json').exists() for c in protocol['validation_conditions']):
            parser.error('Finish every validation condition before test')
        candidates = [c for c in protocol['validation_conditions'] if CONDITIONS[c][0] in ('hand','tool')]
        def score(c):
            result = json.loads((root/'validation'/c/'closed_loop.json').read_text())['overall']['success_rate']
            posture = CONDITIONS[c][3] if len(CONDITIONS[c])>2 else 'fixed'
            return result,-CONDITIONS[c][1],-('fixed','roll','wide_roll').index(posture),CONDITIONS[c][0]=='hand'
        nominee = max(candidates,key=score)
        if args.candidate!=nominee:parser.error('Validation-selected candidate is '+nominee)
        selected = score(nominee)[0]
        baseline = json.loads((root/'validation'/reference/'closed_loop.json').read_text())['overall']['success_rate']
        if selected<.70 or selected-baseline<.03-1e-8:parser.error('Candidate failed the validation target')
        freeze = dict(candidate=nominee,conditions=conditions,condition_config={c:CONDITIONS[c] for c in conditions},
            validation_success=selected,validation_tcp_success=baseline,checkpoint_sha256=digest(checkpoint),
            source_manifest_sha256=digest(root/'source_sha256.json'),
            frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_test_selection=True)
        path = root/'test_freeze.json'
        if path.exists():
            old = json.loads(path.read_text())
            if any(old[k]!=freeze[k] for k in ('candidate','conditions','checkpoint_sha256','source_manifest_sha256')):
                parser.error('Test nomination changed')
        else:write(path,freeze)
    if args.stage=='report':
        for split in ('validation','test'):
            if not args.candidate or not (root/split/args.candidate/'complete.json').exists():continue
            output = root/'reports'/split
            if (output/'paired_effects.pdf').exists():continue
            subprocess.run([sys.executable,str(PROJECT/'scripts/docker_run.py'),'--image',args.image,
                '--source-snapshot',str(source),'--cpu','report','--root',str(root/split),
                '--reference',str(root/split/reference),'--candidate',args.candidate,'--output-dir',str(output)],check=True)
        return
    pending = [c for c in conditions if not (root/args.stage/c/'complete.json').exists()]
    gpus = args.gpus.split(',')
    def run(item):
        index,condition = item
        mode,weight,*extra = CONDITIONS[condition]
        options = []
        if extra:
            options += ['--body-pose',extra[1]]
            if extra[0]:options += ['--body-self-mask']
            if len(extra)>2:options += ['--body-field',extra[2]]
            if len(extra)>3:options += ['--body-representation',extra[3]]
        output = root/args.stage/condition
        with (root/(args.stage+'_'+condition+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(PROJECT/'scripts/docker_run.py'),'--image',args.image,
                '--source-snapshot',str(source),'--gpu',gpus[index%len(gpus)],'evaluate',
                '--controller','embodied_clearance','--body-mode',mode,'--body-weight',str(weight),*options,
                '--uncertainty-padding','.03','--checkpoint',str(checkpoint),'--partition',args.stage,
                '--no-render-videos','--output-dir',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
        return dict(condition=condition,overall=json.loads((output/'closed_loop.json').read_text())['overall'])
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for result in pool.map(run,enumerate(pending)):print(json.dumps(result),flush=True)
    write(root/(args.stage+'_complete.json'),dict(conditions=conditions,new_rollouts=100*len(pending)))


if __name__=='__main__':main()
