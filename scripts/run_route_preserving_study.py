"""Run the frozen, post-hoc deterministic-clearance study in Docker.

This study deliberately reuses previously evaluated baseline/method runs.
Its new controls remove one component each; no parameter selection is performed.
Host execution only coordinates files and Docker processes with the stdlib.
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
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--previous', type=Path, default=Path(
        '/run/user/1016/experiments/gtsn_research_20261003/clearance_20261003'))
    parser.add_argument('--gpus', default='0,1,2,3,4')
    parser.add_argument('--checkpoint', default=
        '/run/user/1016/experiments/gtsn_simplification_20261003/simplified.pt')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if root.exists() or not root.is_relative_to(Path('/run/user/1016/experiments')):
        parser.error('Use a fresh directory in /run/user/1016/experiments')
    gpu_ids = args.gpus.split(',')
    if len(set(gpu_ids)) != len(gpu_ids) or any(not g.isdigit() for g in gpu_ids):
        parser.error('GPU IDs must be distinct integers')
    previous = args.previous.resolve()
    references = {
        'validation': {'baseline': previous/'baseline', 'method': previous/'deterministic_p008'},
        'test': {'baseline': previous/'test/baseline', 'method': previous/'test/deterministic'},
    }
    for conditions in references.values():
        for directory in conditions.values():
            if not (directory/'complete.json').is_file():
                parser.error(f'Missing completed reference: {directory}')
    root.mkdir(parents=True)
    (root/'logs').mkdir()
    for folder in ('src', 'scripts', 'tests', 'configs'):
        shutil.copytree(source/folder, root/'source'/folder)

    def write(name, value):
        (root/name).write_text(json.dumps(value, indent=2)+'\n')

    controls = [('without_surface_memory', 'current', .08),
                ('without_hand_extent', 'tcp', .08),
                ('without_correction_penalty', 'deterministic', 0.)]
    write('protocol.json', dict(
        frozen_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=args.checkpoint, checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        method=dict(mode='deterministic', margin=.04, uncertainty=0., penalty=.08),
        controls=[dict(name=n, mode=m, penalty=p, uncertainty=0., margin=.04) for n, m, p in controls],
        research_question='Does local clearance improve a frozen route, and what is lost when memory, hand extent, or route preservation is removed?',
        selection='No tuning or selection in this study; all three controls reported on both splits.',
        historical_test_already_known=True,
        method_choice='Post-hoc interpretation of previous deterministic test 83/100 versus baseline 78/100; not a blind confirmation.',
        reused={split: {name: str(path) for name, path in conditions.items()} for split, conditions in references.items()},
        source_sha256={str(p.relative_to(root/'source')): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (root/'source').rglob('*') if p.is_file()}))
    for split, conditions in references.items():
        (root/split).mkdir()
        for name, path in conditions.items():
            (root/split/name).symlink_to(path, target_is_directory=True)
    available = Queue()
    for gpu in gpu_ids:
        available.put(gpu)

    def run(spec):
        split, name, mode, penalty = spec
        gpu = available.get()
        try:
            command = [sys.executable, str(source/'scripts/docker_run.py'),
                       '--source-snapshot', str(root/'source'), '--gpu', gpu, 'evaluate',
                       '--checkpoint', args.checkpoint, '--partition', split,
                       '--clearance-mode', mode, '--clearance-margin', '.04',
                       '--clearance-uncertainty', '0', '--clearance-penalty', str(penalty),
                       '--output-dir', str(root/split/name)]
            with (root/'logs'/f'{split}_{name}.log').open('w') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
            print(f'Completed {split}/{name}', flush=True)
        finally:
            available.put(gpu)

    specs = [(split, *control) for split in references for control in controls]
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        list(executor.map(run, specs))
    for split in references:
        command = [sys.executable, str(source/'scripts/docker_run.py'),
                   '--source-snapshot', str(root/'source'), '--cpu', 'report',
                   '--root', str(root/split), '--reference', str(root/split/'baseline'),
                   '--output-dir', str(root/f'report_{split}')]
        with (root/'logs'/f'report_{split}.log').open('w') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    write('complete.json', dict(new_validation_rollouts=300, new_test_rollouts=300,
                               reused_validation_rollouts=200, reused_test_rollouts=200,
                               finished_at=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
