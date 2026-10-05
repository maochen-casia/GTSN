"""Run a paired validation set in isolated project Docker containers.

Host Python uses only the standard library. All model/simulator execution is
inside Docker. Four idle GPUs may be selected; no existing container is touched.
"""
import argparse
import json
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--image', default='gtsn-persistent:20261005')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--full-validation', '--full-partition', dest='full_validation', action='store_true')
    parser.add_argument('--partition', choices=('validation', 'test'), default='validation')
    parser.add_argument('--controller', choices=('clearance', 'compact'), default='clearance')
    args = parser.parse_args()
    if args.partition == 'test' and not args.full_validation:
        parser.error('Test evaluation uses the entire fixed partition; pass --full-partition')
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if not output.is_relative_to(Path('/run/user/1016/experiments')):
        parser.error('Experiment outputs must be under /run/user/1016/experiments')
    if output.exists():
        parser.error('Output directory already exists')
    output.mkdir(parents=True)
    protocol = json.loads((args.analysis/'rollout_protocol.json').read_text())
    protocol['partition'] = args.partition
    protocol['controller'] = args.controller
    if args.controller == 'compact':
        protocol.pop('clearance_margin', None)
        protocol.pop('clearance_penalty', None)
        protocol['clearance'] = dict(mode='disabled')
        protocol['geometry_refiner'] = 'none; CompactPolicy identity refinement'
    if args.full_validation:
        protocol['episodes'] = None
        protocol['selection'] = f'entire fixed {args.partition} split'
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    devices = args.gpus.split(',')
    variants = list(protocol['checkpoints'])
    if len(devices) < len(variants):
        parser.error('Select one available GPU per variant')
    def run(item):
        index, variant = item
        command = ['python3', str(root/'scripts/docker_run.py'), '--image', args.image, '--gpu', devices[index],
                   'evaluate', '--checkpoint', protocol['checkpoints'][variant], '--partition', args.partition,
                   '--output-dir', str(output/variant), '--no-render-videos', '--controller', args.controller]
        if protocol['episodes'] is not None:
            for episode in protocol['episodes']:
                command += ['--episode', episode]
        with (output/f'{variant}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        print('complete', variant, flush=True)
    with ThreadPoolExecutor(max_workers=len(variants)) as executor:
        list(executor.map(run, enumerate(variants)))
    summaries = {v: json.loads((output/v/'closed_loop.json').read_text()) for v in variants}
    (output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    (output/'complete.json').write_text(json.dumps(dict(variants=variants, full_validation=args.full_validation))+'\n')


if __name__ == '__main__':
    main()
