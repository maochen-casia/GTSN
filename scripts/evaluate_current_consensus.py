"""Snapshot the current best consensus checkpoint and evaluate without touching training."""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def launch(args):
    source = args.training_root.resolve()
    root = args.root or source.with_name(source.name+'_interim_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    root = root.resolve()
    if root.parent != Path('/run/user/1016/experiments') or root == source:
        raise ValueError('Evaluation needs a fresh sibling experiment directory')
    root.mkdir(exist_ok=False)
    (root/'train').mkdir()
    # Training atomically replaces best.pt. One open file descriptor pins a
    # complete checkpoint even if a newer best is published during the copy.
    with (source/'train/best.pt').open('rb') as original, (root/'train/best.pt').open('xb') as copy:
        shutil.copyfileobj(original, copy, length=8*1024*1024)
    shutil.copy2(__file__, root/'evaluate_current_consensus.py')
    shutil.copy2(source/'protocol.json', root/'training_protocol.json')
    original_launch = json.loads((source/'launch.json').read_text())
    image = original_launch['image_id']
    name = root.name.replace('_', '-')
    command = ['docker', 'run', '-d', '--name', name, '--init', '--network', 'none', '--read-only',
               '--user', f'{os.getuid()}:{os.getgid()}', '--gpus', f'device={args.gpu}',
               '--shm-size', '2g', '--tmpfs', '/tmp:rw,exec,size=2g']
    for variable in ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                     'LP_NUM_THREADS=1', 'MPLCONFIGDIR=/tmp/matplotlib',
                     'XDG_CACHE_HOME=/tmp/cache', 'PYTHONDONTWRITEBYTECODE=1']:
        command += ['--env', variable]
    for host, target, readonly in [(source/'source', Path('/workspace'), True),
                                    (root, root, False),
                                    (root/'train/best.pt', root/'train/best.pt', True),
                                    (Path('/run/user/1016/tsn-1k'), Path('/run/user/1016/tsn-1k'), True)]:
        command += ['--mount', f'type=bind,source={host},target={target}'+(',readonly' if readonly else '')]
    command += [image, 'python', str(root/'evaluate_current_consensus.py'), '--inside',
                '--root', str(root), '--training-root', str(source)]
    (root/'launch.json').write_text(json.dumps(dict(command=command, container=name,
        gpu=args.gpu, training_root=str(source), image_id=image), indent=2)+'\n')
    subprocess.run(command, check=True)
    print(json.dumps(dict(output=str(root), container=name, gpu=args.gpu)), flush=True)


def evaluate(args):
    sys.path.insert(0, '/workspace/scripts')
    from run_fresh_consensus import sha256, source_hashes
    from tsn.common.checkpoint import load_checkpoint
    from tsn.common.config import read_json, write_json
    root = args.root
    processes = []
    try:
        write_json(root/'status.json', dict(state='running', stage='verify_snapshot'))
        assert source_hashes() == read_json(root/'training_protocol.json')['source_sha256']
        path = root/'train/best.pt'
        digest = sha256(path)
        checkpoint = load_checkpoint(path)
        epoch = checkpoint['epoch']
        selection = dict(checkpoint_sha256=digest, conditions={'state_memory': ['state', 'memory']},
                         primary='state_memory', checkpoint_epoch=epoch,
                         validation_waypoint_rmse_m=checkpoint['validation']['state_memory'],
                         training_root=str(args.training_root),
                         criterion='best validation waypoint RMSE at snapshot time',
                         test_used_for_selection=False, execute=15, servo=.08,
                         temporal=0., adaptive=False, interim=True,
                         note='This evaluation does not alter ongoing training or final checkpoint selection.')
        write_json(root/'selection.json', selection)
        del checkpoint
        gc.collect()
        print(json.dumps(selection), flush=True)
        (root/'logs').mkdir()
        # Both small inference jobs share the otherwise idle GPU. Training GPUs
        # are not visible in this container.
        for partition in ('validation', 'test'):
            log = (root/'logs'/f'{partition}.log').open('w')
            command = [sys.executable, '/workspace/scripts/run_fresh_consensus.py',
                       '--root', str(root), '--stage', 'rollout',
                       '--partition', partition, '--condition', 'state_memory']
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append((partition, process, log))
        while True:
            progress = {partition: dict(exit_code=process.poll(), completed_episodes=len(list(
                (root/'rollouts'/partition/'state_memory'/'episodes').glob('*/metrics.json'))))
                for partition, process, _ in processes}
            write_json(root/'status.json', dict(state='running', stage='closed_loop',
                       checkpoint_epoch=epoch, partitions=progress))
            if any(process.poll() not in (None, 0) for _, process, _ in processes):
                raise RuntimeError('An evaluation worker failed; see partition logs')
            if all(process.poll() == 0 for _, process, _ in processes):
                break
            time.sleep(10)
        assert sha256(path) == digest
        results = {}
        for partition, _, _ in processes:
            output = root/'rollouts'/partition/'state_memory'
            assert read_json(output/'complete.json')['episodes'] == 100
            closed = read_json(output/'closed_loop.json')
            results[partition] = {key: value for key, value in closed.items() if key != 'results'}
        write_json(root/'results.json', dict(checkpoint_epoch=epoch, checkpoint_sha256=digest,
                   interim=True, **results))
        write_json(root/'status.json', dict(state='complete', checkpoint_epoch=epoch,
                   validation_episodes=100, test_episodes=100))
        print(json.dumps(results), flush=True)
    except Exception:
        write_json(root/'status.json', dict(state='failed', traceback=traceback.format_exc()))
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _, process, log in processes:
            process.wait()
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path,
                        default=Path('/run/user/1016/experiments/gtsn_consensus_20261003_fresh'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    if args.inside:
        evaluate(args)
    else:
        launch(args)


if __name__ == '__main__':
    main()
