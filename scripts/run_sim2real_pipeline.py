"""Complete training, full evaluations and audit in one detached Docker job."""
import argparse
import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from tsn.common.config import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--processes', type=int, default=4)
    args = parser.parse_args()
    root = args.root

    def status(phase, **details):
        value = dict(phase=phase, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details)
        write_json(root/'status.json', value)
        print(value, flush=True)

    def run(command, log, phase):
        with (root/log).open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                details = {}
                if phase == 'training' and (root/'training/progress.json').exists():
                    details['progress'] = read_json(root/'training/progress.json')
                status(phase, **details)
                time.sleep(30)
            if process.returncode:
                raise RuntimeError(f'{phase} failed with exit {process.returncode}; see {log}')

    try:
        while not (root/'recovery/manifest.json').exists():
            status('generating_recovery', archives=len(list((root/'recovery').glob('episode_*.npz'))))
            if 'Traceback (most recent call last)' in (root/'recovery.log').read_text():
                raise RuntimeError('Recovery generation failed; see recovery.log')
            time.sleep(30)
        run([sys.executable, str(root/'source/scripts/report_sim2real.py'), '--root', str(root),
             '--recovery-only'], 'recovery_audit.log', 'auditing_recovery')
        command = ['torchrun', '--master-addr=127.0.0.1', '--master-port=29500',
                   f'--nproc-per-node={args.processes}', '-m', 'tsn.cli.train',
                   '--config', str(root/'source/configs/sim2real.json'), '--output-dir', str(root/'training')]
        run(command, 'training.log', 'training')
        checkpoint = root/'training/best.pt'
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024), b''):
                digest.update(block)
        selected = min(read_json(root/'training/epochs.json'), key=lambda row:row['validation_rmse_m'])
        write_json(root/'selected_checkpoint.json', dict(checkpoint=str(checkpoint), sha256=digest.hexdigest(),
                   selected_epoch=selected['epoch'], validation_rmse_m=selected['validation_rmse_m'],
                   frozen_utc_before_test=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   test_used_for_selection=False))
        processes, handles = {}, []
        for device, partition in enumerate(('validation', 'test')):
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(device)}
            stream = (root/f'{partition}.log').open('w'); handles.append(stream)
            processes[partition] = subprocess.Popen([sys.executable, '-m', 'tsn.cli.evaluate',
                '--checkpoint', str(checkpoint), '--partition', partition, '--output-dir', str(root/partition),
                '--no-render-videos'], stdout=stream, stderr=subprocess.STDOUT, env=env)
        while any(process.poll() is None for process in processes.values()):
            counts = {name: len(read_json(root/name/'closed_loop.json')['results'])
                      if (root/name/'closed_loop.json').exists() else 0 for name in processes}
            status('evaluating', episodes_completed=counts)
            time.sleep(30)
        for handle in handles:
            handle.close()
        if any(process.returncode for process in processes.values()):
            raise RuntimeError('Full evaluation failed; see validation.log and test.log')
        run([sys.executable, str(root/'source/scripts/report_sim2real.py'), '--root', str(root)],
            'audit.log', 'auditing')
        summary = read_json(root/'summary.json')
        status('complete', selected_epoch=summary['selected_epoch'],
               success_rates={name: values['success_rate'] for name, values in summary['partitions'].items()})
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
