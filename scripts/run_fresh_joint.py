"""Train, evaluate and audit a fresh joint replacement model inside Docker."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from safetensors import safe_open

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import read_json, write_json
from tsn.data.splits import make_splits, episode_catalog, validate_splits
from tsn.evaluation.metrics import summarize_rollouts
from run_geometry_study import digest, source_hashes, validate_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root, workers = args.root, args.workers
    source, config = root/'source', read_json(root/'config.json')
    splits = make_splits(config['benchmark'])
    validate_splits(splits, episode_catalog(Path(config['benchmark']['root'])))
    environment = {**os.environ, 'PYTHONPATH': f'{source}/src:{source}/vendor/Pi3'}
    torch.set_num_threads(1)

    def status(phase, **fields):
        value = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        write_json(root/'status.json', value)
        print(value, flush=True)

    def run(command, log, phase):
        with (root/log).open('w') as stream:
            process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                progress = root/'training/progress.json'
                status(phase, progress=read_json(progress) if progress.exists() else None)
                time.sleep(20)
            if process.returncode:
                raise RuntimeError(f'{phase} failed ({process.returncode}); see {log}')

    def training_command(path, output):
        return [sys.executable, '-m', 'torch.distributed.run', '--master-addr=127.0.0.1', '--master-port=29500',
                f'--nproc_per_node={workers}', '-m', 'tsn.cli.train',
                '--config', str(path), '--output-dir', str(output)]

    def evaluate(checkpoint, partition):
        destination = root/partition
        destination.mkdir()
        write_json(destination/'config.json', dict(checkpoint=str(checkpoint), sha256=digest(checkpoint),
            episodes=splits[partition], partition=partition, eval=config['eval']))
        queue = [splits[partition][i:i+5] for i in range(0, len(splits[partition]), 5)]
        active, rows, number = {}, {}, 0
        try:
            while queue or active:
                for slot in range(workers):
                    if slot in active or not queue:
                        continue
                    ids = queue.pop(0)
                    output = root/'batches'/partition/f'{number:03d}'
                    number += 1
                    output.parent.mkdir(parents=True, exist_ok=True)
                    stream = output.with_suffix('.log').open('w')
                    command = [sys.executable, str(source/'scripts/evaluate_geometry_batch.py'),
                        '--checkpoint', str(checkpoint), '--partition', partition,
                        '--output-dir', str(output), '--episodes', *ids]
                    process = subprocess.Popen(command, env={**environment, 'CUDA_VISIBLE_DEVICES': str(slot)},
                                               stdout=stream, stderr=subprocess.STDOUT)
                    active[slot] = process, stream, ids, output
                for slot, (process, stream, ids, output) in list(active.items()):
                    if process.poll() is None:
                        continue
                    stream.close()
                    if process.returncode:
                        raise RuntimeError(f'Evaluation failed: {output}.log')
                    for episode in ids:
                        folder = output/'episodes'/episode
                        rows[episode] = validate_trace(folder, episode)
                        target = destination/'episodes'/episode
                        target.parent.mkdir(exist_ok=True)
                        shutil.move(str(folder), target)
                    del active[slot]
                ordered = [rows[i] for i in splits[partition] if i in rows]
                if ordered:
                    write_json(destination/'closed_loop.json', {**summarize_rollouts(ordered), 'results': ordered})
                status(f'evaluating_{partition}', episodes=len(rows), total=len(splits[partition]))
                if active:
                    time.sleep(15)
        finally:
            for process, stream, _, _ in active.values():
                if process.poll() is None:
                    process.terminate()
                process.wait()
                stream.close()
        assert set(rows) == set(splits[partition])
        write_json(destination/'complete.json', dict(episodes=len(rows), partition=partition))
        return read_json(destination/'closed_loop.json')

    try:
        write_json(root/'source_sha256.json', source_hashes(source))
        write_json(root/'splits.json', splits)
        official = Path(config['model']['perception']['pretrained_weights'])
        provenance = read_json(official.parent/'provenance.json')
        status('verifying_official_weights')
        assert digest(official) == provenance['sha256']
        recovery = Path(config['train']['recovery_root'])
        manifest = read_json(recovery/'manifest.json')
        assert manifest['collection'] == 'independent joint perturbations only; no policy rollouts'
        assert manifest['source_partition'] == 'train'
        assert set(manifest['episode_ids']) <= set(splits['train'])
        write_json(root/'data_audit.json', dict(benchmark_manifest_sha256=digest(Path(config['benchmark']['root'])/'manifest.json'),
            recovery_manifest_sha256=digest(recovery/'manifest.json'), recovery_episodes=len(manifest['episode_ids']),
            collection=manifest['collection'], training_only=True, model_prediction_cache_used=False))
        protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), initialization='official Pi3 encoder only',
            training_seed=config['train']['seed'], benchmark_seed=config['benchmark']['seed'],
            official_weights=provenance, previous_experiment_checkpoints_accessible=False,
            previous_experiment_checkpoint_used=False, module_warmup_used=False, all_parameters_trainable=True,
            encoder_frozen=False, geometry=config['model']['learned_geometry'], epochs=config['train']['epochs'],
            draws_per_epoch=config['train']['draws_per_epoch'], global_batch_size=config['train']['batch_size'],
            local_batch_size=config['train']['batch_size']//workers, distributed_processes=workers,
            learning_rate=config['train']['learning_rate'], selection='minimum validation waypoint RMSE',
            test_used_for_selection=False, benchmark_already_inspected=True)
        write_json(root/'protocol.json', protocol)
        run([sys.executable, '-m', 'unittest', 'discover', '-s', str(source/'tests'), '-v'], 'tests.log', 'testing')
        write_json(root/'tests_complete.json', dict(passed=True))
        smoke = copy.deepcopy(config)
        smoke['train'].update(epochs=1, draws_per_epoch=config['train']['batch_size']*2)
        write_json(root/'smoke_config.json', smoke)
        run(training_command(root/'smoke_config.json', root/'smoke'), 'smoke.log', 'distributed_smoke')
        gradients = read_json(root/'smoke/gradient_audit.json')
        assert gradients['perception.encoder']['gradient_l1'] > 0
        assert read_json(root/'smoke/initialization.json')['frozen_parameter_names'] == []
        smoke_saved = load_checkpoint(root/'smoke/best.pt')
        first_hashes = read_json(root/'smoke/initial_parameter_sha256.json')
        encoder_changes = [key for key, value in smoke_saved['model'].items() if key.startswith('perception.encoder.')
            and key in first_hashes and hashlib.sha256(value.numpy().tobytes()).hexdigest() != first_hashes[key]]
        assert encoder_changes
        write_json(root/'smoke_audit.json', dict(distributed_forward_backward_passed=True,
            encoder_gradient_l1=gradients['perception.encoder']['gradient_l1'], encoder_parameters_changed=len(encoder_changes),
            reused_to_initialize_full_training=False))
        del smoke_saved
        gc.collect()
        (root/'smoke/best.pt').unlink()
        run(training_command(root/'config.json', root/'training'), 'training.log', 'training')
        assert source_hashes(source) == read_json(root/'source_sha256.json')
        checkpoint = root/'training/best.pt'
        epochs = read_json(root/'training/epochs.json')
        selected = min(epochs, key=lambda row: row['validation_rmse_m'])
        assert len(epochs) == config['train']['epochs']
        saved = load_checkpoint(checkpoint)
        assert saved['selected_epoch'] == selected['epoch'] and saved['splits'] == splits
        initial = read_json(root/'training/initial_parameter_sha256.json')
        changed, unchanged = [], []
        for key, expected in initial.items():
            value = saved['model'][key]
            assert torch.isfinite(value).all()
            (changed if hashlib.sha256(value.numpy().tobytes()).hexdigest() != expected else unchanged).append(key)
        encoder = [key for key in initial if key.startswith('perception.encoder.')]
        with safe_open(official, framework='pt', device='cpu') as weights:
            assert all(initial[key] == hashlib.sha256(weights.get_tensor(key[len('perception.'):]).numpy().tobytes()).hexdigest()
                       for key in encoder)
        groups = ('perception.encoder.', 'perception.decoder.', 'perception.point_head.', 'perception.goal_head.',
                  'perception.action_map_head.', 'perception.joint_head.', 'route.', 'clearance.error_head.',
                  'clearance.trust_head.', 'memory.selector.', 'embodiment.')
        assert all(any(key.startswith(prefix) for key in changed) for prefix in groups)
        initialization = read_json(root/'training/initialization.json')
        assert initialization['trainable_parameters'] == initialization['parameters']
        assert initialization['navigation_initialized_from_scratch'] and initialization['geometry_update'] is None
        audit = dict(official_initial_encoder_exact=True, changed_encoder_parameters=sum(key in changed for key in encoder),
            total_encoder_parameters=len(encoder), changed_parameters=len(changed), unchanged_parameters=unchanged,
            every_model_group_changed=True, all_parameters_trainable=True,
            trainable_parameters=initialization['trainable_parameters'], selected_epoch=selected['epoch'],
            validation_rmse_m=selected['validation_rmse_m'], checkpoint_sha256=digest(checkpoint))
        write_json(root/'model_audit.json', audit)
        del saved
        gc.collect()
        write_json(root/'test_freeze.json', dict(checkpoint=str(checkpoint), sha256=audit['checkpoint_sha256'],
            selected_epoch=selected['epoch'], criterion=protocol['selection'], test_used_for_selection=False))
        validation, test = evaluate(checkpoint, 'validation'), evaluate(checkpoint, 'test')
        summary = dict(checkpoint=str(checkpoint), **audit,
            validation_successes=sum(r['success'] for r in validation['results']),
            test_successes=sum(r['success'] for r in test['results']),
            validation=validation['overall'], test=test['overall'],
            validation_by_route=validation['by_route'], test_by_route=test['by_route'],
            full_partitions=True, test_used_for_selection=False)
        assert digest(checkpoint) == audit['checkpoint_sha256']
        assert source_hashes(source) == read_json(root/'source_sha256.json')
        write_json(root/'summary.json', summary)
        text = f'''# Fresh joint learned-geometry training

Official Pi3 encoder weights are the only pretrained initialization. All {audit['trainable_parameters']:,} parameters, including the encoder, were trainable. No previous experiment checkpoint, module warmup or cached model prediction was used. The existing recovery observations are independent train-only perturbations.

The joint policy uses four attention blocks with residual connections and FFNs in each learned C1/C2 module. C1 performs hard learned point selection; C2 uses its neural risk and learned attention pooling. The full model trained for {len(epochs)} epochs, {config['train']['draws_per_epoch']:,} draws per epoch, global batch {config['train']['batch_size']}, learning rate {config['train']['learning_rate']:g}, on {workers} GPUs. Training seed: {config['train']['seed']}; benchmark split seed: {config['benchmark']['seed']}.

Selected epoch: **{selected['epoch']}** by minimum validation waypoint RMSE (**{selected['validation_rmse_m']*1000:.2f} mm**). Test rollouts did not select the model.

| Partition | Success | Collision | Orientation success |
|---|---:|---:|---:|
| Validation | {summary['validation_successes']}/100 | {validation['overall']['collision_rate']:.0%} | {validation['overall']['xyz_orientation_success_rate']:.0%} |
| Test | {summary['test_successes']}/100 | {test['overall']['collision_rate']:.0%} | {test['overall']['xyz_orientation_success_rate']:.0%} |

Success means collision-free reaching within 10 mm of the XYZ goal. Orientation is reported separately. These are single-seed development results on the previously inspected benchmark.

All model groups changed during training. Initial encoder parameter fingerprints exactly match the official file; {audit['changed_encoder_parameters']}/{audit['total_encoder_parameters']} encoder parameter tensors changed in the selected model. All 200 rollout traces passed finite-value, RGB-only, partition-membership and 15-step execution checks. See `model_audit.json`, `data_audit.json`, `training/gradient_audits`, and the immutable `source/` snapshot.

Checkpoint: `training/best.pt`  
SHA-256: `{audit['checkpoint_sha256']}`
'''
        (root/'RESULTS.md').write_text(text)
        status('complete', **summary)
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
