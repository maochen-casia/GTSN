"""Fresh end-to-end consensus experiment; execute only in the project Docker image."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from tsn.common.checkpoint import save_checkpoint, load_checkpoint
from tsn.common.config import read_json, write_json
from tsn.common.seed import seed_everything, seed_worker
from tsn.data.consensus_dataset import ConsensusDataset
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.recovery_dataset import RecoveryDataset
from tsn.data.splits import make_splits, episode_catalog, ROUTES
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.evaluation.open_loop import device_batch
from tsn.features.state import policy_state
from tsn.models.consensus_policy import ConsensusHead, ConsensusPolicy
from tsn.models.fresh_consensus import FreshConsensus, MODES
from tsn.models.gtsn_policy import geometry_nll
from tsn.training.losses import imitation_loss, predicted_map_loss

SEED = 20261002
CONDITIONS = {
    'state': ['state'], 'state_visual': ['state', 'visual'],
    'state_uncertainty': ['state', 'uncertainty'], 'state_memory': ['state', 'memory'],
    'memory_current_only': ['state', 'memory'], 'memory_no_geometry': ['state', 'memory'],
}


class Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.file = path.open('a', buffering=1)

    def write(self, value):
        self.stream.write(value)
        self.file.write(value)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {str(p.relative_to('/workspace')): sha256(p)
            for folder in ('src', 'scripts', 'configs', 'vendor/Pi3')
            for p in sorted((Path('/workspace')/folder).rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.json')}


def loader(dataset, batch_size, workers, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(SEED), drop_last=False)


def datasets(config, splits, smoke=False):
    root = Path(config['benchmark']['root'])
    catalog = episode_catalog(root)
    # The smoke check deliberately includes a late expert frame and a recovery reset.
    ids = splits['train'][:1] if smoke else splits['train']
    expert = FrameDataset(root, ids, catalog, 30, frame_stride=2, include_rgb=True)
    recovery_root = Path(config['train']['recovery_sources'][0]['path'])
    manifest = read_json(recovery_root/'manifest.json')
    assert manifest['source_partition'] == 'train' and manifest['source_dataset'] == str(root)
    recovery = RecoveryDataset(recovery_root, 30, include_rgb=True)
    assert set(recovery.route_by_episode) <= set(splits['train'])
    assert set(recovery.route_by_episode) == set(manifest['episode_ids'])
    assert len(recovery) == manifest['samples']
    train = ConsensusDataset(expert, recovery)
    val = ConsensusDataset(FrameDataset(root, splits['validation'][:1] if smoke else splits['validation'],
                                       catalog, 30, frame_stride=1, include_rgb=True))
    return train, val


def sampling_weights(dataset):
    expert = dataset.expert
    routes = np.concatenate([np.repeat(ROUTES.index(expert.catalog[ep]), expert.offsets[i+1]-expert.offsets[i])
                             for i, ep in enumerate(expert.ids)])
    source_routes = [routes, np.asarray(dataset.recovery.route_indices)]
    weights = []
    for fraction, labels in zip((.65, .35), source_routes):
        weight = np.zeros(len(labels), dtype=np.float64)
        for route, probability in enumerate((.2, .4, .4)):
            mask = labels == route
            assert mask.any()
            weight[mask] = fraction*probability/mask.sum()
        weights.append(weight)
    return torch.from_numpy(np.concatenate(weights))


def supervision(model, batch):
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        teacher = model.maps(batch['depth'], batch['K'], batch['T_B_C'],
                             batch['goal_pose'][:, :3], batch['future_ee'], batch['valid_future'])
        tcp = model.kinematics(batch['qpos'][:, :7])
        target_q = batch['qpos'][:, None, :7]+batch['target'][:, 4::5]
        waypoint = model.kinematics(target_q)[..., :3, 3]-tcp[:, None, :3, 3]
        depth = F.interpolate(batch['depth'][:, None], (80, 80), mode='nearest-exact')
        valid = torch.isfinite(depth) & (depth >= model.maps.settings.near_m) & (depth <= model.maps.settings.far_m)
        geom_valid = F.adaptive_avg_pool2d(valid.float(), (4, 4)).flatten(1) >= .999
    return teacher, waypoint, geom_valid


def loss_fn(prediction, batch, teacher, waypoint, geom_valid):
    action = imitation_loss(prediction['base'], batch['target'], batch['valid_future'], .02, 10., .25)
    maps, _ = predicted_map_loss(prediction['maps'], teacher)
    routes = sum(F.huber_loss(prediction['waypoints'][m], waypoint, delta=.02) for m in MODES)
    pooled = F.adaptive_avg_pool2d(teacher[:, :3], (4, 4)).flatten(2).transpose(1, 2)
    geometry = sum(geometry_nll(prediction['aux'][m], pooled, geom_valid) for m in ('uncertainty', 'memory'))
    loss = action+.25*maps+routes+.001*geometry
    return loss, {'action': action, 'maps': maps, 'routes': routes, 'geometry_nll': geometry}


@torch.no_grad()
def evaluate(model, parallel, batches):
    model.eval()
    errors = {m: 0. for m in (*MODES, 'state_memory')}
    count = 0
    for raw in batches:
        batch = device_batch(raw, torch.device('cuda'))
        prediction = parallel(batch)
        _, waypoint, _ = supervision(model, batch)
        values = prediction['waypoints']
        values['state_memory'] = (values['state']+values['memory'])/2
        for mode, value in values.items():
            errors[mode] += float((value.float()-waypoint).square().sum())
        count += waypoint.numel()
    return {m: (value/count)**.5 for m, value in errors.items()}


def gradient_audit(model, optimizer, before):
    missing, nonfinite, zero = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
        elif not parameter.grad.count_nonzero():
            zero.append(name)
    groups = {}
    for name, parameter in model.named_parameters():
        group = '.'.join(name.split('.')[:2])
        groups.setdefault(group, {'parameters': 0, 'gradient_l1': 0.})
        groups[group]['parameters'] += parameter.numel()
        if parameter.grad is not None:
            groups[group]['gradient_l1'] += float(parameter.grad.abs().sum())
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert optimizer_ids == {id(p) for p in model.parameters()}
    assert not missing and not nonfinite, (missing, nonfinite)
    assert all(g['gradient_l1'] > 0 for g in groups.values()), groups
    changed = {name: not torch.equal(parameter.detach(), before[name].to(parameter.device))
               for name, parameter in model.named_parameters() if name in before}
    return dict(all_parameters_trainable=True, all_parameters_in_optimizer=True,
                missing_gradients=missing, nonfinite_gradients=nonfinite,
                zero_gradient_tensors_first_step=zero, groups=groups, changed_probe_tensors=changed)


def train(args, config, splits):
    out = args.root/'train'
    out.mkdir(exist_ok=False)
    print('Verifying official weights and creating fresh modules', flush=True)
    weights = Path(config['model']['pi3']['pretrained_weights'])
    provenance = read_json(weights.parent/'provenance.json')
    assert sha256(weights) == provenance['sha256'], 'Official Pi3 weight hash mismatch'
    model = FreshConsensus(config['model']).cuda()
    assert all(p.requires_grad for p in model.parameters())
    initialization = {**model.backbone.initialization, 'official_weight_provenance': provenance,
                      'policy_checkpoint': None, 'head_checkpoints': [], 'feature_cache': None,
                      'all_parameters_trainable': True,
                      'total_parameters': sum(p.numel() for p in model.parameters())}
    write_json(out/'initialization.json', initialization)
    print(json.dumps(initialization), flush=True)
    encoder = list(model.backbone.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    optimizer = torch.optim.AdamW([
        {'params': encoder, 'lr': 1e-5, 'name': 'official_pi3_encoder'},
        {'params': [p for p in model.parameters() if id(p) not in encoder_ids],
         'lr': 3e-4, 'name': 'fresh_decoder_and_heads'}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    parallel = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    print('Loading raw expert and perturbation observations', flush=True)
    train_data, val_data = datasets(config, splits, args.smoke)
    if args.smoke:
        indices = [min(len(train_data.expert)-1, 30), len(train_data.expert)] * max(1, args.batch_size//2)
        train_loader = loader(torch.utils.data.Subset(train_data, indices), args.batch_size, 0)
    else:
        sampler = WeightedRandomSampler(sampling_weights(train_data), len(train_data.expert), True,
                                        generator=torch.Generator().manual_seed(SEED))
        train_loader = loader(train_data, args.batch_size, args.workers, sampler)
    val_loader = loader(val_data, args.batch_size, args.workers)
    write_json(out/'data.json', dict(expert_samples=len(train_data.expert),
               recovery_samples=len(train_data.recovery), validation_samples=len(val_data),
               recovery_manifest_sha256=sha256(train_data.recovery.root/'manifest.json'),
               source_fractions=[.65, .35], route_fractions=[.2, .4, .4],
               history_length=4, history_spacing=15, recovery_history='current observation only'))
    probe_names = ['backbone.encoder.patch_embed.proj.weight', 'backbone.decoder.0.attn.qkv.weight',
                   'backbone.point_head.weight', 'backbone.action_map_head.weight',
                   'heads.state.output.4.weight', 'heads.memory.output.4.weight']
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name in probe_names}
    assert len(before) == len(probe_names)
    best, records = float('inf'), []
    for epoch in range(1, (1 if args.smoke else args.epochs)+1):
        model.train()
        started = time.monotonic()
        running, samples = {}, 0
        order_hash = hashlib.sha256()
        for step, raw in enumerate(train_loader, 1):
            order_hash.update(raw['sample_index'].numpy().tobytes())
            batch = device_batch(raw, torch.device('cuda'))
            optimizer.zero_grad(set_to_none=True)
            prediction = parallel(batch)
            teacher, waypoint, geom_valid = supervision(model, batch)
            loss, terms = loss_fn(prediction, batch, teacher, waypoint, geom_valid)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm)
            if epoch == 1 and step == 1:
                audit = gradient_audit(model, optimizer, before)
            optimizer.step()
            if epoch == 1 and step == 1:
                audit = gradient_audit(model, optimizer, before)
                assert all(audit['changed_probe_tensors'].values()), audit
                write_json(out/'gradient_audit.json', audit)
                print('All-parameter gradient and optimizer-update audit passed', flush=True)
                del before
            size = len(batch['qpos'])
            samples += size
            for key, value in {'total': loss, **terms}.items():
                running[key] = running.get(key, 0.)+float(value.detach())*size
            if step == 1 or step % 25 == 0:
                seconds = time.monotonic()-started
                progress = dict(state='running', stage='training', epoch=epoch, epochs=args.epochs,
                                step=step, steps=len(train_loader), elapsed_epoch_seconds=seconds,
                                loss={k: v/samples for k, v in running.items()})
                write_json(args.root/'status.json', progress)
                print(json.dumps(progress), flush=True)
            if args.smoke:
                model.eval()
                with torch.inference_mode():
                    state = policy_state(batch['qpos'][:1], batch['goal_pose'][:1], model.maps.settings)
                    for condition, modes in CONDITIONS.items():
                        policy = ConsensusPolicy(model.backbone,
                            ConsensusHead([Intervention(model.heads[m], condition) for m in modes]),
                            model.kinematics, servo_radius=.08, execute=15, orientation='baseline').cuda().eval()
                        for observation in range(2):
                            policy.observe_step(observation*15)
                            chunk = policy(batch['rgb'][:1], state, batch['K'][:1], batch['T_B_C'][:1])
                            assert chunk.shape == (1, 30, 7) and torch.isfinite(chunk).all()
                write_json(args.root/'smoke.json', dict(passed=True, batch_size=size,
                           seconds=time.monotonic()-started, loss=float(loss.detach()),
                           inference_conditions=list(CONDITIONS),
                           gpu_memory_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]))
                return
        metrics = evaluate(model, parallel, val_loader)
        improved = metrics['state_memory'] < best
        if improved:
            best = metrics['state_memory']
            selected = epoch
        scheduler.step()
        record = dict(epoch=epoch, rmse_m=metrics, loss={k:v/samples for k,v in running.items()},
                      seconds=time.monotonic()-started, selected_epoch=selected,
                      sampling_sha256=order_hash.hexdigest())
        records.append(record)
        write_json(out/'epochs.json', records)
        payload = dict(format_version=1, epoch=epoch, config=config, splits=splits,
                       model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       initialize_from_checkpoint=None, initialization=initialization, validation=metrics,
                       selected_epoch=selected)
        save_checkpoint(out/'latest.pt', payload)
        if improved:
            # Best checkpoint is standalone and needs no historical experiment.
            save_checkpoint(out/'best.pt', {k:v for k,v in payload.items() if k not in ('optimizer', 'scheduler')})
        write_json(out/'summary.json', dict(completed_epochs=epoch, selected_epoch=selected,
                   best_validation_consensus_rmse_m=best, initialize_from_checkpoint=None))
        print(json.dumps(record), flush=True)


class Intervention(nn.Module):
    def __init__(self, head, condition):
        super().__init__()
        self.head, self.condition = head, condition

    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        if self.condition == 'memory_current_only':
            tokens, geometry, poses = tokens[:, -1:], geometry[:, -1:], poses[:, -1:]
            ages, mask = ages[:, -1:], mask[:, -1:]
        elif self.condition == 'memory_no_geometry':
            geometry = torch.zeros_like(geometry)
        return self.head(tokens, geometry, poses, ages, mask, state, tcp)


def rollout(args):
    selection = read_json(args.root/'selection.json')
    assert selection['conditions'][args.condition] == CONDITIONS[args.condition]
    checkpoint_path = args.root/'train/best.pt'
    assert sha256(checkpoint_path) == selection['checkpoint_sha256']
    checkpoint = load_checkpoint(checkpoint_path)
    model = FreshConsensus(checkpoint['config']['model'], initialize_backbone=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.cuda().eval()
    heads = [Intervention(model.heads[m], args.condition) for m in CONDITIONS[args.condition]]
    policy = ConsensusPolicy(model.backbone, ConsensusHead(heads), model.kinematics,
                             servo_radius=.08, execute=15, orientation='baseline').cuda().eval()
    root = Path(checkpoint['config']['benchmark']['root'])
    output = args.root/'rollouts'/args.partition/args.condition
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', dict(condition=args.condition, partition=args.partition,
               checkpoint_sha256=selection['checkpoint_sha256'], epoch=checkpoint['epoch'],
               modes=CONDITIONS[args.condition], source_sha256=source_hashes()))
    result = evaluate_rollouts(checkpoint['splits'][args.partition], episode_catalog(root), root, output,
                              policy, model.maps, torch.device('cuda'), checkpoint['config']['eval'])
    assert not result['privileged_action_map']
    for ep in checkpoint['splits'][args.partition]:
        with np.load(output/'episodes'/ep/'trajectory.npz') as trajectory:
            assert trajectory['reference_indices'].size == 0
            assert np.isfinite(trajectory['qpos']).all() and np.isfinite(trajectory['predicted_joint_targets']).all()
    write_json(output/'complete.json', dict(episodes=100, passed=True))


def evaluate_all(args):
    # Freeze weights and all rerun conditions before any closed-loop test is read.
    write_json(args.root/'selection.json', dict(
        checkpoint_sha256=sha256(args.root/'train/best.pt'), conditions=CONDITIONS,
        primary='state_memory', criterion='minimum validation state/memory waypoint RMSE',
        test_used_for_selection=False, execute=15, servo=.08, temporal=0., adaptive=False))
    jobs = [(partition, condition) for partition in ('validation', 'test') for condition in CONDITIONS]
    count = torch.cuda.device_count()
    logs = args.root/'logs'
    logs.mkdir(exist_ok=True)
    for offset in range(0, len(jobs), count):
        running = []
        for gpu, (partition, condition) in enumerate(jobs[offset:offset+count]):
            log = (logs/f'{partition}_{condition}.log').open('w')
            command = [sys.executable, __file__, '--root', str(args.root), '--stage', 'rollout',
                       '--partition', partition, '--condition', condition]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running.append((process, log, partition, condition))
        for process, log, partition, condition in running:
            code = process.wait()
            log.close()
            if code:
                for other, _, _, _ in running:
                    if other.poll() is None:
                        other.terminate()
                raise RuntimeError(f'Rollout failed: {partition}/{condition}; see logs')
    results = {partition: {condition: read_json(args.root/'rollouts'/partition/condition/'closed_loop.json')['overall']
                           for condition in CONDITIONS} for partition in ('validation', 'test')}
    write_json(args.root/'results.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', 'rollout'], default='all')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--condition', choices=list(CONDITIONS))
    parser.add_argument('--partition', choices=['validation', 'test'])
    args = parser.parse_args()
    if args.stage == 'all':
        sys.stdout = Tee(sys.stdout, args.root/'run.log')
        sys.stderr = Tee(sys.stderr, args.root/'run.log')
    torch.set_num_threads(1)
    seed_everything(SEED)
    if args.stage == 'rollout':
        rollout(args)
        return
    try:
        assert args.epochs > 0 and args.batch_size > 0
        config = {key: read_json(f'/workspace/configs/{key}/{filename}.json') for key, filename in
                  [('benchmark', 'tsn-1k'), ('model', 'pi3_small'), ('train', 'pi3_small'), ('eval', 'pi3_small')]}
        config['train'].update(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.workers,
                               seed=SEED, run_name=args.root.name, initialize_from_checkpoint=None,
                               checkpoint_selection={'method': 'consensus_validation_waypoint_rmse'})
        splits = make_splits(config['benchmark'])
        write_json(args.root/'splits.json', splits)
        write_json(args.root/'protocol.json', dict(config=config, seed=SEED, conditions=CONDITIONS,
                   primary='state_memory', modes=MODES, all_modules_jointly_trainable=True,
                   historical_checkpoints_loaded=[], historical_feature_caches_used=[],
                   loss='joint imitation + 0.25 dense maps + sum of four waypoint Huber losses + 0.001 sum of two geometry NLLs',
                   source_sha256=source_hashes(), docker_image_id=os.environ.get('GTSN_DOCKER_IMAGE_ID'),
                   smoke=args.smoke))
        train(args, config, splits)
        if args.smoke:
            write_json(args.root/'status.json', dict(state='complete', stage='smoke'))
            return
        gc.collect()
        torch.cuda.empty_cache()
        write_json(args.root/'status.json', dict(state='running', stage='closed_loop_evaluation'))
        evaluate_all(args)
        assert read_json(args.root/'protocol.json')['source_sha256'] == source_hashes()
        write_json(args.root/'audit.json', dict(passed=True, all_parameters_trainable=True,
                   previous_checkpoint_initialization=False, epochs=args.epochs,
                   validation_episodes_per_condition=100, test_episodes_per_condition=100,
                   conditions=list(CONDITIONS), source_snapshot_unchanged=True))
        write_json(args.root/'status.json', dict(state='complete', results=str(args.root/'results.json')))
    except Exception:
        write_json(args.root/'status.json', dict(state='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
