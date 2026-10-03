"""Frozen-perception GTSN pilot: cache, matched training, and closed-loop evaluation.

Run inside the project Docker image using scripts/run_gtsn_docker.py.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import read_json, write_json
from tsn.common.seed import seed_everything
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.loaders import make_loader
from tsn.data.recovery_dataset import RecoveryDataset
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.evaluation.open_loop import device_batch
from tsn.features.state import policy_state
from tsn.models.factory import make_maps, make_policy
from tsn.models.gtsn_policy import GTSNHead, GTSNPolicy, MODES, geometry_nll
from tsn.training.losses import imitation_loss

CHECKPOINT = Path('/run/user/1016/experiments/pi3_small_perturbation10_20261002/train/best.pt')
SEED = 20261002


def source_hashes():
    paths = list(Path('/workspace/src/tsn').rglob('*.py')) + [Path(__file__)]
    return {str(p.relative_to('/workspace')): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def backbone():
    checkpoint = load_checkpoint(CHECKPOINT)
    cfg = checkpoint['config']
    model = make_policy(cfg['model'], initialize_backbone=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.requires_grad_(False).cuda().eval()
    return model, cfg, checkpoint['splits']


@torch.inference_mode()
def cache(root):
    root.mkdir(parents=True, exist_ok=True)
    model, cfg, splits = backbone()
    data_root = Path(cfg['benchmark']['root'])
    catalog = episode_catalog(data_root)
    validate_splits(splits, catalog)
    write_json(root / 'protocol.json', {'config': cfg, 'splits': splits, 'seed': SEED,
        'source_sha256': source_hashes(), 'backbone_checkpoint': str(CHECKPOINT),
        'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'docker_image_id': os.environ.get('GTSN_DOCKER_IMAGE_ID'),
        'modes': list(MODES), 'epochs': 10, 'history_length': 4, 'history_spacing_control_steps': 15,
        'test_selection': 'validation success, then collision, then action RMSE',
        'state_correction': 'state-only residual on common frozen RGB/map baseline; not a state-only policy'})
    maps = make_maps(cfg['model']).cuda()
    parallel = torch.nn.DataParallel(model, device_ids=list(range(min(4, torch.cuda.device_count()))))
    for partition in ('train', 'validation', 'test'):
        directory = root / 'cache' / partition
        if (directory / 'metadata.json').exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        ids = splits[partition]
        stride = 2 if partition == 'train' else 1
        expert = FrameDataset(data_root, ids, catalog, 30, frame_stride=stride, include_rgb=True)
        recovery = None
        if partition == 'train':
            recovery = RecoveryDataset(Path(cfg['train']['recovery_sources'][0]['path']), 30, include_rgb=True)
            assert set(recovery.route_by_episode) <= set(ids)
        dataset = ConcatDataset([expert, recovery]) if recovery is not None else expert
        n = len(dataset)
        specs = {'tokens': ((16, 768), np.float16), 'geometry': ((16, 6), np.float32),
                 'teacher': ((16, 3), np.float32), 'geometry_valid': ((16,), bool),
                 'baseline': ((30, 7), np.float32), 'state': ((16,), np.float32),
                 'pose': ((4, 4), np.float32), 'target': ((30, 7), np.float32),
                 'valid': ((30,), bool), 'route': ((), np.int64), 'episode': ((), np.int64),
                 'frame': ((), np.int64), 'source': ((), np.int64)}
        arrays = {k: np.lib.format.open_memmap(directory / f'{k}.npy', mode='w+', dtype=d, shape=(n, *s))
                  for k, (s, d) in specs.items()}
        lookup = {episode: i for i, episode in enumerate(ids)}
        loader = make_loader(dataset, {'device': 'cuda', 'batch_size': 128, 'num_workers': 8}, False, SEED)
        offset, started = 0, time.monotonic()
        for raw in loader:
            batch = device_batch(raw, torch.device('cuda'))
            state = policy_state(batch['qpos'], batch['goal_pose'], maps.settings)
            base, tokens, geometry = parallel(batch['rgb'], state, batch['K'], batch['T_B_C'], return_features=True)
            teacher = maps(batch['depth'], batch['K'], batch['T_B_C'], batch['goal_pose'][:, :3],
                           batch['future_ee'], batch['valid_future'])
            # Supervise only fully observed pooled cells. Zero coordinates cannot
            # encode unknown pixels as free space or as an exact point target.
            depth = F.interpolate(batch['depth'][:, None], (80, 80), mode='nearest-exact')
            valid_depth = torch.isfinite(depth) & (depth >= maps.settings.near_m) & (depth <= maps.settings.far_m)
            geom_valid = F.adaptive_avg_pool2d(valid_depth.float(), (4, 4)).flatten(1) >= .999
            values = {'tokens': tokens, 'geometry': geometry, 'baseline': base, 'state': state,
                      'pose': batch['T_B_C'], 'target': batch['target'], 'valid': batch['valid_future'],
                      'route': batch['route'], 'frame': batch['frame_index'], 'geometry_valid': geom_valid,
                      'teacher': F.adaptive_avg_pool2d(teacher[:, :3], (4, 4)).flatten(2).transpose(1, 2)}
            size = len(state)
            for key, value in values.items():
                arrays[key][offset:offset+size] = value.float().cpu().numpy() if value.is_floating_point() else value.cpu().numpy()
            arrays['episode'][offset:offset+size] = [lookup[x.removesuffix('_recovery')] for x in raw['episode_id']]
            arrays['source'][offset:offset+size] = np.arange(offset, offset+size) >= len(expert)
            offset += size
            if offset % 2560 == 0 or offset == n:
                print(f'cache {partition} {offset}/{n}, {time.monotonic()-started:.1f}s', flush=True)
        for value in arrays.values():
            value.flush()
        history, ages, mask = build_history(arrays['episode'], arrays['frame'], arrays['source'])
        for key, value in [('history', history), ('ages', ages), ('mask', mask)]:
            np.save(directory / f'{key}.npy', value)
        write_json(directory / 'metadata.json', {'samples': n, 'expert_samples': len(expert),
            'recovery_samples': n-len(expert), 'ids': ids, 'stride': stride,
            'elapsed_seconds': time.monotonic()-started, 'history_audit_passed': True})
        expert.close()
        if recovery is not None:
            recovery.close()


def build_history(episode, frame, source, length=4):
    """Map each anchor to real earlier observations, without crossing reset boundaries."""
    n = len(episode)
    history = np.repeat(np.arange(n)[:, None], length, axis=1)
    mask = np.zeros((n, length), dtype=bool)
    ages = np.zeros((n, length), dtype=np.float32)
    mask[:, -1] = True
    for ep in np.unique(episode):
        indices = np.flatnonzero((episode == ep) & (source == 0))
        frames = frame[indices]
        if not np.all(np.diff(frames) > 0):
            raise ValueError('Expert frames must be strictly ordered within episode')
        for slot in range(length-1):
            targets = frames - 15 * (length-1-slot)
            previous = np.searchsorted(frames, targets, side='right') - 1
            valid = previous >= 0
            anchors, old = indices[valid], indices[previous[valid]]
            history[anchors, slot] = old
            ages[anchors, slot] = frame[anchors] - frame[old]
            mask[anchors, slot] = True
    assert np.all(episode[history] == episode[:, None])
    assert np.all(frame[history] <= frame[:, None])
    assert np.all(mask[source == 1].sum(1) == 1)
    return history, ages, mask


def load_data(directory):
    return {p.stem: torch.from_numpy(np.array(np.load(p, mmap_mode='r'), copy=True)).cuda()
            for p in directory.glob('*.npy')}


def inputs(data, index):
    history = data['history'][index]
    return (data['tokens'][history], data['geometry'][history], data['pose'][history],
            data['ages'][index], data['mask'][index], data['state'][index], data['baseline'][index])


@torch.inference_mode()
def evaluate_head(model, data):
    model.eval()
    errors, risks, geometry_errors, variances, valids = [], [], [], [], []
    for index in torch.arange(len(data['state']), device='cuda').split(256):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction, aux = model(*inputs(data, index))
        errors.append((prediction.float()-data['target'][index]).square().mean((1, 2)).cpu().numpy())
        risks.append(aux['risk'].cpu().numpy())
        geometry_errors.append((aux['mean'].float()-data['teacher'][index]).square().cpu().numpy())
        variances.append(aux['logvar'].float().exp().cpu().numpy())
        valids.append(data['geometry_valid'][index].cpu().numpy())
    errors, risk = np.concatenate(errors), np.concatenate(risks)
    error, variance, valid = np.concatenate(geometry_errors), np.concatenate(variances), np.concatenate(valids)
    error, variance = error[valid], variance[valid]
    metrics = {'rmse_rad': float(np.sqrt(errors.mean())),
               'point_mean_rmse_normalized': float(np.sqrt(error.mean())),
               'predicted_variance_mean': float(variance.mean()),
               'squared_error_over_variance': float((error/variance).mean()),
               'marginal_95_coverage': float((error <= 1.96**2*variance).mean()),
               'geometry_valid_tokens': int(valid.sum()),
               'risk_mean': float(risk.mean())}
    return metrics, errors, risk


def train(root, mode, epochs=10):
    output = root / 'heads' / mode
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    model = GTSNHead(mode).cuda()
    train_data, val = load_data(root / 'cache/train'), load_data(root / 'cache/validation')
    metadata = read_json(root / 'cache/train/metadata.json')
    routes, sources = train_data['route'].cpu().numpy(), train_data['source'].cpu().numpy()
    weights = np.zeros(len(routes), dtype=np.float64)
    for s, fraction in enumerate((.65, .35)):
        for r, probability in enumerate((.2, .4, .4)):
            mask = (sources == s) & (routes == r)
            assert mask.any()
            weights[mask] = fraction * probability / mask.sum()
    sampler = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    initial, _, _ = evaluate_head(model, val)
    best, best_epoch = initial['rmse_rad'], 0
    torch.save(model.state_dict(), output / 'best.pt')
    records = []
    for epoch in range(1, epochs+1):
        started = time.monotonic()
        model.train()
        order = torch.multinomial(torch.from_numpy(weights), metadata['expert_samples'], replacement=True, generator=sampler)
        order_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
        losses = []
        for index in order.cuda().split(128):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, aux = model(*inputs(train_data, index))
                action_loss = imitation_loss(prediction, train_data['target'][index], train_data['valid'][index], .02, 10., .25)
                geom_loss = geometry_nll(aux, train_data['teacher'][index], train_data['geometry_valid'][index])
                loss = action_loss + (.002 * geom_loss if mode in ('uncertainty', 'memory') else 0)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append([float(action_loss.detach()), float(geom_loss.detach())])
        metrics, _, _ = evaluate_head(model, val)
        if metrics['rmse_rad'] < best:
            best, best_epoch = metrics['rmse_rad'], epoch
            torch.save(model.state_dict(), output / 'best.pt')
        scheduler.step()
        records.append({'epoch': epoch, 'validation': metrics, 'loss': np.mean(losses, axis=0).tolist(),
                        'sampling_sha256': order_hash, 'elapsed_seconds': time.monotonic()-started})
        write_json(output / 'epochs.json', records)
        print(f'{mode} {epoch}/{epochs}: RMSE={metrics["rmse_rad"]:.6f}, best={best:.6f}, {records[-1]["elapsed_seconds"]:.1f}s', flush=True)
    torch.save(model.state_dict(), output / 'latest.pt')
    model.load_state_dict(torch.load(output / 'best.pt', weights_only=True))
    metrics, errors, _ = evaluate_head(model, val)
    # Calibrate the adaptive observation threshold using training observations only.
    _, _, train_risks = evaluate_head(model, train_data)
    threshold = float(np.quantile(train_risks, .65))
    np.save(output / 'validation_frame_mse.npy', errors)
    write_json(output / 'results.json', {'mode': mode, 'completed_epochs': epochs, 'selected_epoch': best_epoch,
        'initial_validation': initial, 'validation': metrics, 'adaptive_threshold': threshold,
        'adaptive_threshold_source': '65th percentile of train-only attended geometry RMS uncertainty',
        'trainable_parameters': sum(p.numel() for p in model.parameters()), 'seed': SEED})


def rollout_condition(root, mode, schedule, partition, limit=0):
    output = root / 'rollouts' / partition / f'{mode}_{schedule}'
    if (output / 'complete.json').exists():
        return
    model, cfg, splits = backbone()
    head = GTSNHead(mode).cuda()
    head.load_state_dict(torch.load(root / 'heads' / mode / 'best.pt', weights_only=True))
    threshold = read_json(root / 'heads' / mode / 'results.json')['adaptive_threshold']
    policy = GTSNPolicy(model, head, schedule, threshold).cuda().eval()
    options = read_json(Path('/workspace/configs/eval/pi3_small.json'))
    if schedule == 'fixed5':
        options['execute_horizon'] = 5
    catalog = episode_catalog(Path(cfg['benchmark']['root']))
    ids = splits[partition][:limit] if limit else splits[partition]
    result = evaluate_rollouts(ids, catalog, Path(cfg['benchmark']['root']), output, policy,
                              make_maps(cfg['model']).cuda(), torch.device('cuda'), options)
    assert not result['privileged_action_map']
    for ep in ids:
        trajectory = np.load(output / 'episodes' / ep / 'trajectory.npz')
        assert len(trajectory['reference_indices']) == 0
        assert np.isfinite(trajectory['qpos']).all()
        assert np.isfinite(trajectory['predicted_joint_targets']).all()
    write_json(output / 'complete.json', {'episodes': len(ids), 'partition': partition, 'mode': mode,
                                         'schedule': schedule, 'no_privileged_inputs': True})
    if partition == 'test':
        metrics, errors, _ = evaluate_head(head, load_data(root / 'cache/test'))
        write_json(output / 'open_loop.json', metrics)
        np.save(output / 'frame_mse.npy', errors)


def parallel_jobs(root, jobs):
    """Independent numerical experiments on project GPUs 0--3, not agent delegation."""
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    for start in range(0, len(jobs), 4):
        running = []
        for gpu, (name, extra) in enumerate(jobs[start:start+4]):
            log = (logs / f'{name}.log').open('w')
            process = subprocess.Popen([sys.executable, __file__, '--root', str(root), *extra],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}, stdout=log, stderr=subprocess.STDOUT)
            running.append((name, process, log))
        while any(p.poll() is None for _, p, _ in running):
            print('jobs: ' + ', '.join(f'{name}={p.poll()}' for name, p, _ in running), flush=True)
            time.sleep(20)
        for name, p, log in running:
            log.close()
            if p.returncode:
                raise RuntimeError(f'{name} exited {p.returncode}; see logs')


def summarize(root):
    protocol = read_json(root / 'protocol.json')
    table = {}
    for path in sorted((root / 'rollouts/validation').glob('*/complete.json')):
        name = path.parent.name
        result = read_json(path.parent / 'closed_loop.json')
        table[name] = {**result['overall'], 'by_route': result['by_route'],
                       'mean_replans': float(np.mean([x['replans'] for x in result['results']])),
                       'mean_inference_ms': float(np.mean([x['mean_inference_ms'] for x in result['results']]))}
    assert len(table) == 6 and all(v['episodes'] == 100 for v in table.values())
    def ranking(name):
        mode = name.rsplit('_', 1)[0]
        rmse = read_json(root / 'heads' / mode / 'results.json')['validation']['rmse_rad']
        return (-table[name]['success_rate'], table[name]['collision_rate'], rmse, name)
    selected = min(table, key=ranking)
    write_json(root / 'selection.json', {'selected': selected, 'validation': table,
        'criterion': 'success descending, collision ascending, validation RMSE ascending, name',
        'test_metrics_seen_for_selection': False})
    histories = [read_json(root / 'heads' / mode / 'epochs.json') for mode in MODES]
    assert all(len(h) == 10 for h in histories)
    assert all([r['sampling_sha256'] for r in h] == [r['sampling_sha256'] for r in histories[0]] for h in histories)
    assert protocol['source_sha256'] == source_hashes(), 'Source changed during experiment'
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('stage', choices=['cache', 'train', 'rollout', 'all', 'summarize'])
    parser.add_argument('--mode', choices=MODES, default='memory')
    parser.add_argument('--schedule', choices=['fixed15', 'fixed5', 'adaptive'], default='fixed15')
    parser.add_argument('--partition', choices=['validation', 'test'], default='validation')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_everything(SEED)
    if args.stage == 'cache':
        cache(args.root)
    elif args.stage == 'train':
        train(args.root, args.mode)
    elif args.stage == 'rollout':
        rollout_condition(args.root, args.mode, args.schedule, args.partition, args.limit)
    elif args.stage == 'summarize':
        summarize(args.root)
    else:
        args.root.mkdir(parents=True, exist_ok=True)
        write_json(args.root / 'status.json', {'state': 'running', 'stage': 'cache'})
        cache(args.root)
        # Release frozen perception and cache staging CUDA memory before workers.
        torch.cuda.empty_cache()
        write_json(args.root / 'status.json', {'state': 'running', 'stage': 'training'})
        parallel_jobs(args.root, [(m, ['train', '--mode', m]) for m in MODES])
        write_json(args.root / 'status.json', {'state': 'running', 'stage': 'validation_rollouts'})
        conditions = [(m, 'fixed15') for m in MODES] + [('memory', s) for s in ('adaptive', 'fixed5')]
        parallel_jobs(args.root, [(m+'_'+s, ['rollout', '--mode', m, '--schedule', s]) for m, s in conditions])
        selected = summarize(args.root)
        mode, schedule = selected.rsplit('_', 1)
        write_json(args.root / 'status.json', {'state': 'running', 'stage': 'test', 'selected': selected})
        rollout_condition(args.root, mode, schedule, 'test')
        write_json(args.root / 'audit.json', {'passed': True, 'matched_sampling': True, 'epochs_per_condition': 10,
            'conditions': list(MODES), 'validation_rollout_conditions': 6, 'validation_episodes_each': 100,
            'test_episodes': 100, 'selected': selected, 'source_hashes_match': True,
            'train_validation_test_disjoint': True, 'causal_history': True, 'no_privileged_policy_inputs': True})
        write_json(args.root / 'status.json', {'state': 'complete', 'selected': selected})


if __name__ == '__main__':
    main()
