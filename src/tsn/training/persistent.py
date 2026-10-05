"""Frozen-feature persistent-memory experiment; existing artifacts stay read-only."""
import copy
import json
import hashlib
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import create_output, read_json, write_json
from tsn.common.seed import seed_everything
from tsn.models.compact_policy import CompactRouteHead, compact_state, load_route_components
from tsn.models.persistent_policy import PersistentSceneHead, compact_sequence, metric_points


def observation_sequences(arrays, period=15):
    """Full expert prefixes at deployment cadence; recovery samples are isolated.

    Select the latest cached frame at or before each target time, handling the
    original stride-two cache without inventing unavailable frames.
    """
    result = []
    for episode in np.unique(arrays['episode'][arrays['source'] == 0]):
        indices = np.flatnonzero((arrays['episode'] == episode) & (arrays['source'] == 0))
        frames = arrays['frame'][indices]
        if not np.all(np.diff(frames) > 0):
            raise ValueError('Expert cache must be ordered within each episode')
        targets = np.arange(frames[0], frames[-1]+1, period)
        selected = indices[np.unique(np.searchsorted(frames, targets, side='right')-1)]
        result.append(selected)
    result.extend(np.array([i]) for i in np.flatnonzero(arrays['source'] != 0))
    return result


def prepare_cache(checkpoint_path, original_cache, output, device, batch_size=64):
    """Reuse pooled features; add 20x20 predicted/teacher points only once."""
    create_output(output)
    backbone, _, _, maps, checkpoint = load_route_components(checkpoint_path, device)
    backbone.requires_grad_(False).eval()
    dataset_root = Path(checkpoint['config']['benchmark']['root'])
    recovery_path = Path(checkpoint['config']['train']['recovery_sources'][0]['path'])
    write_json(output/'protocol.json', dict(checkpoint=str(checkpoint_path), original_cache=str(original_cache),
               frozen_backbone=True, test_used=False, observation_period=15,
               point_source='RGB predicted pointmaps; depth is supervision only', recovery_history='isolated'))
    for partition in ('train', 'validation'):
        source = original_cache / partition
        arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in source.glob('*.npy')}
        metadata = read_json(source/'metadata.json')
        if metadata['episode_ids'] != checkpoint['splits'][partition]:
            raise ValueError('Cache and checkpoint episode splits differ')
        sequences = observation_sequences(arrays)
        selected = np.unique(np.concatenate(sequences))
        directory = output/partition
        directory.mkdir()
        remap = np.full(len(arrays['state']), -1, dtype=np.int64)
        remap[selected] = np.arange(len(selected))
        for name in ('tokens', 'geometry', 'teacher', 'geometry_valid', 'state', 'pose',
                     'waypoint', 'tcp', 'route', 'episode', 'frame', 'source'):
            np.save(directory/f'{name}.npy', arrays[name][selected])
        offsets = np.cumsum([0] + [len(s) for s in sequences])
        np.save(directory/'sequence_offsets.npy', offsets)
        np.save(directory/'sequence_indices.npy', remap[np.concatenate(sequences)])
        np.save(directory/'original_indices.npy', selected)
        count = len(selected)
        specs = dict(points=((400, 3), np.float32), point_teacher=((400, 3), np.float32),
                     point_valid=((400,), bool))
        extra = {name: np.lib.format.open_memmap(directory/f'{name}.npy', mode='w+', dtype=dtype,
                 shape=(count, *shape)) for name, (shape, dtype) in specs.items()}
        # Recovery archives can contain independently reset observations with
        # duplicate timestamps. Match the original cache's deterministic order.
        recovery_rows = []
        for archive in sorted(recovery_path.glob('episode_*.npz')):
            with np.load(archive, allow_pickle=False) as data:
                ep = str(data['episode_id'])
                if ep in metadata['episode_ids']:
                    recovery_rows.extend((archive, i) for i in range(len(data['frame_index'])))
        expert_count = int(metadata['expert_samples'])
        started = time.monotonic()
        loaded_archive, recovery = None, None
        for start in range(0, count, batch_size):
            chosen = selected[start:start+batch_size]
            images, depths, intrinsics = [], [], []
            for original in chosen:
                ep = metadata['episode_ids'][int(arrays['episode'][original])]
                frame = int(arrays['frame'][original])
                if arrays['source'][original] == 0:
                    with h5py.File(dataset_root/ep/'episode.h5', 'r') as handle:
                        images.append(handle['rgb'][frame])
                        depths.append(handle['depth_m'][frame])
                        intrinsics.append(handle['intrinsics'][:])
                else:
                    archive, row = recovery_rows[int(original)-expert_count]
                    if archive != loaded_archive:
                        if recovery is not None:
                            recovery.close()
                        recovery = np.load(archive, allow_pickle=False)
                        loaded_archive = archive
                    if str(recovery['episode_id']) != ep or int(recovery['frame_index'][row]) != frame:
                        raise ValueError('Recovery cache/archive alignment mismatch')
                    images.append(recovery['rgb'][row])
                    depths.append(recovery['depth'][row])
                    intrinsics.append(recovery['K'][row])
            tensor = lambda name: torch.from_numpy(np.array(arrays[name][chosen], copy=True)).to(device)
            with torch.inference_mode():
                rgb = torch.from_numpy(np.stack(images)).to(device)
                K = torch.from_numpy(np.stack(intrinsics)).float().to(device)
                _, dense = backbone(rgb, tensor('state'), K, tensor('pose'), return_maps=True)
                predicted = metric_points(dense)
                depth = torch.from_numpy(np.stack(depths)).float().to(device)
                # Teacher backprojection uses the same map/pixel-center convention
                # as GeometryMaps; future/goal scores never enter memory inputs.
                teacher = maps(depth, K, tensor('pose'), torch.zeros(len(chosen), 3, device=device),
                               torch.zeros(len(chosen), 1, 3, device=device),
                               torch.zeros(len(chosen), 1, dtype=torch.bool, device=device))
                teacher = metric_points(teacher)
                depth_small = F.interpolate(depth[:, None], (maps.settings.height, maps.settings.width), mode='nearest-exact')
                mask = torch.isfinite(depth_small) & (depth_small >= maps.settings.near_m) & (depth_small <= maps.settings.far_m)
                mask = F.interpolate(mask.float(), (20, 20), mode='nearest-exact').flatten(1).bool()
            for name, value in (('points', predicted), ('point_teacher', teacher), ('point_valid', mask)):
                extra[name][start:start+len(chosen)] = value.cpu().numpy()
            if start//batch_size % 10 == 0 or start+batch_size >= count:
                progress = dict(partition=partition, completed=min(start+batch_size, count), total=count,
                                elapsed_seconds=time.monotonic()-started)
                write_json(output/'status.json', progress)
                print('cache', json.dumps(progress), flush=True)
        if recovery is not None:
            recovery.close()
        for value in extra.values():
            value.flush()
        write_json(directory/'metadata.json', dict(samples=count, expert_samples=int((arrays['source'][selected] == 0).sum()),
            episode_ids=metadata['episode_ids'], sequences=len(sequences), observation_period=15,
            original_cache=str(source), frozen_checkpoint=str(checkpoint_path)))
    write_json(output/'complete.json', dict(frozen_backbone=True, partitions=['train', 'validation']))


class SequenceCache:
    """Small cadence-sampled cache resident on the training device."""
    def __init__(self, directory, device):
        self.device = device
        arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in directory.glob('*.npy')}
        offsets, indices = arrays.pop('sequence_offsets'), arrays.pop('sequence_indices')
        self.sequences = [np.array(indices[offsets[i]:offsets[i+1]]) for i in range(len(offsets)-1)]
        self.arrays = {k: torch.from_numpy(np.array(v, copy=True)).to(device) for k, v in arrays.items()}
        self.metadata = read_json(directory/'metadata.json')

    def batch(self, selected):
        sequences = [self.sequences[i] for i in selected]
        length = max(map(len, sequences))
        index = np.stack([np.pad(s, (0, length-len(s)), mode='edge') for s in sequences])
        valid = torch.arange(length, device=self.device)[None] < torch.tensor([len(s) for s in sequences], device=self.device)[:, None]
        index = torch.from_numpy(index).to(self.device)
        values = {k: self.arrays[k][index] for k in ('tokens', 'geometry', 'pose', 'frame', 'state', 'tcp',
                  'waypoint', 'teacher', 'geometry_valid', 'points', 'point_teacher', 'point_valid')}
        values['valid'] = valid
        return values

    def sampling(self):
        lengths = np.array([len(s) for s in self.sequences])
        first = torch.tensor([s[0] for s in self.sequences], device=self.device)
        routes = self.arrays['route'][first].cpu().numpy()
        sources = self.arrays['source'][first].cpu().numpy()
        weights = np.zeros(len(lengths))
        for source, fraction in ((0, .65), (1, .35)):
            for route, probability in enumerate((.2, .4, .4)):
                selected = (sources == source) & (routes == route)
                if not selected.any():
                    raise ValueError('Training needs all source/route groups')
                weights[selected] = lengths[selected] * fraction * probability / lengths[selected].sum()
        return torch.from_numpy(weights)


def predict(head, variant, batch):
    args = [batch[k] for k in ('tokens', 'geometry', 'pose', 'frame', 'valid', 'state', 'tcp')]
    if isinstance(head, PersistentSceneHead):
        prediction, aux, _ = head.sequence(*args, points=batch['points'])
        return prediction, aux
    return compact_sequence(head, *args, current_only=variant == 'current')


def sequence_mean(value, mask):
    """Mean of valid elements per sequence, avoiding length-dependent sampling."""
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    mask = mask.expand_as(value).float()
    return (value.float()*mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)


def memory_teacher(head, batch):
    """Causal geometry targets at the writer's fixed addresses; loss only."""
    points, teacher = batch['points'].float(), batch['point_teacher'].float()
    mask = batch['point_valid'] & batch['valid'][..., None]
    mask = mask & (points[..., 0] > .10) & (points[..., 0] < 1.05) & (points[..., 1].abs() < .60)
    mask = mask & (points[..., 2] > .04) & (points[..., 2] < .65)
    mask = mask & ((points-batch['tcp'][..., None, :3, 3]).norm(dim=-1) > .07)
    mask = mask & ((points-batch['pose'][..., None, :3, 3]).norm(dim=-1) > .04)
    address = (points[..., None, :]-head.anchors).square().sum(-1).argmin(-1)
    values = torch.cat((teacher, torch.ones_like(teacher[..., :1])), -1) * mask[..., None]
    write = values.new_zeros(*values.shape[:2], len(head.anchors), 4)
    write = write.scatter_add(2, address[..., None].expand_as(values), values).cumsum(1)
    return write[..., :3]/write[..., 3:].clamp_min(1), write[..., 3] > 0


def training_loss(head, prediction, auxiliary, batch):
    valid = batch['valid']
    action = sequence_mean(F.huber_loss(prediction, batch['waypoint'], delta=.02, reduction='none'), valid)
    nll = .5 * ((auxiliary['mean']-batch['teacher']).square() * (-auxiliary['logvar']).exp() + auxiliary['logvar'])
    geometry = sequence_mean(nll, batch['geometry_valid'] & valid[..., None])
    loss = action + .001*geometry
    if isinstance(head, PersistentSceneHead) and head.use_points:
        mask = batch['point_valid'] & valid[..., None]
        point_error = F.huber_loss(auxiliary['point_mean'], batch['point_teacher'], delta=.02, reduction='none')
        point_loss = sequence_mean(point_error, mask)
        confidence_target = (mask & ((batch['points']-batch['point_teacher']).norm(dim=-1) < .05)).float()
        calibration = sequence_mean(F.binary_cross_entropy_with_logits(auxiliary['point_logits'], confidence_target, reduction='none'), valid)
        target, target_valid = memory_teacher(head, batch)
        accumulated = sequence_mean(F.huber_loss(auxiliary['memory_points'], target, delta=.02, reduction='none'),
                                    target_valid & auxiliary['memory_valid'] & valid[..., None])
        loss = loss + .05*point_loss + .05*accumulated + .00001*calibration
    return loss.mean()


@torch.inference_mode()
def score(head, variant, data, batch_size):
    head.eval()
    squared, count, late_squared, late_count = 0., 0, 0., 0
    episode_squared = []
    for start in range(0, len(data.sequences), batch_size):
        selected = list(range(start, min(start+batch_size, len(data.sequences))))
        batch = data.batch(selected)
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=data.device.type == 'cuda'):
            prediction, auxiliary = predict(head, variant, batch)
        error = (prediction-batch['waypoint']).square().mean((-1, -2))
        valid = batch['valid']
        squared += float(error[valid].sum())
        count += int(valid.sum())
        late = valid & (torch.arange(valid.shape[1], device=data.device)[None] >= 4)
        late_squared += float(error[late].sum())
        late_count += int(late.sum())
        episode_squared.extend(sequence_mean(error, valid).cpu().tolist())
    return dict(rmse_m=(squared/count)**.5, after_four_rmse_m=(late_squared/max(late_count, 1))**.5,
                observations=count, after_four_observations=late_count,
                episode_mse=episode_squared)


@torch.inference_mode()
def benchmark(head, variant, data, batch_size=16, repeats=20):
    """Forward-only head throughput; excludes Pi3, cache and transfer costs."""
    candidates = sorted(range(len(data.sequences)), key=lambda i: len(data.sequences[i]), reverse=True)[:batch_size]
    batch = data.batch(candidates)
    for _ in range(3):
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=data.device.type == 'cuda'):
            predict(head, variant, batch)
    if data.device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(repeats):
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=data.device.type == 'cuda'):
            predict(head, variant, batch)
    if data.device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = (time.perf_counter()-started)/repeats
    return dict(valid_frames=int(batch['valid'].sum()), padded_frames=batch['valid'].numel(),
                forward_seconds=seconds, valid_frames_per_second=float(batch['valid'].sum())/seconds,
                gpu_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if data.device.type == 'cuda' else None)


def train_study(checkpoint_path, cache, output, device, epochs=30, seeds=(20261005,),
                variants=('current', 'four', 'scene', 'scene_points'), batch_size=32, draws=2048, resume=False):
    if not resume:
        create_output(output)
    else:
        protocol = read_json(output/'protocol.json')
        expected = dict(checkpoint=str(checkpoint_path), cache=str(cache), epochs=epochs, seeds=list(seeds),
                        variants=list(variants), batch_size=batch_size, sequence_draws_per_epoch=draws)
        if any(protocol[k] != v for k, v in expected.items()):
            raise ValueError('Resume requires an identical experiment protocol')
    checkpoint = load_checkpoint(checkpoint_path)
    _, compact_weights = compact_state(checkpoint)
    training, validation = SequenceCache(cache/'train', device), SequenceCache(cache/'validation', device)
    probabilities = training.sampling()
    write_json(output/'protocol.json', dict(checkpoint=str(checkpoint_path), cache=str(cache), epochs=epochs,
        seeds=list(seeds), variants=list(variants), batch_size=batch_size, sequence_draws_per_epoch=draws,
        pretrained_lr=3e-5, new_parameter_lr=3e-4, frozen_backbone=True, test_used=False,
        checkpoint_selection='all validation observations waypoint RMSE, including initialization',
        sampling='sequence probability proportional to frame sampling mass; mean loss per sequence',
        source_mix=[.65, .35], route_mix=[.2, .4, .4], full_episode_prefixes=True))
    all_results = []
    for seed in seeds:
        for variant in variants:
            seed_everything(seed)
            directory = output/variant/f'seed_{seed}'
            if resume and (directory/'summary.json').exists():
                all_results.append(read_json(directory/'summary.json'))
                print('reuse completed', variant, seed, flush=True)
                continue
            directory.mkdir(parents=True, exist_ok=resume)
            compact = CompactRouteHead()
            compact.load_state_dict(compact_weights)
            if variant in ('scene', 'scene_points'):
                head = PersistentSceneHead(use_points=variant == 'scene_points')
                head.initialize(compact)
            else:
                head = compact
            head.to(device)
            old_names = set(compact_weights)
            optimizer = torch.optim.AdamW([
                dict(params=[p for name, p in head.named_parameters() if name in old_names], lr=3e-5),
                dict(params=[p for name, p in head.named_parameters() if name not in old_names], lr=3e-4)], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            best = score(head, variant, validation, batch_size)
            initial = copy.deepcopy(best)
            selected_epoch = 0
            torch.save(head.state_dict(), directory/'head.pt')
            records = [dict(epoch=0, **{k: v for k, v in best.items() if k != 'episode_mse'})]
            generator = torch.Generator().manual_seed(seed)
            training_started = time.monotonic()
            for epoch in range(1, epochs+1):
                started = time.monotonic()
                order = torch.multinomial(probabilities, draws, replacement=True, generator=generator).numpy()
                head.train()
                loss_sum = observations = 0
                for start in range(0, draws, batch_size):
                    batch = training.batch(order[start:start+batch_size])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                        prediction, auxiliary = predict(head, variant, batch)
                        loss = training_loss(head, prediction, auxiliary, batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss: {variant} epoch {epoch}')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(order[start:start+batch_size])
                    observations += int(batch['valid'].sum())
                current = score(head, variant, validation, batch_size)
                if current['rmse_m'] < best['rmse_m']:
                    best, selected_epoch = current, epoch
                    torch.save(head.state_dict(), directory/'head.pt')
                scheduler.step()
                record = dict(epoch=epoch, loss=loss_sum/draws, seconds=time.monotonic()-started,
                              training_observations=observations, **{k: v for k, v in current.items() if k != 'episode_mse'})
                records.append(record)
                write_json(directory/'epochs.json', records)
                write_json(output/'status.json', dict(variant=variant, seed=seed, **record))
                print('train', variant, seed, json.dumps(record), flush=True)
            head.load_state_dict(torch.load(directory/'head.pt', map_location=device, weights_only=True))
            exported = dict(checkpoint)
            exported.update(head={k: v.cpu() for k, v in head.state_dict().items()},
                architecture='persistent_scene' if isinstance(head, PersistentSceneHead) else 'compact_consensus',
                variant=variant if isinstance(head, PersistentSceneHead) else 'single_memory',
                memory_options=head.options() if isinstance(head, PersistentSceneHead) else {},
                route_history_length=1 if variant == 'current' else 4,
                experiment=dict(variant=variant, seed=seed, selected_epoch=selected_epoch, frozen_backbone=True))
            save_checkpoint(directory/'best.pt', exported)
            summary = dict(variant=variant, seed=seed, selected_epoch=selected_epoch,
                initial=initial, selected=best, training_seconds=time.monotonic()-training_started,
                parameters=sum(p.numel() for p in head.parameters()),
                benchmark=benchmark(head, variant, validation))
            write_json(directory/'summary.json', summary)
            all_results.append(summary)
            write_json(output/'summary.json', dict(results=all_results, frozen_backbone=True, test_used=False))
            del head, optimizer
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    write_json(output/'complete.json', dict(models=len(all_results), epochs=epochs, frozen_backbone=True))


@torch.inference_mode()
def memory_ablation(head, data, batch_size=32):
    """Reset at each frame using the SAME trained weights; no retraining."""
    squared = count = 0
    head.eval()
    for start in range(0, len(data.sequences), batch_size):
        batch = data.batch(list(range(start, min(start+batch_size, len(data.sequences)))))
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=data.device.type == 'cuda'):
            args = [batch[k] for k in ('tokens', 'geometry', 'pose', 'frame', 'valid', 'state', 'tcp')]
            prediction, _, _ = head.sequence(*args, points=batch['points'], resets=batch['valid'])
        error = (prediction-batch['waypoint']).square().mean((-1, -2))
        squared += float(error[batch['valid']].sum())
        count += int(batch['valid'].sum())
    return (squared/count)**.5


@torch.inference_mode()
def geometry_retention(head, data):
    """Recall of first-view teacher surfaces absent from the four latest views.

    Depth defines evaluation targets only. All three candidate representations
    use RGB-predicted positions. This is a surface recall diagnostic, not a
    collision metric or a guarantee that the scene is static.
    """
    counts = dict(targets=0, current=0, four=0, persistent=0)
    distance_sum = dict(current=0., four=0., persistent=0.)
    episodes = 0
    for index, sequence in enumerate(data.sequences):
        if len(sequence) <= 4:
            continue
        batch = data.batch([index])
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=data.device.type == 'cuda'):
            _, auxiliary = predict(head, 'scene_points', batch)
        teacher = batch['point_teacher'][0].float()
        observed = batch['point_valid'][0]
        target = teacher[0, observed[0]]
        workspace = (target[:, 0] > .10) & (target[:, 0] < 1.05) & (target[:, 1].abs() < .60)
        workspace &= (target[:, 2] > .04) & (target[:, 2] < .65)
        workspace &= (target-batch['tcp'][0, 0, :3, 3]).norm(dim=-1) > .07
        target = target[workspace]
        if not len(target):
            continue
        has_targets = False
        for t in range(4, len(sequence)):
            recent_teacher = teacher[t-3:t+1][observed[t-3:t+1]]
            if not len(recent_teacher):
                continue
            hidden = torch.cdist(target[None], recent_teacher[None])[0].amin(-1) > .04
            old = target[hidden]
            if not len(old):
                continue
            has_targets = True
            counts['targets'] += len(old)
            candidates = dict(current=batch['points'][0, t], four=batch['points'][0, t-3:t+1].flatten(0, 1),
                persistent=auxiliary['memory_points'][0, t][auxiliary['memory_valid'][0, t]])
            for name, points in candidates.items():
                distance = torch.cdist(old[None], points.float()[None])[0].amin(-1) if len(points) else old.new_full((len(old),), 1.)
                counts[name] += int((distance <= .04).sum())
                distance_sum[name] += float(distance.sum())
        episodes += int(has_targets)
    return dict(episodes_with_targets=episodes, target_observation_pairs=counts['targets'], tolerance_m=.04,
        recall={k: counts[k]/max(counts['targets'], 1) for k in distance_sum},
        mean_nearest_distance_m={k: v/max(counts['targets'], 1) for k, v in distance_sum.items()},
        definition='first-view surface points >4cm from all depth surfaces in the four latest observations')


def analyze_study(roots, cache, output, device):
    """Aggregate all seeds; freeze a validation-selected seed for each rollout."""
    create_output(output)
    data = SequenceCache(cache/'validation', device)
    results = []
    for root in roots:
        results.extend(read_json(root/'summary.json')['results'])
    aggregate, selected_paths, ablations = {}, {}, {}
    for variant in ('current', 'four', 'scene', 'scene_points'):
        rows = [r for r in results if r['variant'] == variant]
        if not rows:
            continue
        errors = np.array([r['selected']['rmse_m'] for r in rows])
        late = np.array([r['selected']['after_four_rmse_m'] for r in rows])
        aggregate[variant] = dict(seeds=len(rows), rmse_m_mean=float(errors.mean()), rmse_m_std=float(errors.std(ddof=1)) if len(rows)>1 else 0.,
            after_four_rmse_m_mean=float(late.mean()),
            mean_training_seconds=float(np.mean([r['training_seconds'] for r in rows])),
            forward_frames_per_second=float(np.mean([r['benchmark']['valid_frames_per_second'] for r in rows])),
            selected_epochs=[r['selected_epoch'] for r in rows])
        chosen = min(rows, key=lambda r: r['selected']['rmse_m'])
        path = next(root/variant/f"seed_{chosen['seed']}"/'best.pt' for root in roots if (root/variant/f"seed_{chosen['seed']}"/'best.pt').exists())
        selected_paths[variant] = str(path)
        if variant in ('scene', 'scene_points'):
            checkpoint = load_checkpoint(path)
            head = PersistentSceneHead(**checkpoint['memory_options']).to(device)
            head.load_state_dict(checkpoint['head'])
            reset_error = memory_ablation(head, data)
            ablations[variant] = dict(persistent_rmse_m=chosen['selected']['rmse_m'], reset_each_frame_rmse_m=reset_error,
                                      memory_residual_norm=float(head.memory_residual.weight.norm()))
            if variant == 'scene_points':
                ablations[variant]['geometry_retention'] = geometry_retention(head, data)
            del head
    write_json(output/'summary.json', dict(aggregate=aggregate, selected_checkpoints=selected_paths,
                                         memory_ablations=ablations, frozen_backbone=True, test_used=False))
    # Define the paired rollout set before looking at closed-loop outcomes.
    checkpoint = load_checkpoint(next(iter(selected_paths.values())))
    from tsn.data.splits import episode_catalog, ROUTES
    catalog = episode_catalog(Path(checkpoint['config']['benchmark']['root']))
    ids = checkpoint['splits']['validation']
    pilot = [ep for route in ROUTES for ep in [x for x in sorted(ids) if catalog[x] == route][:4]]
    write_json(output/'rollout_protocol.json', dict(partition='validation', episodes=pilot,
               selection='first four sorted validation episodes per route', paired=True,
               checkpoints=selected_paths, clearance_margin=.04, clearance_penalty=.08,
               geometry_refiner='same four raw predicted clouds for all variants'))
    write_json(output/'complete.json', dict(models=len(results), analyzed=True))


def report_study(analysis, roots, rollout_root, output, cache):
    """Paired episode bootstrap and a source snapshot for reproducible review."""
    create_output(output)
    report = read_json(analysis/'summary.json')
    rows = [r for root in roots for r in read_json(root/'summary.json')['results']]
    offsets = np.load(cache/'validation/sequence_offsets.npy')
    lengths = np.diff(offsets)
    rng = np.random.default_rng(20261005)
    indices = rng.integers(len(lengths), size=(5000, len(lengths)))
    weights = lengths[indices]
    baseline = np.mean([r['selected']['episode_mse'] for r in rows if r['variant'] == 'four'], axis=0)
    bootstrap = {}
    for variant in ('current', 'scene', 'scene_points'):
        mse = np.mean([r['selected']['episode_mse'] for r in rows if r['variant'] == variant], axis=0)
        differences = np.sqrt((baseline[indices]*weights).sum(1)/weights.sum(1))-np.sqrt((mse[indices]*weights).sum(1)/weights.sum(1))
        estimate = np.sqrt((baseline*lengths).sum()/lengths.sum())-np.sqrt((mse*lengths).sum()/lengths.sum())
        bootstrap[variant] = dict(rmse_reduction_m=float(estimate),
            episode_bootstrap_95_ci_m=np.quantile(differences, [.025, .975]).tolist())
    report['paired_offline_comparison_against_four'] = bootstrap
    report['uncertainty_method'] = '5000 paired episode bootstraps; seed-averaged MSE; validation used for checkpoint selection, so exploratory'
    rollout = {variant: read_json(rollout_root/variant/'closed_loop.json') for variant in report['aggregate']}
    from tsn.evaluation.metrics import summarize_rollouts
    report['closed_loop'] = {variant: summarize_rollouts(value['results']) for variant, value in rollout.items()}
    paired = {}
    baseline_results = {r['episode_id']: r for r in rollout['four']['results']}
    for variant, value in rollout.items():
        results = {r['episode_id']: r for r in value['results']}
        if set(results) != set(baseline_results):
            raise ValueError('Rollout sets are not paired')
        ids = sorted(results)
        difference = np.array([int(results[ep]['success'])-int(baseline_results[ep]['success']) for ep in ids])
        boot = difference[rng.integers(len(ids), size=(5000, len(ids)))].mean(1)
        paired[variant] = dict(episodes=len(ids), success_rate_difference=float(difference.mean()),
            episode_bootstrap_95_ci=np.quantile(boot, [.025, .975]).tolist(),
            gained_successes=int((difference>0).sum()), lost_successes=int((difference<0).sum()))
    report['paired_closed_loop_against_four'] = paired
    report['rollout_protocol'] = read_json(rollout_root/'protocol.json')
    report.pop('test_used', None)
    report['test_used_for_training'] = False
    report['test_used_for_model_selection'] = False
    report['test_evaluated'] = report['rollout_protocol']['partition'] == 'test'
    report['limitations'] = ['Frozen Pi3 weights were already trained; this measures replacing the temporal head.',
        'Geometry slots are coarse spatial patches, not free tracked landmarks or a complete collision map.',
        'Robot suppression uses hand/camera proximity only, not full robot segmentation.',
        'Geometry retention is a depth-supervised recall diagnostic; correlated queries are not independent trials.',
        ('Clearance is disabled; only compact route execution, IK and terminal servo are used.'
         if report['rollout_protocol'].get('controller') == 'compact' else
         'The deterministic clearance refiner retains the same four raw predicted clouds in every variant.'),
        'Validation was used for checkpoint selection; validation comparisons are exploratory. Test checkpoints were locked before test evaluation.',
        'The prefix scan uses O(T log T) work in PyTorch; no fused linear-work kernel is included.']
    source = Path(__file__).resolve().parents[3]
    snapshot = output/'source'
    snapshot.mkdir()
    for name in ('src', 'scripts', 'tests', 'configs'):
        shutil.copytree(source/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('Dockerfile', 'pyproject.toml', 'README.md'):
        shutil.copy2(source/name, snapshot/name)
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in snapshot.rglob('*') if p.is_file()}
    write_json(output/'source_sha256.json', hashes)
    write_json(output/'report.json', report)
    write_json(output/'complete.json', dict(models=len(rows), paired_episodes=len(baseline_results), frozen_backbone=True))
