"""Corrective-only training for the retained current-view control.

Only the added residual trains. Geometry losses on frozen parameters have been
removed because they contribute no gradient. Teacher search is collection-only.
"""
import time

import numpy as np
import torch
from torch.nn import functional as F

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import create_output, read_json, write_json
from tsn.common.seed import seed_everything
from tsn.models.compact_policy import CompactRouteHead
from tsn.models.current_view_policy import CurrentViewHead


class SequenceCache:
    """Small cadence-sampled cache resident on the training device."""
    def __init__(self, directory, device, recoveries=()):
        self.device = device
        arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in directory.glob('*.npy')}
        offsets, indices = arrays.pop('sequence_offsets'), arrays.pop('sequence_indices')
        self.sequences = [np.array(indices[offsets[i]:offsets[i+1]]) for i in range(len(offsets)-1)]
        self.metadata = read_json(directory/'metadata.json')
        arrays.pop('original_indices', None)
        arrays.setdefault('supervision_valid', np.ones(len(arrays['state']), dtype=bool))
        arrays.setdefault('action_weight', np.ones(len(arrays['state']), dtype=np.float32))
        arrays.setdefault('policy_unsafe', np.zeros(len(arrays['state']), dtype=bool))
        if recoveries:
            arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
            seen_recovery = set()
            for recovery in recoveries:
                metadata = read_json(recovery/'metadata.json')
                if metadata.get('teacher_mode') != 'corrective':
                    raise ValueError('Only corrective training caches are supported')
                if metadata['episode_ids'] != self.metadata['episode_ids'] or metadata.get('partition') != 'train':
                    raise ValueError('Recovery episodes must match the training split exactly')
                if not (recovery/'complete.json').is_file() or metadata.get('teacher_verified_steps') != 30:
                    raise ValueError('Only completed physics-verified recovery caches are accepted')
                collected = set(metadata['collected_episode_ids'])
                if not collected <= set(self.metadata['episode_ids']) or collected & seen_recovery:
                    raise ValueError('Recovery shards must be disjoint subsets of training episodes')
                seen_recovery.update(collected)
                extra = {k: np.load(recovery/f'{k}.npy', allow_pickle=False) for k in arrays
                         if k not in ('action_weight', 'policy_unsafe') or (recovery/f'{k}.npy').exists()}
                extra.setdefault('action_weight', np.ones(len(extra['state']), dtype=np.float32))
                extra.setdefault('policy_unsafe', np.zeros(len(extra['state']), dtype=bool))
                if not np.isfinite(extra['action_weight']).all() or (extra['action_weight'] <= 0).any():
                    raise ValueError('Action weights must be finite and positive')
                if not np.all(extra['source'] == 2) or not extra['supervision_valid'].any():
                    raise ValueError('Invalid on-policy recovery source or empty supervision')
                offset = len(arrays['state'])
                indices = np.load(recovery/'sequence_indices.npy', allow_pickle=False)
                offsets = np.load(recovery/'sequence_offsets.npy', allow_pickle=False)
                self.sequences.extend(np.array(indices[offsets[i]:offsets[i+1]])+offset for i in range(len(offsets)-1))
                arrays = {k: np.concatenate((v, extra[k]), 0) for k, v in arrays.items()}
        self.arrays = {k: torch.from_numpy(np.array(v, copy=True)).to(device) for k, v in arrays.items()}

    def batch(self, selected):
        sequences = [self.sequences[i] for i in selected]
        length = max(map(len, sequences))
        index = np.stack([np.pad(s, (0, length-len(s)), mode='edge') for s in sequences])
        valid = torch.arange(length, device=self.device)[None] < torch.tensor([len(s) for s in sequences], device=self.device)[:, None]
        index = torch.from_numpy(index).to(self.device)
        values = {k: self.arrays[k][index] for k in ('tokens', 'geometry', 'pose', 'frame', 'state', 'tcp',
                  'waypoint', 'teacher', 'geometry_valid', 'points', 'point_teacher', 'point_valid', 'supervision_valid',
                  'action_weight', 'policy_unsafe')}
        values['valid'] = valid
        return values

    def sampling(self):
        labeled = self.arrays['supervision_valid'].cpu().numpy()
        lengths = np.array([int(labeled[s].sum()) for s in self.sequences])
        first = torch.tensor([s[0] for s in self.sequences], device=self.device)
        routes = self.arrays['route'][first].cpu().numpy()
        sources = self.arrays['source'][first].cpu().numpy()
        weights = np.zeros(len(lengths))
        for route, probability in enumerate((.2, .4, .4)):
            selected = (sources == 2) & (routes == route)
            if not selected.any() or lengths[selected].sum() == 0:
                raise ValueError('Training needs verified histories for every route')
            weights[selected] = lengths[selected]*probability/lengths[selected].sum()
        return torch.from_numpy(weights)


def compact_sequence(head, batch):
    """Apply the deployed four-observation head causally at every timestep."""
    tokens, valid = batch['tokens'], batch['valid']
    batch_size, length = tokens.shape[:2]
    cursor = torch.arange(length, device=tokens.device)[None].minimum(valid.sum(1)[:, None]-1)
    offsets = torch.arange(3, -1, -1, device=tokens.device)
    indices = (cursor[..., None]-offsets).clamp_min(0)
    b = torch.arange(batch_size, device=tokens.device)[:, None, None]
    mask = valid[b, indices] & (cursor[..., None] >= offsets)
    gathered = [batch[k][b, indices].reshape(batch_size*length, 4, *batch[k].shape[2:])
                for k in ('tokens', 'geometry', 'pose')]
    ages = (batch['frame'][..., None]-batch['frame'][b, indices]).reshape(-1, 4)
    prediction, auxiliary = head(*gathered, ages, mask.reshape(-1, 4),
                                 batch['state'].reshape(-1, 16), batch['tcp'].reshape(-1, 4, 4))
    return prediction.reshape(batch_size, length, 6, 3), {
        k: v.reshape(batch_size, length, *v.shape[1:]) for k, v in auxiliary.items()}


def predict(head, batch):
    if isinstance(head, CompactRouteHead) and not hasattr(head, 'sequence'):
        return compact_sequence(head, batch)
    prediction, auxiliary, _ = head.sequence(*[batch[k] for k in
        ('tokens', 'geometry', 'pose', 'frame', 'valid', 'state', 'tcp')], points=batch['points'])
    return prediction, auxiliary


def sequence_mean(value, mask):
    """Mean of valid elements per sequence, avoiding length-dependent sampling."""
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    mask = mask.expand_as(value).float()
    return (value.float()*mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)


def training_loss(prediction, batch):
    mask = (batch['valid'] & batch['supervision_valid']).float()*batch.get('action_weight', 1.)
    return sequence_mean(F.huber_loss(prediction, batch['waypoint'], delta=.02, reduction='none'), mask).mean()


@torch.inference_mode()
def score(head, data, batch_size):
    head.eval()
    squared, count = 0., 0
    for start in range(0, len(data.sequences), batch_size):
        batch = data.batch(list(range(start, min(start+batch_size, len(data.sequences)))))
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=data.device.type == 'cuda'):
            prediction, _ = predict(head, batch)
        valid = batch['valid'] & batch['supervision_valid']
        error = (prediction-batch['waypoint']).square().mean((-1, -2))
        squared += float(error[valid].sum())
        count += int(valid.sum())
    return dict(rmse_m=(squared/max(count, 1))**.5, observations=count)


@torch.inference_mode()
def corrective_fit(head, data, batch_size):
    """Training diagnostics: corrected-target fit versus drift on safe proposals."""
    selected = [i for i, sequence in enumerate(data.sequences) if int(data.arrays['source'][sequence[0]]) == 2]
    totals = {name: dict(observations=0, squared=0., parent_squared=0., drift_squared=0.)
              for name in ('safe', 'corrected')}
    for start in range(0, len(selected), batch_size):
        batch = data.batch(selected[start:start+batch_size])
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=data.device.type == 'cuda'):
            prediction, auxiliary = predict(head, batch)
        parent = auxiliary['current_prediction']
        error = (prediction-batch['waypoint']).square().mean((-1, -2))
        parent_error = (parent-batch['waypoint']).square().mean((-1, -2))
        drift = (prediction-parent).square().mean((-1, -2))
        valid = batch['valid'] & batch['supervision_valid']
        for name, risky in (('safe', False), ('corrected', True)):
            mask = valid & (batch['policy_unsafe'] == risky)
            totals[name]['observations'] += int(mask.sum())
            for key, value in (('squared', error), ('parent_squared', parent_error), ('drift_squared', drift)):
                totals[name][key] += float(value[mask].sum())
    return {name: dict(observations=row['observations'],
                target_rmse_m=(row['squared']/max(row['observations'], 1))**.5,
                parent_target_rmse_m=(row['parent_squared']/max(row['observations'], 1))**.5,
                change_from_parent_rms_m=(row['drift_squared']/max(row['observations'], 1))**.5)
            for name, row in totals.items()}



def train(checkpoint_path, initial_head, initial_residual, cache, recoveries, output, device,
          epochs=20, draws=4096, batch_size=32, seed=20261005):
    if not recoveries or min(epochs, draws, batch_size) < 1:
        raise ValueError('Verified corrective caches and positive training sizes are required')
    create_output(output)
    checkpoint, parent = load_checkpoint(checkpoint_path), load_checkpoint(initial_head)
    if checkpoint['splits'] != parent['splits']:
        raise ValueError('Backbone and initialization splits differ')
    seed_everything(seed)
    source = CurrentViewHead.from_checkpoint(parent)
    head = CurrentViewHead(source.grid_shape, source.point_width, corrective=True)
    missing, unexpected = head.load_state_dict(source.state_dict(), strict=False)
    if unexpected or any(not k.startswith('history_output.') for k in missing):
        raise ValueError('Incompatible parent initialization')
    if initial_residual is not None:
        initialization = load_checkpoint(initial_residual)
        if initialization['seed'] != seed:
            raise ValueError('Saved residual initialization requires its recorded seed')
        head.history_output.load_state_dict(initialization['state'], strict=True)
    for name, parameter in head.named_parameters():
        parameter.requires_grad_(name.startswith('history_output.'))
    frozen = {k: v.clone() for k, v in head.state_dict().items() if not k.startswith('history_output.')}
    head.to(device)
    training = SequenceCache(cache/'train', device, recoveries)
    validation = SequenceCache(cache/'validation', device)
    for partition, data in (('train', training), ('validation', validation)):
        if data.metadata['episode_ids'] != checkpoint['splits'][partition]:
            raise ValueError('Cache and model splits differ')
    probabilities = training.sampling()
    write_json(output/'protocol.json', dict(checkpoint=str(checkpoint_path), initial_head=str(initial_head),
        initial_residual=str(initial_residual) if initial_residual else None, cache=str(cache),
        recovery_caches=[str(p) for p in recoveries], epochs=epochs, draws=draws, batch_size=batch_size,
        seed=seed, source_mix=[0, 0, 1], route_mix=[.2, .4, .4], frozen_parent=True,
        trainable_parameters=[n for n, p in head.named_parameters() if p.requires_grad],
        checkpoint_selection='fixed final epoch', controller='compact', clearance='disabled', test_used=False))
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    generator = torch.Generator().manual_seed(seed)
    records = [dict(epoch=0, **score(head, validation, batch_size))]
    started = time.monotonic()
    for epoch in range(1, epochs+1):
        order = torch.multinomial(probabilities, draws, replacement=True, generator=generator).numpy()
        head.train()
        total = 0.
        for start in range(0, draws, batch_size):
            indices = order[start:start+batch_size]
            batch = training.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                prediction, _ = predict(head, batch)
                loss = training_loss(prediction, batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite corrective loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(indices)
        scheduler.step()
        record = dict(epoch=epoch, loss=total/draws, **score(head, validation, batch_size))
        records.append(record)
        write_json(output/'epochs.json', records)
        write_json(output/'status.json', record)
        print(record, flush=True)
    for name, value in frozen.items():
        if not torch.equal(value, head.state_dict()[name].cpu()):
            raise RuntimeError('Frozen parent changed: '+name)
    exported = dict(checkpoint)
    exported.pop('memory_options', None)
    exported.update(architecture='current_view', variant='corrective_current', head_options=head.options(),
        head={k: v.cpu() for k, v in head.state_dict().items()},
        experiment=dict(seed=seed, selected_epoch=epochs, frozen_parent=True))
    save_checkpoint(output/'best.pt', exported)
    write_json(output/'summary.json', dict(initial=records[0], selected=records[-1],
        frozen_parent_verified=True, seconds=time.monotonic()-started,
        corrective_training_fit=corrective_fit(head, training, batch_size)))
    write_json(output/'complete.json', dict(epochs=epochs, test_used=False))
