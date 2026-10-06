"""Fine-tune the original compact head on the retained corrective histories."""
import copy
import hashlib
import time

import torch

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import create_output, write_json
from tsn.common.seed import seed_everything
from tsn.models.compact_policy import CompactRouteHead, compact_state
from tsn.training.corrective import SequenceCache, predict, score, training_loss


@torch.inference_mode()
def fit_diagnostics(head, initial, data, batch_size):
    selected = [i for i, s in enumerate(data.sequences) if int(data.arrays['source'][s[0]]) == 2]
    totals = {name: dict(observations=0, squared=0., initial_squared=0., drift_squared=0.)
              for name in ('safe', 'corrected')}
    head.eval()
    initial.eval()
    for start in range(0, len(selected), batch_size):
        batch = data.batch(selected[start:start+batch_size])
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=data.device.type == 'cuda'):
            prediction, _ = predict(head, batch)
            before, _ = predict(initial, batch)
        errors = dict(squared=(prediction-batch['waypoint']).square().mean((-1, -2)),
                      initial_squared=(before-batch['waypoint']).square().mean((-1, -2)),
                      drift_squared=(prediction-before).square().mean((-1, -2)))
        valid = batch['valid'] & batch['supervision_valid']
        for name, risky in (('safe', False), ('corrected', True)):
            mask = valid & (batch['policy_unsafe'] == risky)
            totals[name]['observations'] += int(mask.sum())
            for key, value in errors.items():
                totals[name][key] += float(value[mask].sum())
    return {name: dict(observations=row['observations'],
                      target_rmse_m=(row['squared']/max(row['observations'], 1))**.5,
                      initial_target_rmse_m=(row['initial_squared']/max(row['observations'], 1))**.5,
                      change_from_initial_rms_m=(row['drift_squared']/max(row['observations'], 1))**.5)
            for name, row in totals.items()}


def train(checkpoint_path, cache, recoveries, output, device, epochs=20, draws=4096,
          batch_size=32, seed=20261005, learning_rate=3e-5):
    if not recoveries or min(epochs, draws, batch_size) < 1 or learning_rate <= 0:
        raise ValueError('Verified corrective caches and positive training sizes are required')
    create_output(output)
    checkpoint = load_checkpoint(checkpoint_path)
    _, weights = compact_state(checkpoint)
    seed_everything(seed)
    head = CompactRouteHead()
    head.load_state_dict(weights, strict=True)
    head.to(device)
    initial = copy.deepcopy(head).eval().requires_grad_(False)
    training = SequenceCache(cache/'train', device, recoveries)
    validation = SequenceCache(cache/'validation', device)
    for partition, data in (('train', training), ('validation', validation)):
        if data.metadata['episode_ids'] != checkpoint['splits'][partition]:
            raise ValueError('Cache and model splits differ')
    probabilities = training.sampling()
    first = torch.tensor([s[0] for s in training.sequences], device=device)
    if (probabilities[training.arrays['source'][first].cpu() != 2] != 0).any():
        raise RuntimeError('Noncorrective observations entered training sampling')
    write_json(output/'protocol.json', dict(
        checkpoint=str(checkpoint_path), cache=str(cache), recovery_caches=[str(p) for p in recoveries],
        epochs=epochs, draws=draws, batch_size=batch_size, seed=seed, learning_rate=learning_rate,
        optimizer='AdamW', weight_decay=1e-4, scheduler='CosineAnnealingLR', gradient_clip=1.,
        source_mix=[0, 0, 1], route_mix=[.2, .4, .4],
        loss='weighted per-sequence Huber on six TCP offsets; delta=0.02 m',
        frozen_backbone=True, trainable_parameters=[n for n, _ in head.named_parameters()],
        checkpoint_selection='fixed final epoch', controller='compact', clearance='disabled',
        route_history_length=4, test_used=False,
        initialization_note='Original compact weights; evaluated four-frame baseline checkpoint was removed',
        recovery_metadata_sha256={str(p): hashlib.sha256((p/'metadata.json').read_bytes()).hexdigest()
                                  for p in recoveries}))
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    generator = torch.Generator().manual_seed(seed)
    records = [dict(epoch=0, **score(head, validation, batch_size))]
    draws_hash = hashlib.sha256()
    started = time.monotonic()
    for epoch in range(1, epochs+1):
        epoch_started = time.monotonic()
        order = torch.multinomial(probabilities, draws, replacement=True, generator=generator).numpy()
        draws_hash.update(order.tobytes())
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
        record = dict(epoch=epoch, loss=total/draws, seconds=time.monotonic()-epoch_started,
                      **score(head, validation, batch_size))
        records.append(record)
        write_json(output/'epochs.json', records)
        write_json(output/'status.json', record)
        print(record, flush=True)
    exported = dict(checkpoint)
    exported.update(architecture='compact_consensus', variant='single_memory', route_history_length=4,
                    head={k: v.cpu() for k, v in head.state_dict().items()},
                    experiment=dict(variant='compact_corrective', seed=seed, selected_epoch=epochs,
                                    frozen_backbone=True, learning_rate=learning_rate,
                                    training_protocol=str(output/'protocol.json')))
    save_checkpoint(output/'best.pt', exported)
    reloaded = CompactRouteHead().to(device)
    reloaded.load_state_dict(compact_state(load_checkpoint(output/'best.pt'))[1], strict=True)
    for k, v in head.state_dict().items():
        if not torch.equal(v, reloaded.state_dict()[k]):
            raise RuntimeError('Export/reload mismatch: '+k)
    write_json(output/'summary.json', dict(
        initial=records[0], selected=records[-1], seconds=time.monotonic()-started,
        trainable_parameters=sum(p.numel() for p in head.parameters()),
        sampled_sequence_indices_sha256=draws_hash.hexdigest(), export_reload_verified=True,
        corrective_training_fit=fit_diagnostics(head, initial, training, batch_size)))
    write_json(output/'complete.json', dict(epochs=epochs, test_used=False, frozen_backbone=True))
