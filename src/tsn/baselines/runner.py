"""Train with validation-only selection; evaluate through the shared simulator."""
import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from tsn.baselines.data import CachedDataset, cache_signature
from tsn.baselines.policies import BaselinePolicy
from tsn.common.checkpoint import save_checkpoint
from tsn.common.config import create_output, read_json, write_json
from tsn.common.seed import seed_everything
from tsn.data.navigation import sampling_weights
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.features.maps import GeometryMaps
from tsn.models.kinematics import PandaKinematics


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def waypoint_rmse(model, loader, kinematics, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    squared, count = 0., 0
    joint_squared, joint_count = 0., 0
    for raw in loader:
        batch = move(raw, device)
        prediction = model.sample(batch, generator)
        qpos = batch['qpos'][:, None, :7]
        actual = kinematics(qpos+batch['target'][:, 4::5])[..., :3, 3]
        predicted = kinematics(qpos+prediction[:, 4::5])[..., :3, 3]
        squared += float((actual-predicted).square().sum())
        count += actual.numel()
        joint_squared += float((prediction-batch['target']).square().sum())
        joint_count += prediction.numel()
    if not count:
        raise ValueError('Empty validation partition')
    return (squared/count)**.5, (joint_squared/joint_count)**.5


def train(config, cache, output, device):
    output = Path(output)
    create_output(output)
    options = config['train']
    seed_everything(options['seed'])
    receipt = read_json(Path(cache)/'complete.json')
    if receipt['signature'] != cache_signature(config):
        raise ValueError('Cache does not match the configured benchmark/preprocessing')
    splits = read_json(Path(cache)/'splits.json')
    validate_splits(splits, episode_catalog(Path(config['benchmark']['root'])))
    model = BaselinePolicy(config).to(device)
    modality = 'points' if model.requires_depth else 'rgb'
    training, validation = [CachedDataset(cache, p, modality) for p in ('train', 'validation')]
    normalization = read_json(Path(cache)/'normalization.json')
    model.action_center.copy_(torch.tensor(normalization['center'], device=device))
    model.action_scale.copy_(torch.tensor(normalization['scale'], device=device))
    generator = torch.Generator().manual_seed(options['seed'])
    sampler = WeightedRandomSampler(sampling_weights(training.route, training.source), options['draws_per_epoch'],
                                    replacement=True, generator=generator)
    loader = DataLoader(training, batch_size=options['batch_size'], sampler=sampler,
        num_workers=options['num_workers'], persistent_workers=options['num_workers'] > 0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(validation, batch_size=options['batch_size'], shuffle=False,
        num_workers=options['num_workers'], persistent_workers=options['num_workers'] > 0, pin_memory=device.type == 'cuda')
    write_json(output/'config.json', config)
    write_json(output/'splits.json', splits)
    write_json(output/'initialization.json', {'from_scratch': True, 'method': model.method,
        'parameters': sum(x.numel() for x in model.parameters()), 'observation_modality': model.observation_modality,
        'depth_input_at_inference': model.requires_depth, 'history_slots': 2, 'training_cache': receipt,
        'test_used_for_selection': False, 'seed': options['seed'], 'draws_per_epoch': options['draws_per_epoch'],
        'global_batch_size': options['batch_size'], 'checkpoint_selection': 'minimum validation waypoint RMSE',
        'normalization': normalization, 'prediction_horizon': 30, 'execute_horizon': 15})
    if model.method == 'carp':
        optimizer = torch.optim.AdamW(model.tokenizer.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
        records = []
        for epoch in range(1, config['baseline']['tokenizer_epochs']+1):
            model.tokenizer.train()
            total, count = 0., 0
            started = time.monotonic()
            for raw in loader:
                batch = move(raw, device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.tokenizer.loss(model.normalize(batch['target']))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite tokenizer loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.tokenizer.parameters(), options['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(batch['target'])
                count += len(batch['target'])
            records.append({'epoch': epoch, 'loss': total/count, 'seconds': time.monotonic()-started})
            write_json(output/'tokenizer_epochs.json', records)
            print({'tokenizer': records[-1]}, flush=True)
        model.tokenizer.requires_grad_(False)
        model.tokenizer.eval()
        save_checkpoint(output/'tokenizer.pt', {'model': model.tokenizer.state_dict(), 'epochs': len(records),
                                                'fit_partition': 'train'})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=options['learning_rate'], weight_decay=options['weight_decay'])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    kinematics = PandaKinematics().to(device)
    best, records, updates = float('inf'), [], 0
    for epoch in range(1, options['epochs']+1):
        model.train()
        if model.method == 'carp':
            model.tokenizer.eval()
        total, count = 0., 0
        started = time.monotonic()
        for index, raw in enumerate(loader, 1):
            batch = move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite baseline loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, options['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            decay = min(.999, 1-(1+updates)**(-.75))
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.parameters()):
                    averaged.lerp_(current, 1-decay)
            size = len(batch['target'])
            total += float(loss.detach())*size
            count += size
            if index % 128 == 0 or index == len(loader):
                progress = {'method': model.method, 'epoch': epoch, 'batch': index, 'batches': len(loader),
                            'samples': count, 'loss': total/count, 'seconds': time.monotonic()-started}
                write_json(output/'progress.json', progress)
                print(progress, flush=True)
        score, joint_score = waypoint_rmse(ema, val_loader, kinematics, device, options['seed']+100000)
        record = {'epoch': epoch, 'loss': total/count, 'validation_rmse_m': score,
                  'validation_joint_rmse_rad': joint_score, 'seconds': time.monotonic()-started}
        records.append(record)
        checkpoint = {'format_version': 1, 'architecture': 'tsn_baseline', 'method': model.method,
            'config': config, 'splits': splits, 'model': {k: v.detach().cpu() for k, v in ema.state_dict().items()},
            'selected_epoch': epoch, 'validation_rmse_m': score, 'test_used_for_selection': False, 'ema': True}
        if score < best:
            best = score
            save_checkpoint(output/'best.pt', checkpoint)
        save_checkpoint(output/'latest.pt', checkpoint)
        write_json(output/'epochs.json', records)
        print(record, flush=True)
    checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
    write_json(output/'selected_checkpoint.json', {'selected_epoch': checkpoint['selected_epoch'],
        'validation_rmse_m': best, 'sha256': hashlib.sha256((output/'best.pt').read_bytes()).hexdigest(),
        'test_used_for_selection': False})
    write_json(output/'complete.json', {'epochs': len(records), 'best_validation_rmse_m': best,
        'test_used_for_selection': False, 'updates': updates, 'tokenizer_fitted_on_train_only': model.method == 'carp'})


def evaluate(checkpoint_path, partition, output, device, episodes=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['architecture'] != 'tsn_baseline':
        raise ValueError('Expected a standalone baseline checkpoint')
    config, splits = checkpoint['config'], checkpoint['splits']
    seed_everything(config['train']['seed'])
    root = Path(config['benchmark']['root'])
    catalog = episode_catalog(root)
    validate_splits(splits, catalog)
    ids = episodes or splits[partition]
    if not ids or len(ids) != len(set(ids)) or set(ids)-set(splits[partition]):
        raise ValueError('Episodes must be unique members of the selected partition')
    output = Path(output)
    create_output(output)
    model = BaselinePolicy(config).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    maps = GeometryMaps(config['model']['maps']).to(device)
    write_json(output/'config.json', {'checkpoint': str(Path(checkpoint_path).resolve()),
        'checkpoint_sha256': hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        'partition': partition, 'episodes': ids, 'dataset_root': str(root), 'eval': config['eval'],
        'full_partition': set(ids) == set(splits[partition]), 'method': model.method,
        'depth_input_at_inference': model.requires_depth})
    write_json(output/'splits.json', splits)
    evaluate_rollouts(ids, catalog, root, output, model, maps, device, config['eval'])
    write_json(output/'complete.json', {'episodes': len(ids), 'partition': partition,
        'architecture': 'tsn_baseline', 'method': model.method})
