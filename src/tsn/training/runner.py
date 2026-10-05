"""Train the compact memory head on frozen Pi3 features, then export one model."""
import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset

from tsn.common.checkpoint import save_checkpoint
from tsn.common.config import create_output, read_json, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.data.history import build_history
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.loaders import make_loader
from tsn.data.recovery_dataset import RecoveryDataset
from tsn.data.splits import episode_catalog, validate_splits
from tsn.features.state import policy_state
from tsn.models.clearance_policy import load_clearance_policy
from tsn.training.losses import geometry_nll


@torch.inference_mode()
def cache_features(root, model, maps, cfg, splits, options, device):
    """Cache train/validation observations once with the frozen Pi3 backbone."""
    model.eval().requires_grad_(False)
    data_root = Path(cfg['benchmark']['root'])
    catalog = episode_catalog(data_root)
    validate_splits(splits, catalog)
    for partition in ('train', 'validation'):
        directory = root/'cache'/partition
        directory.mkdir(parents=True, exist_ok=False)
        ids = splits[partition]
        expert = FrameDataset(data_root, ids, catalog, 30,
                              frame_stride=2 if partition == 'train' else 1, include_rgb=True)
        recovery = None
        if partition == 'train':
            recovery = RecoveryDataset(Path(cfg['train']['recovery_sources'][0]['path']), 30, include_rgb=True)
            if not set(recovery.route_by_episode) <= set(ids):
                raise ValueError('Recovery data overlaps held-out episodes')
        dataset = ConcatDataset([expert, recovery]) if recovery is not None else expert
        if recovery is not None and any(catalog[ep] != route for ep, route in recovery.route_by_episode.items()):
            raise ValueError('Recovery route labels differ from the benchmark')
        specs = {'tokens': ((16, 768), np.float16), 'geometry': ((16, 6), np.float32),
                 'teacher': ((16, 3), np.float32), 'geometry_valid': ((16,), bool),
                 'state': ((16,), np.float32),
                 'pose': ((4, 4), np.float32), 'waypoint': ((6, 3), np.float32),
                 'tcp': ((4, 4), np.float32), 'route': ((), np.int64),
                 'episode': ((), np.int64), 'frame': ((), np.int64), 'source': ((), np.int64)}
        arrays = {k: np.lib.format.open_memmap(directory/f'{k}.npy', mode='w+', dtype=d,
                                              shape=(len(dataset), *s)) for k, (s, d) in specs.items()}
        lookup = {ep: i for i, ep in enumerate(ids)}
        batches = make_loader(dataset, dict(device=str(device), batch_size=options['cache_batch_size'], num_workers=options['num_workers']), False, options['seed'])
        offset, started = 0, time.monotonic()
        for raw in batches:
            b = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
            state = policy_state(b['qpos'], b['goal_pose'], maps.settings)
            _, tokens, geometry = model.backbone(b['rgb'], state, b['K'], b['T_B_C'], return_features=True)
            teacher = maps(b['depth'], b['K'], b['T_B_C'], b['goal_pose'][:, :3], b['future_ee'], b['valid_future'])
            depth = F.interpolate(b['depth'][:, None], (maps.settings.height, maps.settings.width), mode='nearest-exact')
            valid = torch.isfinite(depth) & (depth >= maps.settings.near_m) & (depth <= maps.settings.far_m)
            tcp = model.kinematics(b['qpos'][:, :7])
            waypoint = model.kinematics(b['qpos'][:, None, :7]+b['target'][:, 4::5])[..., :3, 3]-tcp[:, None, :3, 3]
            values = dict(tokens=tokens, geometry=geometry, state=state,
                          pose=b['T_B_C'], waypoint=waypoint, tcp=tcp, route=b['route'], frame=b['frame_index'],
                          geometry_valid=F.adaptive_avg_pool2d(valid.float(), (4, 4)).flatten(1) >= .999,
                          teacher=F.adaptive_avg_pool2d(teacher[:, :3], (4, 4)).flatten(2).transpose(1, 2))
            size = len(state)
            for key, value in values.items():
                arrays[key][offset:offset+size] = value.float().cpu().numpy() if value.is_floating_point() else value.cpu().numpy()
            arrays['episode'][offset:offset+size] = [lookup[x.removesuffix('_recovery')] for x in raw['episode_id']]
            arrays['source'][offset:offset+size] = np.arange(offset, offset+size) >= len(expert)
            offset += size
            if offset % 2560 == 0 or offset == len(dataset):
                write_json(root/'status.json', dict(state='running', stage='cache', partition=partition, frames=offset, total=len(dataset)))
                print(f'cache {partition} {offset}/{len(dataset)} {time.monotonic()-started:.1f}s', flush=True)
        for value in arrays.values():
            value.flush()
        for key, value in zip(('history', 'ages', 'mask'), build_history(arrays['episode'], arrays['frame'], arrays['source'])):
            np.save(directory/f'{key}.npy', value)
        write_json(directory/'metadata.json', dict(samples=len(dataset), expert_samples=len(expert),
                   episode_ids=ids, test_data_cached=False))
        expert.close()
        if recovery is not None:
            recovery.close()


class FeatureCache:
    """Read only the requested batches from disk; do not put the entire cache on GPU."""
    def __init__(self, directory, device):
        self.arrays = {path.stem: np.load(path, mmap_mode='r', allow_pickle=False)
                       for path in directory.glob('*.npy')}
        self.device = device

    def __len__(self):
        return len(self.arrays['state'])

    def tensor(self, name, indices):
        return torch.from_numpy(np.array(self.arrays[name][indices], copy=True)).to(self.device)

    def inputs(self, indices):
        history = self.arrays['history'][indices]
        return (self.tensor('tokens', history), self.tensor('geometry', history),
                self.tensor('pose', history), self.tensor('ages', indices),
                self.tensor('mask', indices), self.tensor('state', indices), self.tensor('tcp', indices))


def sampling_weights(routes, sources):
    """Match the experiment's 65:35 expert/recovery mix and 20:40:40 routes."""
    weights = np.zeros(len(routes), dtype=np.float64)
    for source, fraction in enumerate((.65, .35)):
        for route, probability in enumerate((.2, .4, .4)):
            selected = (sources == source) & (routes == route)
            if not selected.any():
                raise ValueError('Each expert/recovery source must contain all three routes')
            weights[selected] = fraction * probability / selected.sum()
    if not np.all(weights > 0):
        raise ValueError('Unknown source or route label')
    return torch.from_numpy(weights)


@torch.inference_mode()
def validation_rmse(head, data, batch_size):
    head.eval()
    squared, count = 0., 0
    for start in range(0, len(data), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data)))
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=data.device.type == 'cuda'):
            prediction, _ = head(*data.inputs(indices))
        squared += float((prediction - data.tensor('waypoint', indices)).square().sum())
        count += prediction.numel()
    if count == 0:
        raise ValueError('Validation cache is empty')
    return (squared / count) ** .5


def fit_head(head, training, validation, options, output, expert_samples):
    """Select the head by validation waypoint RMSE, including its initial weights."""
    head.requires_grad_(True)
    weights = sampling_weights(training.arrays['route'], training.arrays['source'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, options['epochs'])
    generator = torch.Generator().manual_seed(options['seed'])
    batch_size = options['batch_size']
    best = validation_rmse(head, validation, batch_size)
    selected = 0
    records = [dict(epoch=0, validation_rmse_m=best)]
    torch.save(head.state_dict(), output / 'head.pt')
    for epoch in range(1, options['epochs'] + 1):
        started = time.monotonic()
        order = torch.multinomial(weights, expert_samples, replacement=True, generator=generator).numpy()
        head.train()
        total = 0.
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(training.device.type, dtype=torch.bfloat16, enabled=training.device.type == 'cuda'):
                prediction, auxiliary = head(*training.inputs(indices))
                loss = F.huber_loss(prediction, training.tensor('waypoint', indices), delta=.02)
                loss = loss + .001 * geometry_nll(auxiliary, training.tensor('teacher', indices),
                                                training.tensor('geometry_valid', indices))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise RuntimeError('Missing or non-finite compact-head gradients')
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        score = validation_rmse(head, validation, batch_size)
        if score < best:
            best, selected = score, epoch
            torch.save(head.state_dict(), output / 'head.pt')
        scheduler.step()
        record = dict(epoch=epoch, validation_rmse_m=score, loss=total / len(order),
                      sampling_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                      seconds=time.monotonic() - started)
        records.append(record)
        write_json(output / 'epochs.json', records)
        print(record, flush=True)
    head.load_state_dict(torch.load(output / 'head.pt', map_location=training.device, weights_only=True))
    return dict(selected_epoch=selected, best_validation_rmse_m=best, completed_epochs=options['epochs'])


def train(checkpoint_path, output, options):
    device = require_device(options['device'])
    seed_everything(options['seed'])
    model, maps, checkpoint = load_clearance_policy(checkpoint_path, device, allow_training_source=True)
    config, splits = copy.deepcopy(checkpoint['config']), checkpoint['splits']
    recoveries = config['train'].get('recovery_sources', [])
    if len(recoveries) != 1:
        raise ValueError('Compact training requires one existing recovery source')
    config['train'] = {**options, 'recovery_sources': recoveries,
                       'checkpoint_selection': 'validation_waypoint_rmse', 'frozen_backbone': True}
    create_output(output)
    write_json(output / 'config.json', config)
    write_json(output / 'splits.json', splits)
    write_json(output / 'protocol.json', dict(initial_checkpoint=str(checkpoint_path.resolve()),
               variant='single_memory', frozen_backbone=True, test_used_for_selection=False))
    cache_features(output, model, maps, config, splits, options, device)
    model.backbone.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    training = FeatureCache(output / 'cache/train', device)
    validation = FeatureCache(output / 'cache/validation', device)
    samples = read_json(output / 'cache/train/metadata.json')['expert_samples']
    summary = fit_head(model.head, training, validation, options, output, samples)
    save_checkpoint(output / 'best.pt', dict(format_version=1, architecture='compact_consensus',
                    variant='single_memory', config=config, splits=splits,
                    backbone=model.backbone.state_dict(), head=model.head.cpu().state_dict(), **summary))
    write_json(output / 'summary.json', {**summary, 'variant': 'single_memory', 'frozen_backbone': True})
    write_json(output / 'status.json', dict(state='complete', **summary))
