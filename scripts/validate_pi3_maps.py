"""Open-loop validation with map interventions, cached features and paired CIs.

Run inside gtsn-pi3:20261002. This diagnostic intentionally uses privileged
teacher maps as counterfactual inputs; it does not change deployed inference.
"""
import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import experiment_path, write_json
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.loaders import make_loader
from tsn.data.splits import ROUTES, episode_catalog, validate_splits
from tsn.evaluation.map_diagnostics import GROUPS, MapQuality, episode_bootstrap, oracle_variants, replace_groups
from tsn.evaluation.metrics import PredictionMetrics
from tsn.evaluation.open_loop import device_batch
from tsn.features.state import policy_state
from tsn.models.factory import make_maps, make_policy


def allocate_cache(directory, count, height, width):
    directory.mkdir(parents=True, exist_ok=True)
    shapes = {'predicted': ((count, 6, height, width), np.float16),
              'teacher': ((count, 6, height, width), np.float16),
              'state': ((count, 16), np.float32), 'target': ((count, 30, 7), np.float32),
              'valid': ((count, 30), np.bool_), 'route': ((count,), np.int64),
              'episode': ((count,), np.int64)}
    return {key: np.lib.format.open_memmap(directory / f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
            for key, (shape, dtype) in shapes.items()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('/run/user/1016/experiments/pi3_small_perturbation10_20261002/train/best.pt'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    root = experiment_path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    checkpoint = load_checkpoint(args.checkpoint)
    cfg, splits = checkpoint['config'], checkpoint['splits']
    catalog = episode_catalog(Path(cfg['benchmark']['root']))
    validate_splits(splits, catalog)
    ids = splits['validation']
    assert not set(ids) & (set(splits['train']) | set(splits['test']))
    device = torch.device('cuda:0')
    model = make_policy(cfg['model'], initialize_backbone=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(device).eval()
    maps = make_maps(cfg['model']).to(device)
    epoch = checkpoint['epoch']
    del checkpoint
    parallel = torch.nn.DataParallel(model, device_ids=list(range(min(4, torch.cuda.device_count()))))
    options = {'device': 'cuda', 'num_workers': 8, 'batch_size': args.batch_size}
    dataset = FrameDataset(Path(cfg['benchmark']['root']), ids, catalog, 30, include_rgb=True)
    loader = make_loader(dataset, options, training=False, seed=20261002)
    cache = allocate_cache(root / 'validation_cache', len(dataset), maps.settings.height, maps.settings.width)
    variants = oracle_variants()
    variants.update({f'zero_{group}': (group,) for group in GROUPS})
    variants['zero_all'] = tuple(GROUPS)
    metrics = {name: PredictionMetrics(30) for name in variants}
    squared = {name: np.zeros((len(ids), 2), dtype=np.float64) for name in variants}
    counts = np.zeros((len(ids), 2), dtype=np.float64)
    episode_routes = np.array([ROUTES.index(catalog[episode]) for episode in ids])
    quality = MapQuality(maps.settings, device)
    write_json(root / 'protocol.json', {'checkpoint': str(args.checkpoint), 'checkpoint_epoch': epoch,
               'partition': 'validation', 'episode_ids': ids, 'episodes': len(ids), 'frames': len(dataset),
               'routes': dict(Counter(catalog[x] for x in ids)), 'frame_stride': 1, 'batch_size': args.batch_size,
               'teacher_map_interventions_are_privileged': True, 'policy_parameters_frozen': True,
               'variant_groups': {key: list(value) for key, value in variants.items()},
               'bootstrap': '5000 paired route-stratified episode resamples; seed 20261002',
               'source_sha256': {str(path.relative_to(Path('/workspace'))): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), Path('/workspace/src/tsn/evaluation/map_diagnostics.py')]}})
    offset = 0
    for index, raw in enumerate(loader):
        batch = device_batch(raw, device)
        state = policy_state(batch['qpos'], batch['goal_pose'], maps.settings)
        actions, predicted = parallel(batch['rgb'], state, batch['K'], batch['T_B_C'], return_maps=True)
        teacher = maps(batch['depth'], batch['K'], batch['T_B_C'], batch['goal_pose'][:, :3],
                       batch['future_ee'], batch['valid_future'])
        quality.update(predicted, teacher, batch['depth'])
        episode = np.array([ids.index(x) for x in raw['episode_id']])
        for key, value in [('predicted', predicted), ('teacher', teacher), ('state', state),
                           ('target', batch['target']), ('valid', batch['valid_future']), ('route', batch['route'])]:
            cache[key][offset:offset+len(state)] = value.float().cpu().numpy() if value.is_floating_point() else value.cpu().numpy()
        cache['episode'][offset:offset+len(state)] = episode
        n = batch['valid_future'].sum(1).cpu().numpy() * 7
        n15 = batch['valid_future'][:, :15].sum(1).cpu().numpy() * 7
        np.add.at(counts[:, 0], episode, n)
        np.add.at(counts[:, 1], episode, n15)
        for name, groups in variants.items():
            if name == 'predicted':
                prediction = actions
            else:
                replacement = replace_groups(predicted.float(), teacher, groups, zero=name.startswith('zero_'))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model.action_policy(replacement, state)
            metrics[name].update(prediction, batch['target'], batch['valid_future'], batch['route'])
            error = (prediction.float() - batch['target']).square().sum(-1) * batch['valid_future']
            values = torch.stack([error.sum(1), error[:, :15].sum(1)], 1).cpu().numpy()
            np.add.at(squared[name], episode, values)
        offset += len(state)
        if index % 10 == 0 or offset == len(dataset):
            print(f'Validation {offset}/{len(dataset)} frames; elapsed {time.monotonic()-started:.1f}s', flush=True)
            write_json(root / 'status.json', {'state': 'running', 'stage': 'validation_interventions', 'frames': offset})
    assert offset == len(dataset)
    for value in cache.values():
        value.flush()
    baseline = metrics['predicted'].result()['rmse_rad']
    results = {}
    for name, metric in metrics.items():
        result = metric.result()
        result.update(relative_rmse_reduction=1-result['rmse_rad']/baseline,
                      **episode_bootstrap(squared[name][:, 0], counts[:, 0], squared['predicted'][:, 0], episode_routes))
        result['executed_first15'] = {'rmse_rad': float(np.sqrt(squared[name][:, 1].sum()/counts[:, 1].sum())),
            **episode_bootstrap(squared[name][:, 1], counts[:, 1], squared['predicted'][:, 1], episode_routes)}
        results[name] = result
    write_json(root / 'results.json', {'partition': 'validation', 'episodes': len(ids), 'frames': len(dataset),
               'checkpoint_epoch': epoch, 'interventions': results, 'map_quality': quality.result(),
               'elapsed_seconds': time.monotonic()-started})
    np.savez(root / 'paired_episode_errors.npz', counts=counts, episode_routes=episode_routes, **squared)
    dataset.close()
    write_json(root / 'status.json', {'state': 'complete', 'stage': 'validation_interventions'})
    print({name: result['rmse_rad'] for name, result in results.items()}, flush=True)


if __name__ == '__main__':
    main()
