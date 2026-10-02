"""Cache frozen predicted/teacher maps from training demonstrations and recovery."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import experiment_path, write_json
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.loaders import make_loader
from tsn.data.recovery_dataset import RecoveryDataset
from tsn.data.splits import episode_catalog
from tsn.evaluation.open_loop import device_batch
from tsn.features.state import policy_state
from tsn.models.factory import make_maps, make_policy
from validate_pi3_maps import allocate_cache


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('/run/user/1016/experiments/pi3_small_perturbation10_20261002/train/best.pt'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = experiment_path(args.output_dir)
    torch.set_num_threads(1)
    started = time.monotonic()
    checkpoint = load_checkpoint(args.checkpoint)
    cfg, splits = checkpoint['config'], checkpoint['splits']
    catalog = episode_catalog(Path(cfg['benchmark']['root']))
    expert = FrameDataset(Path(cfg['benchmark']['root']), splits['train'], catalog, 30,
                          frame_stride=cfg['train']['frame_stride'], include_rgb=True)
    recovery = RecoveryDataset(Path(cfg['train']['recovery_sources'][0]['path']), 30, include_rgb=True)
    assert set(recovery.route_by_episode) <= set(splits['train'])
    assert not set(splits['train']) & (set(splits['validation']) | set(splits['test']))
    dataset = ConcatDataset([expert, recovery])
    model = make_policy(cfg['model'], initialize_backbone=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    torch.save(model.action_policy.state_dict(), root / 'action_head_initial.pt')
    write_json(root / 'head_config.json', cfg['model'])
    write_json(root / 'training_config.json', cfg['train'])
    del checkpoint
    model.cuda().eval()
    maps = make_maps(cfg['model']).cuda()
    parallel = torch.nn.DataParallel(model, device_ids=list(range(min(4, torch.cuda.device_count()))))
    cache = allocate_cache(root / 'training_cache', len(dataset), maps.settings.height, maps.settings.width)
    source = np.lib.format.open_memmap(root / 'training_cache/source.npy', mode='w+', dtype=np.int64, shape=(len(dataset),))
    source[:] = np.arange(len(dataset)) >= len(expert)
    source.flush()
    ids = splits['train']
    episode_lookup = {episode: index for index, episode in enumerate(ids)}
    loader = make_loader(dataset, {'device': 'cuda', 'batch_size': 128, 'num_workers': 8}, False, 20261002)
    offset = 0
    for index, raw in enumerate(loader):
        batch = device_batch(raw, torch.device('cuda:0'))
        state = policy_state(batch['qpos'], batch['goal_pose'], maps.settings)
        _, predicted = parallel(batch['rgb'], state, batch['K'], batch['T_B_C'], return_maps=True)
        teacher = maps(batch['depth'], batch['K'], batch['T_B_C'], batch['goal_pose'][:, :3], batch['future_ee'], batch['valid_future'])
        for key, value in [('predicted', predicted), ('teacher', teacher), ('state', state),
                           ('target', batch['target']), ('valid', batch['valid_future']), ('route', batch['route'])]:
            cache[key][offset:offset+len(state)] = value.float().cpu().numpy() if value.is_floating_point() else value.cpu().numpy()
        cache['episode'][offset:offset+len(state)] = [episode_lookup[x.removesuffix('_recovery')] for x in raw['episode_id']]
        offset += len(state)
        if index % 20 == 0 or offset == len(dataset):
            print(f'Training cache {offset}/{len(dataset)}; elapsed {time.monotonic()-started:.1f}s', flush=True)
            write_json(root / 'training_cache_status.json', {'state': 'running', 'frames': offset})
    assert offset == len(dataset)
    for value in cache.values():
        value.flush()
    write_json(root / 'training_cache/metadata.json', {'expert_samples': len(expert), 'recovery_samples': len(recovery),
               'samples': len(dataset), 'frame_stride': cfg['train']['frame_stride'], 'episode_ids': ids,
               'held_out_episode_overlap': 0, 'frozen_encoder_and_map_heads': True,
               'cache_dtype': 'float16 maps, float32 state/actions', 'elapsed_seconds': time.monotonic()-started})
    write_json(root / 'training_cache_status.json', {'state': 'complete', 'frames': offset})
    expert.close()
    recovery.close()


if __name__ == '__main__':
    main()
