"""Cache causal two-frame inputs once; use unchanged TSN expert/recovery labels."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tsn.baselines.observations import observation_state, point_cloud, rgb_image
from tsn.common.config import read_json, write_json
from tsn.data.navigation import NavigationDataset
from tsn.data.splits import episode_catalog, make_splits


def cache_signature(config):
    fields = {'benchmark': config['benchmark'], 'train': {key: config['train'][key] for key in
        ('frame_stride', 'validation_frame_stride', 'observation_hw', 'recovery_root')},
        'maps': config['model']['maps'], 'schema': 'tsn-baselines-two-frame-v1'}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def prepare_cache(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits = make_splits(config['benchmark'])
    root = Path(config['benchmark']['root'])
    catalog = episode_catalog(root)
    reference = read_json(Path(config['train']['recovery_root'])/'source_split.json')
    if splits != reference:
        raise ValueError('Baseline partitions must match the existing recovery/main partitions')
    write_json(output/'splits.json', splits)
    metadata = {}
    for partition in ('train', 'validation'):
        training = partition == 'train'
        dataset = NavigationDataset(root, splits[partition], catalog,
            config['train']['frame_stride' if training else 'validation_frame_stride'],
            config['train']['recovery_root'] if training else None, config['train']['observation_hw'],
            history_length=2, include_depth_history=True)
        directory = output/partition
        directory.mkdir()
        n = len(dataset)
        shapes = {'rgb': (n, 2, 84, 112, 3), 'points': (n, 2, 512, 3), 'state': (n, 2, 34),
                  'target': (n, 30, 7), 'qpos': (n, 9)}
        arrays = {key: np.lib.format.open_memmap(directory/f'{key}.npy', mode='w+',
            dtype=np.uint8 if key == 'rgb' else np.float32, shape=shape) for key, shape in shapes.items()}
        loader = DataLoader(dataset, batch_size=32, num_workers=config['train']['num_workers'], shuffle=False)
        offset = 0
        for batch in loader:
            b = len(batch['qpos'])
            K, pose = batch['history_K'].flatten(0, 1), batch['history_T_B_C'].flatten(0, 1)
            hw = batch['history_depth'].shape[-2:]
            processed = {'rgb': rgb_image(batch['history_rgb'].flatten(0, 1)).reshape(b, 2, 84, 112, 3),
                'points': point_cloud(batch['history_depth'].flatten(0, 1), K, pose).reshape(b, 2, 512, 3),
                'state': observation_state(batch['history_qpos'].flatten(0, 1), batch['history_goal_pose'].flatten(0, 1),
                     K, pose, hw, config['model']['maps']).reshape(b, 2, 34),
                'target': batch['target'], 'qpos': batch['qpos']}
            for key, value in processed.items():
                arrays[key][offset:offset+b] = value.numpy()
            offset += b
            if offset % 1024 < b or offset == n:
                print({'cache_partition': partition, 'samples': offset, 'total': n}, flush=True)
        if offset != n:
            raise RuntimeError('Incomplete cache')
        for array in arrays.values():
            array.flush()
        np.save(directory/'route.npy', dataset.route)
        np.save(directory/'source.npy', dataset.source)
        metadata[partition] = {'samples': n, 'expert_samples': len(dataset.expert),
                               'perturbation_samples': len(dataset.recovery) if dataset.recovery else 0}
        if training:
            targets = arrays['target'].reshape(-1, 7)
            lo, hi = targets.min(0), targets.max(0)
            write_json(output/'normalization.json', {'center': ((lo+hi)/2).tolist(),
                'scale': np.maximum((hi-lo)/2, .01).tolist(), 'fit_partition': 'train', 'method': 'minmax'})
        dataset.close()
        del arrays
    manifest = read_json(Path(config['train']['recovery_root'])/'manifest.json')
    write_json(output/'complete.json', {'signature': cache_signature(config), 'partitions': metadata,
        'history_slots': 2, 'recovery_partition': manifest['source_partition'],
        'recovery_episodes': len(manifest['episode_ids']), 'recovery_collection': manifest['collection'],
        'test_cached': False, 'normalization_fit_partition': 'train'})


class CachedDataset(Dataset):
    def __init__(self, cache, partition, modality):
        directory = Path(cache)/partition
        keys = ['state', 'target', 'qpos', 'points' if modality == 'points' else 'rgb']
        self.arrays = {key: np.load(directory/f'{key}.npy', mmap_mode='r', allow_pickle=False) for key in keys}
        self.route = np.load(directory/'route.npy', allow_pickle=False)
        self.source = np.load(directory/'source.npy', allow_pickle=False)

    def __len__(self):
        return len(self.route)

    def __getitem__(self, index):
        return {key: torch.from_numpy(np.array(value[index], copy=True)) for key, value in self.arrays.items()}
