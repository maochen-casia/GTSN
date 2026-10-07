"""Causal four-observation inputs for expert and independent perturbation data."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.history import build_history
from tsn.data.recovery_dataset import RecoveryDataset
from tsn.data.splits import ROUTES


class NavigationDataset(Dataset):
    """Perturbations reset history; expert context never crosses an episode."""
    def __init__(self, root, ids, catalog, stride, recovery_root=None):
        self.expert = FrameDataset(Path(root), ids, catalog, 30, frame_stride=stride, include_rgb=True)
        self.recovery = RecoveryDataset(Path(recovery_root), 30, include_rgb=True) if recovery_root else None
        episode, frame, route = [], [], []
        lookup = {name: i for i, name in enumerate(ids)}
        for name, count in zip(ids, self.expert.counts):
            frames = list(range(0, count-1, stride))
            episode.extend([lookup[name]]*len(frames)); frame.extend(frames)
            route.extend([ROUTES.index(catalog[name])]*len(frames))
        source = [0]*len(episode)
        if self.recovery:
            if not set(self.recovery.route_by_episode) <= set(ids):
                raise ValueError('Perturbation data overlaps a held-out partition')
            if any(catalog[name] != label for name, label in self.recovery.route_by_episode.items()):
                raise ValueError('Perturbation route labels disagree with the benchmark')
            for file_index, sample_index in self.recovery.samples:
                values = self.recovery._file(file_index)
                episode.append(lookup[str(values['episode_id'])]); frame.append(int(values['frame_index'][sample_index]))
            route.extend(self.recovery.route_indices); source.extend([1]*len(self.recovery))
        self.data = ConcatDataset([self.expert, self.recovery]) if self.recovery else self.expert
        self.route, self.source = np.asarray(route), np.asarray(source)
        self.history, self.ages, self.mask = build_history(np.asarray(episode), np.asarray(frame), self.source)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        current = self.data[index]
        context = [self.data[int(i)] if valid and i != index else current
                   for i, valid in zip(self.history[index], self.mask[index])]
        result = {key: value for key, value in current.items() if key != 'rgb'}
        for key in ('rgb', 'qpos', 'goal_pose', 'K', 'T_B_C'):
            result['history_'+key] = torch.stack([row[key] for row in context])
        result['history_ages'] = torch.from_numpy(self.ages[index])
        result['history_mask'] = torch.from_numpy(self.mask[index])
        return result

    def close(self):
        self.expert.close()
        if self.recovery:
            self.recovery.close()


def sampling_weights(routes, sources):
    """65/35 expert/perturbation mixture, with 20/40/40 route sampling."""
    weights = np.zeros(len(routes), dtype=np.float64)
    for source, fraction in enumerate((.65, .35)):
        for route, probability in enumerate((.2, .4, .4)):
            group = (sources == source) & (routes == route)
            if not group.any():
                raise ValueError('Training needs every route in both expert and perturbation sources')
            weights[group] = fraction*probability/group.sum()
    if not np.all(weights > 0):
        raise ValueError('Unknown source or route label')
    return torch.from_numpy(weights)
