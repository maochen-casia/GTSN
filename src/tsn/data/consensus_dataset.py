"""Raw causal observations for fresh end-to-end consensus training."""
from bisect import bisect_right

import numpy as np
import torch
from torch.utils.data import Dataset


class ConsensusDataset(Dataset):
    def __init__(self, expert, recovery=None, history_length=4, spacing=15):
        self.expert, self.recovery = expert, recovery
        self.history_length, self.spacing = history_length, spacing

    def __len__(self):
        return len(self.expert) + (len(self.recovery) if self.recovery is not None else 0)

    def __getitem__(self, index):
        if index < len(self.expert):
            sample = self.expert[index]
            episode_index = bisect_right(self.expert.offsets, index)-1
            frame = sample['frame_index']
            # Match the original cache's latest available frame at/before t-15k.
            requested = [frame-self.spacing*k for k in reversed(range(self.history_length))]
            valid = [f >= 0 for f in requested]
            frames = [(f//self.expert.stride)*self.expert.stride if f >= 0 else frame for f in requested]
            handle = self.expert._handle(self.expert.ids[episode_index])
            rgb = torch.from_numpy(np.stack([handle['rgb'][f] for f in frames]).astype(np.uint8))
            qpos = torch.from_numpy(np.stack([handle['qpos'][f] for f in frames]).astype(np.float32))
            pose = torch.from_numpy(np.stack([handle['T_base_camera_cv'][f] for f in frames]).astype(np.float32))
            ages = torch.tensor([frame-f if v else 0 for f, v in zip(frames, valid)], dtype=torch.float32)
        else:
            sample = self.recovery[index-len(self.expert)]
            valid = [False]*(self.history_length-1)+[True]
            rgb = sample['rgb'][None].expand(self.history_length, -1, -1, -1)
            qpos = sample['qpos'][None].expand(self.history_length, -1)
            pose = sample['T_B_C'][None].expand(self.history_length, -1, -1)
            ages = torch.zeros(self.history_length)
        return {**sample, 'history_rgb': rgb, 'history_qpos': qpos, 'history_pose': pose,
                'history_ages': ages, 'history_mask': torch.tensor(valid, dtype=torch.bool),
                'sample_index': index}
