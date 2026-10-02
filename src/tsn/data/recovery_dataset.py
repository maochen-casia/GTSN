"""Lazy access to simulator-rendered recovery observations stored per episode."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tsn.data.splits import ROUTES


class RecoveryDataset(Dataset):
    """Recovery samples parallel FrameDataset's batch schema."""

    def __init__(self, root: Path, chunk_size: int, include_rgb: bool = False) -> None:
        self.root = root
        self.chunk_size = int(chunk_size)
        self.include_rgb = include_rgb
        self.paths = sorted(root.glob("episode_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No recovery archives found in {root}")
        self.samples: list[tuple[int, int]] = []
        self.route_indices: list[int] = []
        self.episode_ids: list[str] = []
        self.route_by_episode: dict[str, str] = {}
        for file_index, path in enumerate(self.paths):
            with np.load(path, allow_pickle=False) as data:
                route = str(data["route"])
                episode = str(data["episode_id"])
                count = len(data["depth"])
                if include_rgb and ('rgb' not in data or data['rgb'].shape != (*data['depth'].shape, 3)
                                    or data['rgb'].dtype != np.uint8):
                    raise ValueError(f'{path}: missing or invalid recovery RGB observations')
                if route not in ROUTES or count == 0:
                    raise ValueError(f"Invalid recovery archive: {path}")
                expected = {
                    "T_B_C": (count, 4, 4), "K": (count, 3, 3),
                    "future_ee": (count, self.chunk_size, 3), "qpos": (count, 9),
                    "goal_pose": (count, 7), "target": (count, self.chunk_size, 7),
                    "valid_future": (count, self.chunk_size),
                }
                for key, shape in expected.items():
                    if key not in data or data[key].shape != shape:
                        actual = data[key].shape if key in data else "missing"
                        raise ValueError(f"{path}: {key} has shape {actual}; expected {shape}")
                if data["depth"].ndim != 3 or len(data["depth"]) != count:
                    raise ValueError(f"{path}: invalid depth shape")
                # Legacy archives masked known endpoint holds. Admit only prefix
                # masks whose padded targets repeat the final known target.
                for sample in range(count):
                    valid = data["valid_future"][sample].astype(bool)
                    size = int(valid.sum())
                    if not np.array_equal(valid, np.arange(self.chunk_size) < size):
                        raise ValueError(f"{path}: future mask is not a contiguous prefix")
                    if size < self.chunk_size:
                        anchor = max(0, size - 1)
                        for key in ("target", "future_ee"):
                            if not np.allclose(data[key][sample, size:], data[key][sample, anchor], atol=1e-6):
                                raise ValueError(f"{path}: unavailable future observations are not terminal holds")
            self.samples.extend((file_index, index) for index in range(count))
            self.route_indices.extend([ROUTES.index(route)] * count)
            self.episode_ids.extend([episode] * count)
            if episode in self.route_by_episode and self.route_by_episode[episode] != route:
                raise ValueError(f"Conflicting route labels for {episode}")
            self.route_by_episode[episode] = route
        self._files: OrderedDict[int, Any] = OrderedDict()
        self._pid = os.getpid()

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self, file_index: int) -> Any:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if file_index not in self._files:
            self._files[file_index] = np.load(self.paths[file_index], allow_pickle=False)
        self._files.move_to_end(file_index)
        while len(self._files) > 8:
            _, stale = self._files.popitem(last=False)
            stale.close()
        return self._files[file_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_index, sample_index = self.samples[index]
        data = self._file(file_index)
        sample = {
            "depth": torch.from_numpy(data["depth"][sample_index].astype(np.float32)),
            "T_B_C": torch.from_numpy(data["T_B_C"][sample_index].astype(np.float32)),
            "K": torch.from_numpy(data["K"][sample_index].astype(np.float32)),
            "future_ee": torch.from_numpy(data["future_ee"][sample_index].astype(np.float32)),
            "qpos": torch.from_numpy(data["qpos"][sample_index].astype(np.float32)),
            "goal_pose": torch.from_numpy(data["goal_pose"][sample_index].astype(np.float32)),
            "target": torch.from_numpy(data["target"][sample_index].astype(np.float32)),
            "valid_future": torch.ones(self.chunk_size, dtype=torch.bool),
            "route": self.route_indices[index],
            "episode_id": self.episode_ids[index] + "_recovery",
            "frame_index": int(data["frame_index"][sample_index]),
        }
        if self.include_rgb:
            sample['rgb'] = torch.from_numpy(data['rgb'][sample_index].copy())
        return sample

    def close(self) -> None:
        for value in self._files.values():
            value.close()
        self._files.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = OrderedDict()
        state["_pid"] = os.getpid()
        return state

    def __del__(self) -> None:
        if hasattr(self, "_files"):
            self.close()
