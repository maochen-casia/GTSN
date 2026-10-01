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

    def __init__(self, root: Path, chunk_size: int) -> None:
        self.root = root
        self.chunk_size = int(chunk_size)
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
        return {
            "depth": torch.from_numpy(data["depth"][sample_index].astype(np.float32)),
            "T_B_C": torch.from_numpy(data["T_B_C"][sample_index].astype(np.float32)),
            "K": torch.from_numpy(data["K"][sample_index].astype(np.float32)),
            "future_ee": torch.from_numpy(data["future_ee"][sample_index].astype(np.float32)),
            "qpos": torch.from_numpy(data["qpos"][sample_index].astype(np.float32)),
            "goal_pose": torch.from_numpy(data["goal_pose"][sample_index].astype(np.float32)),
            "target": torch.from_numpy(data["target"][sample_index].astype(np.float32)),
            "valid_future": torch.from_numpy(data["valid_future"][sample_index].astype(bool)),
            "route": self.route_indices[index],
            "episode_id": self.episode_ids[index] + "_recovery",
            "frame_index": int(data["frame_index"][sample_index]),
        }

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
