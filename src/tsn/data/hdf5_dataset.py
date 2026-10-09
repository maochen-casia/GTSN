"""Read the actual tsn-1k schema; generate maps later on the policy device."""

from __future__ import annotations

import os
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tsn.data.splits import ROUTES
from tsn.features.camera import resize_observation

JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8)) + (
    "panda_finger_joint1", "panda_finger_joint2",
)
FR3_JOINT_NAMES = tuple(name.replace('panda', 'fr3') for name in JOINT_NAMES)


def validate_episode(handle: h5py.File, route: str) -> int:
    """Check schema shapes and joint order before admitting an episode into a loader."""
    count = len(handle["qpos"])
    shapes = {
        "qpos": (count, 9), "ee_pose": (count, 7),
        "T_base_camera_cv": (count, 4, 4), "T_ee_camera_cv": (4, 4),
        "intrinsics": (3, 3), "goal_pose_xyz_wxyz": (7,), "time_seconds": (count,),
    }
    for key, shape in shapes.items():
        if handle[key].shape != shape:
            raise ValueError(f"{handle.filename}: {key} must have shape {shape}")
    if count < 2 or handle["depth_m"].ndim != 3 or len(handle["depth_m"]) != count:
        raise ValueError(f"{handle.filename}: invalid depth or trajectory length")
    names = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in handle["joint_names"][:])
    if names not in (JOINT_NAMES, FR3_JOINT_NAMES) or handle.attrs["route_type"] != route:
        raise ValueError(f"{handle.filename}: joint ordering or route mismatch")
    for key in ("intrinsics", "T_ee_camera_cv", "goal_pose_xyz_wxyz"):
        if not np.isfinite(handle[key][:]).all():
            raise ValueError(f"{handle.filename}: non-finite {key}")
    calibration = handle["intrinsics"][:]
    if calibration[0, 0] <= 0 or calibration[1, 1] <= 0 or not np.allclose(calibration[2], [0, 0, 1]):
        raise ValueError(f"{handle.filename}: invalid camera intrinsics")
    times = handle["time_seconds"][:]
    if not np.allclose(np.diff(times), 0.05, atol=1e-6):
        raise ValueError(f"{handle.filename}: baseline expects 20 Hz demonstrations")
    return count


def padded_future(values: np.ndarray, index: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return future targets including valid holds at the known terminal state."""
    if len(values) == 0 or not 0 <= index < len(values) or horizon <= 0:
        raise ValueError("Future targets require a trajectory, valid index and positive horizon")
    suffix = values[index + 1:index + horizon + 1]
    size = len(suffix)
    if not size:
        suffix = values[-1:]
    padded = np.concatenate((suffix, np.repeat(suffix[-1:], horizon - len(suffix), axis=0)))
    return padded, np.ones(horizon, dtype=bool)


class FrameDataset(Dataset):
    """Single frames with future arm residuals and expert TCP positions.

    Each sample contains depth (H,W), camera transforms (4,4), calibration
    (3,3), qpos (9,), goal (7,), target (chunk,7), future_ee (chunk,3), and
    valid_future (chunk,). Terminal holds are supervised in loss and map generation.
    File handles are opened lazily per process and bounded by an LRU cache.
    """

    def __init__(self, root: Path, ids: list[str], catalog: dict[str, str], chunk_size: int,
                 frame_stride: int = 1, max_open_files: int = 8, include_rgb: bool = False,
                 observation_hw=None) -> None:
        if min(chunk_size, frame_stride, max_open_files) <= 0 or not ids:
            raise ValueError("A nonempty split and positive horizon/stride/cache are required")
        self.root, self.ids, self.catalog = root, list(ids), catalog
        self.chunk_size, self.stride, self.max_open = chunk_size, frame_stride, max_open_files
        self.include_rgb = include_rgb
        self.observation_hw = observation_hw
        self.counts: list[int] = []
        self.offsets = [0]
        for episode in ids:
            with h5py.File(root / episode / "episode.h5", "r") as handle:
                count = validate_episode(handle, catalog[episode])
            self.counts.append(count)
            self.offsets.append(self.offsets[-1] + len(range(0, count - 1, frame_stride)))
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()
        self._pid = os.getpid()

    def __len__(self) -> int:
        return self.offsets[-1]

    def _handle(self, episode: str) -> h5py.File:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if episode not in self._handles:
            self._handles[episode] = h5py.File(self.root / episode / "episode.h5", "r")
        self._handles.move_to_end(episode)
        while len(self._handles) > self.max_open:
            _, stale = self._handles.popitem(last=False)
            stale.close()
        return self._handles[episode]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.offsets, index) - 1
        episode = self.ids[episode_index]
        frame = (index - self.offsets[episode_index]) * self.stride
        handle = self._handle(episode)
        stop = min(frame + self.chunk_size + 1, self.counts[episode_index])
        # Read contiguous slices, not repeated fancy indices (unsupported by h5py).
        q = np.asarray(handle["qpos"][frame:stop], dtype=np.float32)
        ee = np.asarray(handle["ee_pose"][frame:stop, :3], dtype=np.float32)
        future_q, mask = padded_future(q[:, :7], 0, self.chunk_size)
        future_ee, _ = padded_future(ee, 0, self.chunk_size)
        arrays = {
            "depth": handle["depth_m"][frame], "T_B_C": handle["T_base_camera_cv"][frame],
            "K": handle["intrinsics"][:], "qpos": q[0],
            "goal_pose": handle["goal_pose_xyz_wxyz"][:],
            "future_ee": future_ee, "target": future_q - q[0, :7],
        }
        sample = {
            **{key: torch.from_numpy(np.asarray(value, dtype=np.float32)) for key, value in arrays.items()},
            "valid_future": torch.from_numpy(mask), "episode_id": episode,
            "frame_index": frame, "route": ROUTES.index(self.catalog[episode]),
        }
        if self.include_rgb:
            sample['rgb'] = torch.from_numpy(np.asarray(handle['rgb'][frame], dtype=np.uint8))
        return resize_observation(sample, self.observation_hw)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def __del__(self) -> None:
        if hasattr(self, "_handles"):
            self.close()
