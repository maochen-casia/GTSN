"""Generate safe joint-perturbed observations for training recovery behavior."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tsn.common.config import read_json, write_json
from tsn.data.hdf5_dataset import padded_future, validate_episode
from tsn.data.splits import ROUTES, episode_catalog, make_splits
from tsn.simulation.episode import EpisodeSimulation


def _aabb_distance(a: np.ndarray, b: np.ndarray) -> float:
    gap = np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0.0)
    return float(np.linalg.norm(gap))


def _obstacle_aabbs(scene: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    result = []
    for item in scene["objects"]:
        center = np.asarray(item["center"], dtype=np.float64)
        half = np.asarray(item["half_size"], dtype=np.float64) + 0.022
        if item["kind"] == "mug":
            half[:2] = np.maximum(half[:2], item["half_size"][0] * 1.98)
        yaw = float(item["yaw"])
        cosine, sine = abs(np.cos(yaw)), abs(np.sin(yaw))
        world_half = np.asarray([
            cosine * half[0] + sine * half[1],
            sine * half[0] + cosine * half[1],
            half[2],
        ])
        result.append((item["name"], np.stack((center - world_half, center + world_half))))
    return result


def _minimum_clearance(simulation: EpisodeSimulation, scene: dict[str, Any],
                       options: dict[str, Any]) -> tuple[float, str]:
    table_top = float(simulation.options["table_center_m"][2] + simulation.options["table_half_size_m"][2])
    obstacles = _obstacle_aabbs(scene)
    minimum, closest = float("inf"), "none"
    for name, link in simulation.links.items():
        if name == "panda_link0" or not link.get_collision_shapes():
            continue
        aabb = np.asarray(link.compute_global_aabb_tight(), dtype=np.float64)
        table_gap = float(aabb[0, 2] - table_top)
        if table_gap < minimum:
            minimum, closest = table_gap, f"{name}<->table"
        for obstacle_name, obstacle_aabb in obstacles:
            gap = _aabb_distance(aabb, obstacle_aabb)
            if gap < minimum:
                minimum, closest = gap, f"{name}<->{obstacle_name}"
    return minimum, closest


def _set_simulation_state(simulation: EpisodeSimulation, qpos: np.ndarray) -> None:
    simulation.robot.set_qpos(np.asarray(qpos, dtype=np.float32))
    simulation.robot.set_qvel(np.zeros_like(qpos, dtype=np.float32))
    for joint, value in zip(simulation.joints, qpos):
        joint.set_drive_target(float(value))
        joint.set_drive_velocity_target(0.0)


def _pad_future(values: np.ndarray, frame: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    future, valid = padded_future(values, frame, horizon)
    return future.astype(np.float32), valid


def _generate_episode(job: dict[str, Any]) -> dict[str, Any]:
    episode = job["episode"]
    root, output = Path(job["dataset_root"]), Path(job["output_dir"])
    path = root / episode / "episode.h5"
    with h5py.File(path, "r") as handle:
        validate_episode(handle, job["route"])
        q_reference = np.asarray(handle["qpos"][:], dtype=np.float32)
        ee_reference = np.asarray(handle["ee_pose"][:], dtype=np.float32)
        calibration = np.asarray(handle["intrinsics"][:], dtype=np.float32)
        camera_extrinsic = np.asarray(handle["T_ee_camera_cv"][:], dtype=np.float64)
        goal = np.asarray(handle["goal_pose_xyz_wxyz"][:], dtype=np.float32)
        names = tuple(value.decode() if isinstance(value, bytes) else str(value)
                      for value in handle["joint_names"][:])
        source_hw = tuple(handle["depth_m"].shape[1:])

    # Use fixed, evenly spaced locations and a per-episode random stream so generation is repeatable.
    frame_indices = np.linspace(
        max(1, round(0.12 * (len(q_reference) - 1))),
        max(1, round(0.88 * (len(q_reference) - 1))),
        int(job["samples_per_episode"]), dtype=int,
    )
    seed = int.from_bytes(hashlib.sha256(episode.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    scene_description = read_json(root / episode / "scene.json")
    accepted: list[dict[str, np.ndarray | int]] = []
    with EpisodeSimulation(
        scene_description, calibration, camera_extrinsic, source_hw,
        q_reference[0], ee_reference[0], names, job["eval_options"],
    ) as simulation:
        limits = simulation.qlimits
        for frame_index in frame_indices:
            for _ in range(int(job["max_attempts"])):
                noise = np.clip(
                    rng.normal(0.0, float(job["noise_std_rad"]), 7),
                    -float(job["noise_max_rad"]), float(job["noise_max_rad"]),
                ).astype(np.float32)
                candidate = q_reference[frame_index].copy()
                candidate[:7] += noise
                if np.any(candidate[:7] < limits[:, 0]) or np.any(candidate[:7] > limits[:, 1]):
                    continue
                _set_simulation_state(simulation, candidate)
                # Match the reference generator: settle one physics step before checking contacts.
                simulation.scene.step()
                if simulation._contacts():
                    continue
                clearance, _ = _minimum_clearance(simulation, scene_description, job["eval_options"])
                if clearance < float(job["minimum_clearance_m"]):
                    continue

                state = simulation.snapshot()
                depth = simulation.render()[1]
                future_q, valid = _pad_future(q_reference[:, :7], int(frame_index), int(job["chunk_size"]))
                future_ee, _ = _pad_future(ee_reference[:, :3], int(frame_index), int(job["chunk_size"]))
                current_q = state["qpos"]
                accepted.append({
                    "depth": depth.astype(np.float16),
                    "T_B_C": state["T_B_C"].astype(np.float32),
                    "K": calibration,
                    "future_ee": future_ee,
                    "qpos": current_q,
                    "goal_pose": goal,
                    "target": future_q - current_q[None, :7],
                    "valid_future": valid,
                    "frame_index": int(frame_index),
                    "clearance": np.float32(clearance),
                })
                break

    if not accepted:
        # A tightened, faithful collision envelope can reject every attempted
        # perturbation in a narrow passage. Record rejection rather than weaken
        # clearance requirements or discard the entire generation job.
        (output / f"{episode}.npz").unlink(missing_ok=True)
        return {"episode": episode, "route": job["route"], "requested": len(frame_indices),
                "accepted": 0, "minimum_clearance_m": None}
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{episode}.npz"
    temporary = output / f"{episode}.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            depth=np.stack([item["depth"] for item in accepted]),
            T_B_C=np.stack([item["T_B_C"] for item in accepted]),
            K=np.repeat(calibration[None], len(accepted), axis=0),
            future_ee=np.stack([item["future_ee"] for item in accepted]),
            qpos=np.stack([item["qpos"] for item in accepted]),
            goal_pose=np.stack([item["goal_pose"] for item in accepted]),
            target=np.stack([item["target"] for item in accepted]),
            valid_future=np.stack([item["valid_future"] for item in accepted]),
            frame_index=np.asarray([item["frame_index"] for item in accepted], dtype=np.int32),
            clearance_m=np.asarray([item["clearance"] for item in accepted], dtype=np.float32),
            route=np.asarray(job["route"]), episode_id=np.asarray(episode),
        )
    temporary.replace(archive)
    return {
        "episode": episode, "route": job["route"], "requested": len(frame_indices),
        "accepted": len(accepted),
        "minimum_clearance_m": float(min(item["clearance"] for item in accepted)),
    }


def generate_recovery(benchmark: dict[str, Any], eval_options: dict[str, Any], output: Path,
                      chunk_size: int, samples_per_episode: int = 4,
                      noise_std_rad: float = 0.035, noise_max_rad: float = 0.10,
                      minimum_clearance_m: float = 0.002, max_attempts: int = 30,
                      workers: int = 4, selected_episodes: list[str] | None = None) -> dict[str, Any]:
    """Generate deterministic recovery data from the fixed training episode split."""
    if samples_per_episode <= 0 or max_attempts <= 0 or workers <= 0 or chunk_size <= 0:
        raise ValueError("Sample, attempt, worker, and chunk counts must be positive")
    root = Path(benchmark["root"])
    catalog = episode_catalog(root)
    split = make_splits(benchmark)
    train_ids = split["train"]
    if selected_episodes is not None:
        if not selected_episodes or set(selected_episodes) - set(train_ids):
            raise ValueError("Selected recovery episodes must be members of the training split")
        selected = set(selected_episodes)
        train_ids = [episode for episode in train_ids if episode in selected]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "source_split.json", split)
    jobs = [{
        "episode": episode, "route": catalog[episode], "dataset_root": str(root),
        "output_dir": str(output), "eval_options": eval_options, "chunk_size": chunk_size,
        "samples_per_episode": samples_per_episode, "noise_std_rad": noise_std_rad,
        "noise_max_rad": noise_max_rad, "minimum_clearance_m": minimum_clearance_m,
        "max_attempts": max_attempts,
    } for episode in train_ids]
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_generate_episode, job): job["episode"] for job in jobs}
        for done_count, future in enumerate(as_completed(pending), start=1):
            result = future.result()
            results.append(result)
            if done_count % 10 == 0 or done_count == len(jobs):
                accepted = sum(value["accepted"] for value in results)
                print(f"[{done_count}/{len(jobs)}] recovery samples accepted: {accepted}", flush=True)
    results.sort(key=lambda value: value["episode"])
    counts = {route: sum(value["accepted"] for value in results if value["route"] == route)
              for route in ROUTES}
    report = {
        "schema_version": "tsn-1k-recovery-v1", "source_dataset": str(root),
        "source_partition": "train", "episode_ids": [value["episode"] for value in results if value["accepted"]],
        "requested_episode_ids": train_ids,
        "parameters": {
            "samples_per_episode": samples_per_episode, "noise_std_rad": noise_std_rad,
            "noise_max_rad": noise_max_rad, "minimum_clearance_m": minimum_clearance_m,
            "max_attempts_per_frame": max_attempts, "chunk_size": chunk_size,
        },
        "episodes": sum(value["accepted"] > 0 for value in results),
        "rejected_episodes": [value["episode"] for value in results if not value["accepted"]],
        "samples": sum(counts.values()), "samples_by_route": counts,
        "route_episode_counts": {route: sum(value["route"] == route and value["accepted"] > 0 for value in results) for route in ROUTES},
        "minimum_clearance_m": min((value["minimum_clearance_m"] for value in results if value["accepted"]), default=None),
        "wall_seconds": time.perf_counter() - start, "results": results,
    }
    write_json(output / "manifest.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2), flush=True)
    return report
