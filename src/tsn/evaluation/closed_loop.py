"""Closed-loop validation and testing of RGB route controllers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

from tsn.common.config import read_json, write_json
from tsn.data.hdf5_dataset import validate_episode
from tsn.evaluation.metrics import summarize_rollouts
from tsn.features.maps import GeometryMaps
from tsn.features.state import policy_state
from tsn.models.policy import NavigationPolicy
from tsn.simulation.episode import EpisodeSimulation, orientation_error, quaternion_matrix


@torch.inference_mode()
def rollout(episode: str, route: str, root: Path, output: Path, model: NavigationPolicy,
            maps: GeometryMaps, device: torch.device, options: dict[str, Any]) -> dict[str, Any]:
    """Execute live RGB predictions, reading only the initial expert robot state."""
    execute = int(options["execute_horizon"])
    max_steps = int(options["max_control_steps"])
    stride = int(options["video_stride"])
    if not 1 <= execute <= model.chunk_size or max_steps <= 0 or stride <= 0:
        raise ValueError("Invalid execute horizon, episode length or video stride")
    if min(options["goal_position_tolerance_m"], options["goal_orientation_tolerance_rad"]) <= 0:
        raise ValueError("Goal tolerances must be positive")
    if options["contact_impulse_threshold_ns"] < 0:
        raise ValueError("Contact impulse threshold must be nonnegative")
    directory = output / "episodes" / episode
    directory.mkdir(parents=True, exist_ok=False)
    with h5py.File(root / episode / "episode.h5", "r") as handle:
        validate_episode(handle, route)
        expert_q = np.asarray(handle['qpos'][:1], dtype=np.float32)
        expert_ee = np.asarray(handle['ee_pose'][:1], dtype=np.float32)
        calibration = handle["intrinsics"][:]
        camera_extrinsic = handle["T_ee_camera_cv"][:]
        source_hw = handle["depth_m"].shape[1:]
        goal = np.asarray(handle["goal_pose_xyz_wxyz"][:], dtype=np.float32)
        names = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in handle["joint_names"][:])
    goal_rotation = quaternion_matrix(goal[3:])
    def is_goal(position: float, rotation: float) -> bool:
        return (position <= options["goal_position_tolerance_m"] and
                (options.get("goal_success_criterion", "xyz_orientation") == "xyz" or
                 rotation <= options["goal_orientation_tolerance_rad"]))
    K = torch.as_tensor(calibration, device=device, dtype=torch.float32)[None]
    goal_tensor = torch.as_tensor(goal, device=device)[None]
    first_collision = None
    steps = replans = clips = 0
    reached_goal = False
    inference_seconds = []
    q_history, ee_history, camera_history = [], [], []
    target_history = []
    writer = None
    started = time.perf_counter()
    model.eval()
    if hasattr(model, 'reset_episode'):
        model.reset_episode()
    execution_history = []
    with EpisodeSimulation(read_json(root / episode / "scene.json"), calibration,
                           camera_extrinsic, source_hw, expert_q[0], expert_ee[0], names, options) as simulation:
        try:
            if options["render_videos"]:
                writer = imageio.get_writer(directory / "wrist.mp4", fps=options["control_hz"] / stride,
                                            codec="libx264", macro_block_size=1)
            state = simulation.snapshot()
            if writer is not None:
                writer.append_data(simulation.render()[0])

            def record() -> tuple[float, float]:
                q_history.append(state["qpos"].copy())
                ee_history.append(state["T_B_E"].copy())
                camera_history.append(state["T_B_C"].copy())
                position = float(np.linalg.norm(state["T_B_E"][:3, 3] - goal[:3]))
                rotation = orientation_error(state["T_B_E"][:3, :3], goal_rotation)
                return position, rotation

            position_error, rotation_error = record()
            minimum_position_error = position_error
            reached_goal = is_goal(position_error, rotation_error)
            while steps < max_steps and not reached_goal:
                rgb, _ = simulation.render()
                anchor = state["qpos"][:7].copy()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_start = time.perf_counter()
                normalized = policy_state(torch.as_tensor(state["qpos"], device=device)[None],
                                          goal_tensor, maps.settings)
                if hasattr(model, 'observe_step'):
                    model.observe_step(steps)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == 'cuda'):
                    chunk = model(torch.as_tensor(rgb, device=device)[None], normalized, K,
                        torch.as_tensor(state['T_B_C'], device=device, dtype=torch.float32)[None])
                chunk = chunk[0].float().cpu().numpy()
                inference_seconds.append(time.perf_counter() - infer_start)
                replans += 1
                chosen_execute = model.execution_horizon(execute) if hasattr(model, 'execution_horizon') else execute
                execution_history.append(chosen_execute)
                for action_index in range(min(chosen_execute, model.chunk_size, max_steps - steps)):
                    # All predictions in the chunk are relative to this replan's anchor.
                    target = anchor + chunk[action_index]
                    target_history.append(target.copy())
                    state, contacts, clipped = simulation.step(target)
                    clips += clipped
                    steps += 1
                    if contacts and first_collision is None:
                        first_collision = {"control_step": steps, "contacts": contacts}
                    position_error, rotation_error = record()
                    minimum_position_error = min(minimum_position_error, position_error)
                    reached_goal = is_goal(position_error, rotation_error)
                    if writer is not None and steps % stride == 0:
                        writer.append_data(simulation.render()[0])
                    if reached_goal or (contacts and options["stop_on_collision"]):
                        break
                if first_collision is not None and options["stop_on_collision"]:
                    break
            if writer is not None and steps % stride:
                writer.append_data(simulation.render()[0])
        finally:
            if writer is not None:
                writer.close()
    success = reached_goal and first_collision is None
    ee_positions = np.asarray(ee_history)[:, :3, 3]
    executed_length = float(np.linalg.norm(np.diff(ee_positions, axis=0), axis=1).sum())
    result = {
        "episode_id": episode, "route": route, "success": success,
        "goal_reached": reached_goal, "collision": first_collision,
        "termination": "collision" if first_collision else ("success" if success else "timeout"),
        "control_steps": steps, "replans": replans, "prediction_horizon": model.chunk_size,
        "execute_horizon": execute, "joint_limit_clip_count": clips,
        "final_position_error_m": position_error, "minimum_position_error_m": minimum_position_error,
        "final_orientation_error_rad": rotation_error,
        "goal_success_criterion": options.get("goal_success_criterion", "xyz_orientation"),
        "xyz_orientation_success": success and rotation_error <= options["goal_orientation_tolerance_rad"],
        "executed_path_length_m": executed_length,
        "mean_inference_ms": 1000 * float(np.mean(inference_seconds)) if inference_seconds else None,
        "wall_seconds": time.perf_counter() - started,
        "privileged_action_map": False,
        "expert_progress_method": None,
        "observation_schedule": getattr(model, 'schedule', 'fixed'),
        "scene_geometry": "tsn-1k complete room, object fittings, Panda v3 and collision envelopes",
    }
    np.savez_compressed(directory / "trajectory.npz", qpos=np.asarray(q_history),
                        T_B_E=np.asarray(ee_history), T_B_C=np.asarray(camera_history),
                        predicted_joint_targets=np.asarray(target_history).reshape(-1, 7),
                        reference_indices=np.asarray([], dtype=np.int64),
                        execution_horizons=np.asarray(execution_history))
    write_json(directory / "metrics.json", result)
    return result


def evaluate_rollouts(ids: list[str], catalog: dict[str, str], root: Path, output: Path,
                      model: NavigationPolicy, maps: GeometryMaps, device: torch.device,
                      options: dict[str, Any]) -> dict[str, Any]:
    """Run each selected episode once and persist partial summaries after every episode."""
    if not ids:
        raise ValueError("Closed-loop evaluation needs at least one episode")
    results = []
    for index, episode in enumerate(ids, start=1):
        result = rollout(episode, catalog[episode], root, output, model, maps, device,
                         options)
        results.append(result)
        print(f"[{index}/{len(ids)}] {episode} {result['termination']} "
              f"position error {result['final_position_error_m']:.4f} m", flush=True)
        summary = {**summarize_rollouts(results), "results": results}
        write_json(output / "closed_loop.json", summary)
    return summary
