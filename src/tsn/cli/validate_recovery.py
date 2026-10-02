"""Docker acceptance checks for terminal supervision and reconstruction fidelity."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from tsn.common.config import default_config, experiment_path, read_json, write_json
from tsn.data.hdf5_dataset import JOINT_NAMES, padded_future
from tsn.features.maps import GeometryMaps
from tsn.simulation.episode import EpisodeSimulation
from tsn.training.losses import imitation_loss
from tsn.training.selection import selection_key


def check_terminal():
    values = np.arange(21, dtype=np.float32).reshape(3, 7)
    for frame in (0, 1, 2):
        future, valid = padded_future(values, frame, 30)
        assert valid.all() and np.array_equal(future[-1], values[-1])
    target = torch.tensor(padded_future(values, 2, 30)[0] - (values[-1] + .426))[None]
    prediction = torch.zeros_like(target, requires_grad=True)
    valid = torch.ones((1, 30), dtype=torch.bool)
    loss = imitation_loss(prediction, target, valid, .02, 10., .25)
    loss.backward()
    assert loss > 0 and prediction.grad.abs().sum() > 0
    settings = read_json(default_config("model", "geometry_policy.json"))["maps"]
    maps = GeometryMaps(settings)
    depth = torch.ones((1, 80, 80))
    K = torch.tensor([[[80., 0, 39.5], [0, 80., 39.5], [0, 0, 1]]])
    terminal = torch.tensor([0., 0., .5])[None, None].expand(1, 30, 3)
    channels = maps(depth, K, torch.eye(4)[None], terminal[:, 0], terminal, valid)
    assert channels[:, -1].max() > 0
    parent = selection_key({"overall": {"success_rate": .6, "collision_rate": .1}}, {"rmse_rad": .1}, {"rmse_rad": .06})
    unsafe = selection_key({"overall": {"success_rate": .5, "collision_rate": .3}}, {"rmse_rad": .01}, {"rmse_rad": .01})
    assert parent > unsafe


def check_scene(root, ids, options):
    results = []
    for episode in ids:
        with h5py.File(root / episode / "episode.h5", "r") as handle:
            q = handle["qpos"][:]
            ee = handle["ee_pose"][:]
            K = handle["intrinsics"][:]
            extrinsic = handle["T_ee_camera_cv"][:]
            with EpisodeSimulation(read_json(root / episode / "scene.json"), K, extrinsic,
                                   handle["depth_m"].shape[1:], q[0], ee[0], JOINT_NAMES, options) as simulation:
                for frame in sorted({0, len(q) // 2, len(q) - 1}):
                    simulation.robot.set_qpos(q[frame])
                    actual = simulation.render()[1]
                    expected = handle["depth_m"][frame]
                    common = (actual > 0) & (expected > 0)
                    error = np.abs(actual[common] - expected[common])
                    result = {"episode": episode, "route": str(handle.attrs["route_type"]), "frame": frame,
                              "missing_fraction": float(np.mean(actual <= 0)),
                              "median_depth_error_m": float(np.median(error)),
                              "p95_depth_error_m": float(np.percentile(error, 95)),
                              "fraction_within_2mm": float(np.mean(error <= .002)),
                              "camera_max_error": float(np.max(np.abs(simulation.snapshot()["T_B_C"] - handle["T_base_camera_cv"][frame])))}
                    print(result, flush=True)
                    results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", action="append")
    parser.add_argument("--all-collection", action="store_true")
    args = parser.parse_args()
    check_terminal()
    root = Path("/run/user/1016/tsn-1k")
    if args.all_collection:
        manifest = read_json("/run/user/1016/experiments/geometry_baseline_gpu_20261001_recovery10_onpolicy_collection/recovery_data/manifest.json")
        ids = manifest["episode_ids"]
    else:
        ids = args.episode or ["episode_003", "episode_211", "episode_606"]
    results = check_scene(root, ids, read_json(default_config("eval", "closed_loop.json")))
    passed = all(r["missing_fraction"] < .01 and r["median_depth_error_m"] < .0015 and
                 r["fraction_within_2mm"] > .95 and r["camera_max_error"] < 1e-5 for r in results)
    output = experiment_path(args.output_dir)
    write_json(output / "acceptance.json", {"terminal_supervision_passed": True, "scene_passed": passed,
                                           "criteria": {"missing_fraction_max": .01, "median_error_max_m": .0015,
                                                        "fraction_within_2mm_min": .95, "camera_error_max": 1e-5},
                                           "frames": len(results), "results": results})
    if not passed:
        raise AssertionError(f"Reconstruction failed acceptance: {output}")


if __name__ == "__main__":
    main()
