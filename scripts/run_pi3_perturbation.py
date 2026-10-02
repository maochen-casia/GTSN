"""Run the learned-map Pi3 experiment inside its project Docker image."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tsn.cli.fix_experiment import run_stage
from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import PROJECT_ROOT, default_config, experiment_path, read_json, write_json
from tsn.data.recovery_dataset import RecoveryDataset
from tsn.data.splits import episode_catalog, make_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = experiment_path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    torch.set_num_threads(1)
    benchmark = read_json(default_config("benchmark", "tsn-1k.json"))
    model = read_json(default_config("model", "pi3_small.json"))
    training = read_json(default_config("train", "pi3_small.json"))
    evaluation = read_json(default_config("eval", "pi3_small.json"))
    assert training["epochs"] == 10 and not training.get("initialize_from_checkpoint")
    assert model["chunk_size"] == 30 and evaluation["execute_horizon"] == 15
    assert len(training["recovery_sources"]) == 1
    split = make_splits(benchmark)
    catalog = episode_catalog(Path(benchmark["root"]))
    source = Path(training["recovery_sources"][0]["path"])
    weights_provenance = Path(model['pi3']['pretrained_weights']).parent / 'provenance.json'
    while not (source / 'manifest.json').is_file() or not weights_provenance.is_file():
        write_json(root / 'status.json', {'state': 'running', 'stage': 'preparation',
                   'waiting_for': [str(source / 'manifest.json'), str(weights_provenance)]})
        print('Waiting for verified Pi3 weights and completed RGB recovery data', flush=True)
        time.sleep(20)
    manifest = read_json(source / "manifest.json")
    assert manifest["source_partition"] == "train"
    assert manifest["source_dataset"] == benchmark["root"]
    recovery = RecoveryDataset(source, model["chunk_size"], include_rgb=True)
    try:
        ids = set(recovery.route_by_episode)
        assert ids == set(manifest["episode_ids"]) and ids <= set(split["train"])
        assert len(recovery) == manifest["samples"]
        route_samples = Counter()
        archive_hashes = {}
        minimum_clearance = float("inf")
        for path in recovery.paths:
            archive_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as data:
                episode = str(data["episode_id"])
                assert str(data["route"]) == catalog[episode]
                route_samples[str(data["route"])] += len(data["depth"])
                for key in ("depth", "T_B_C", "K", "future_ee", "qpos", "goal_pose", "target"):
                    assert np.isfinite(data[key]).all(), (path, key)
                assert data["valid_future"].all(), path
                assert data['rgb'].dtype == np.uint8 and data['rgb'].shape == (*data['depth'].shape, 3)
                original_root = Path(manifest['rgb_augmentation']['source'])
                with np.load(original_root / path.name, allow_pickle=False) as original:
                    assert all(np.array_equal(data[key], original[key]) for key in original.files), path
                minimum_clearance = min(minimum_clearance, float(data["clearance_m"].min()))
        assert minimum_clearance >= 0.002
        write_json(root / "data_audit.json", {
            "passed": True, "samples": len(recovery), "episodes": len(ids),
            "samples_by_route": dict(route_samples), "minimum_clearance_m": minimum_clearance,
            "validation_test_overlap": 0, "all_targets_finite": True,
            "all_terminal_targets_valid": True, "archive_sha256": archive_hashes,
            "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        })
    finally:
        recovery.close()
    write_json(root / "protocol.json", {
        "instruction": "instructions/baseline.md", "initialization": "published Pi3 image encoder; random small decoder and heads; no policy checkpoint",
        "benchmark": benchmark, "model": model, "train": training, "eval": evaluation,
        "recovery": "perturbation only; expert demonstrations remain 65% of training samples",
        "checkpoint_selection": "minimum expert validation RMSE; test excluded",
        "docker_image": os.environ.get("GTSN_DOCKER_IMAGE"),
        "docker_image_id": os.environ.get("GTSN_DOCKER_IMAGE_ID"),
        "pi3_weights": read_json(weights_provenance),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("src", "configs", "scripts", "instructions")
            for path in sorted((PROJECT_ROOT / directory).rglob("*"))
            if path.is_file() and path.suffix in {".py", ".json", ".md"}
        },
    })
    run_stage(root, "regression_tests", ["unittest", "discover", "-s", "/workspace/tests", "-v"])
    run_stage(root, "training", ["tsn.cli.train", "--model-config", str(default_config("model", "pi3_small.json")),
              "--train-config", str(default_config("train", "pi3_small.json")),
              "--run-dir", str(root / "train")])
    best = load_checkpoint(root / "train" / "best.pt")
    latest = load_checkpoint(root / "train" / "latest.pt")
    summary = read_json(root / "train" / "summary.json")
    assert latest["epoch"] == summary["completed_epochs"] == 10
    assert best["epoch"] == summary["best_epoch"] and 1 <= best["epoch"] <= 10
    assert best["splits"] == latest["splits"] == split
    for checkpoint in (best, latest):
        assert checkpoint["initialize_from_checkpoint"] is None
        assert checkpoint["config"]["train"] == training
        assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
    assert not summary["privileged_action_map"]
    assert read_json(root / "train" / "initialization.json")["all_parameters_trainable"]
    from safetensors import safe_open
    with safe_open(model['pi3']['pretrained_weights'], framework='pt', device='cpu') as initial:
        encoder_keys = [key for key in best['model'] if key.startswith('encoder.')]
        changed_encoder_tensors = sum(not torch.equal(best['model'][key], initial.get_tensor(key))
                                      for key in encoder_keys)
    assert changed_encoder_tensors == len(encoder_keys)
    del best, latest
    run_stage(root, "test", ["tsn.cli.evaluate", "--checkpoint", str(root / "train" / "best.pt"),
              "--eval-config", str(default_config("eval", "pi3_small.json")),
              "--output-dir", str(root / "test")])
    best = load_checkpoint(root / "train" / "best.pt")
    closed = read_json(root / "test" / "closed_loop.json")
    predictions = read_json(root / "test" / "open_loop.json")
    results = closed["results"]
    assert len(results) == 100 and {item["episode_id"] for item in results} == set(split["test"])
    assert dict(Counter(item["route"] for item in results)) == {"direct": 20, "over": 40, "side": 40}
    for item in results:
        assert item["prediction_horizon"] == 30 and item["execute_horizon"] == 15
        assert not item["privileged_action_map"] and item["expert_progress_method"] is None
        trajectory = root / "test" / "episodes" / item["episode_id"] / "trajectory.npz"
        with np.load(trajectory, allow_pickle=False) as data:
            assert len(data["predicted_joint_targets"]) == item["control_steps"]
            assert data["reference_indices"].size == 0
            assert all(np.isfinite(data[key]).all() for key in data.files)
    write_json(root / "audit.json", {"passed": True, "completed_epochs": 10,
               "policy_checkpoint_initialization": False, "full_backbone_finetuning": True, "privileged_maps": False, "selected_epoch": best["epoch"],
               "test_episodes": 100, "trajectories": 100, "prediction_horizon": 30,
               "execute_horizon": 15, "split_counts": split["route_counts"],
               "changed_encoder_tensors": changed_encoder_tensors,
               "encoder_tensors": len(encoder_keys), "recovery_labels_preserved": True})
    write_json(root / "results.json", {
        "checkpoint": str(root / "train" / "best.pt"), "checkpoint_epoch": best["epoch"],
        "training": summary, "open_loop": predictions,
        "closed_loop": {key: value for key, value in closed.items() if key != "results"},
        "terminations": dict(Counter(item["termination"] for item in results)),
        "median_final_position_error_m": float(np.median([item["final_position_error_m"] for item in results])),
    })
    write_json(root / "status.json", {"state": "complete", "results": str(root / "results.json")})
    print(f"Experiment complete: {root / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
