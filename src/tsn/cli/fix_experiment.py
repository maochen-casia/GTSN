"""Run the complete corrected recovery experiment and a matched perturbation control."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import PROJECT_ROOT, default_config, experiment_path, read_json, write_json
from tsn.data.recovery_generation import _generate_episode
from tsn.data.splits import episode_catalog
from tsn.training.selection import validation_episodes


def run_stage(root, name, arguments):
    done = root / "stages" / f"{name}.json"
    if done.exists():
        print(f"Completed stage retained: {name}", flush=True)
        return
    log = root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"Starting {name}: {log}", flush=True)
    write_json(root / "status.json", {"stage": name, "state": "running", "log": str(log)})
    with log.open("w") as stream:
        process = subprocess.Popen([sys.executable, "-m", *arguments], stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(20)
            lines = log.read_text(errors="replace").splitlines()
            print(f"{name}: {lines[-1] if lines else 'initializing'}", flush=True)
        if process.returncode:
            write_json(root / "status.json", {"stage": name, "state": "failed", "log": str(log)})
            raise RuntimeError(f"Stage {name} failed; see {log}")
    write_json(done, {"stage": name, "command": arguments, "wall_seconds": time.monotonic() - started})


def generate_validation(root, source, evaluation, workers):
    output = root / "validation_recovery"
    if (output / "manifest.json").exists():
        return
    dataset = Path(source["config"]["benchmark"]["root"])
    catalog = episode_catalog(dataset)
    ids = validation_episodes(source["splits"], catalog, {"direct": 3, "over": 6, "side": 6},
                              int(source["config"]["train"]["seed"]) + 3)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [{"episode": episode, "route": catalog[episode], "dataset_root": str(dataset),
             "output_dir": str(output), "eval_options": evaluation, "chunk_size": 30,
             "samples_per_episode": 4, "noise_std_rad": .035, "noise_max_rad": .10,
             "minimum_clearance_m": .002, "max_attempts": 30} for episode in ids]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = [future.result() for future in as_completed([executor.submit(_generate_episode, job) for job in jobs])]
    write_json(output / "manifest.json", {"source_partition": "validation", "requested_episode_ids": ids,
               "episode_ids": sorted(item["episode"] for item in results if item["accepted"]),
               "source_dataset": str(dataset), "samples": sum(item["accepted"] for item in results), "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = experiment_path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    model = read_json(default_config("model", "recovery_fixed.json"))
    evaluation = read_json(default_config("eval", "recovery_fixed.json"))
    training = read_json(default_config("train", "recovery_fixed.json"))
    parent_path = Path(training["initialize_from_checkpoint"])
    original = load_checkpoint(parent_path)
    write_json(root / "protocol.json", {
        "problems_fixed": [1, 3, 4, 5], "problem_2": "unchanged expert-reference recovery labels",
        "parent_checkpoint": str(parent_path), "model": model, "eval": evaluation,
        "epochs": 5, "learning_rate": .00008, "source_fractions": {
            "control": {"expert": .65, "perturbation": .35},
            "mixed": {"expert": .42, "perturbation": .35, "onpolicy": .23}},
        "route_sampling": {"direct": 1 / 3, "over": 1 / 3, "side": 1 / 3},
        "selection": "parent or final epoch on fixed validation rollouts and recovery states; ties retain parent",
        "test": "all 100 reserved test episodes; same settings for parent, control and mixed",
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for directory in ("src", "configs", "scripts") for path in sorted((PROJECT_ROOT / directory).rglob("*"))
                          if path.is_file() and path.suffix in {".py", ".json"}},
    })
    run_stage(root, "acceptance", ["tsn.cli.validate_recovery", "--all-collection", "--output-dir", str(root / "acceptance")])
    run_stage(root, "perturbation_data", ["tsn.cli.generate_recovery", "--eval-config", str(default_config("eval", "recovery_fixed.json")),
                    "--output-dir", str(root / "perturbation_data"), "--workers", str(args.workers)])
    generate_validation(root, original, evaluation, args.workers)
    adapted = {**original, "config": {**original["config"], "model": model}}
    if not (root / "parent.pt").exists():
        save_checkpoint(root / "parent.pt", adapted)
    run_stage(root, "onpolicy_collection", ["tsn.cli.collect_recovery", "--checkpoint", str(root / "parent.pt"),
                    "--eval-config", str(default_config("eval", "recovery_fixed.json")), "--output-dir", str(root / "onpolicy_collection")])
    for name in ("control", "mixed"):
        options = {**training, "run_name": f"recovery_fixed_{name}",
                   "checkpoint_selection": {**training["checkpoint_selection"], "recovery_data_dir": str(root / "validation_recovery")},
                   "recovery_sources": [{"name": "safe_joint_perturbation", "path": str(root / "perturbation_data"), "sampling_fraction": .35}]}
        if name == "mixed":
            options["recovery_sources"].append({"name": "on_policy_rollout", "path": str(root / "onpolicy_collection" / "recovery_data"), "sampling_fraction": .23})
        config = root / f"{name}_train.json"
        write_json(config, options)
        run_stage(root, f"train_{name}", ["tsn.cli.train", "--model-config", str(default_config("model", "recovery_fixed.json")),
                         "--train-config", str(config), "--run-dir", str(root / name)])
    candidates = {"parent": root / "parent.pt", "control": root / "control" / "best.pt", "mixed": root / "mixed" / "best.pt"}
    for name in ("control", "mixed"):
        if load_checkpoint(candidates[name])["epoch"] == 0:
            # Report rejected fine-tuning candidates without using their test
            # outcomes for selection or hiding them behind the retained parent.
            candidates[f"{name}_final"] = root / name / "latest.pt"
    report = {}
    for name, checkpoint in candidates.items():
        run_stage(root, f"test_{name}", ["tsn.cli.evaluate", "--checkpoint", str(checkpoint),
                         "--eval-config", str(default_config("eval", "recovery_fixed.json")), "--output-dir", str(root / f"test_{name}")])
        closed = read_json(root / f"test_{name}" / "closed_loop.json")
        report[name] = {"checkpoint": str(checkpoint), "checkpoint_epoch": load_checkpoint(checkpoint)["epoch"],
                        "open_loop": read_json(root / f"test_{name}" / "open_loop.json"),
                        "closed_loop": {key: value for key, value in closed.items() if key != "results"},
                        "median_final_position_error_m": float(np.median([item["final_position_error_m"] for item in closed["results"]]))}
    write_json(root / "results.json", report)
    write_json(root / "status.json", {"state": "complete", "results": str(root / "results.json")})
    print(f"Experiment complete: {root / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
