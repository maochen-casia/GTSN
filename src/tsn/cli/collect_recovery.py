"""Collect checkpoint on-policy observations from training episodes."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import default_config, experiment_path, read_json, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.data.splits import ROUTES, episode_catalog, validate_splits
from tsn.evaluation.metrics import summarize_rollouts
from tsn.models.factory import make_maps, make_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, default=default_config("eval", "closed_loop.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--direct-episodes", type=int, default=16,
        help="Number of direct training episodes; over and side use twice this count each",
    )
    parser.add_argument("--device")
    args = parser.parse_args()

    if args.direct_episodes <= 0:
        raise ValueError("--direct-episodes must be positive")
    output = experiment_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    configuration = checkpoint["config"]
    benchmark = configuration["benchmark"]
    split = checkpoint["splits"]
    options = read_json(args.eval_config)
    if args.device:
        options["device"] = args.device
    options["render_videos"] = False
    seed = int(configuration["train"]["seed"])
    seed_everything(seed)
    device = require_device(options["device"])

    root = Path(benchmark["root"])
    catalog = episode_catalog(root)
    validate_splits(split, catalog)
    rng = random.Random(seed + 1)
    selected: list[str] = []
    requested = {"direct": args.direct_episodes, "over": 2 * args.direct_episodes,
                 "side": 2 * args.direct_episodes}
    for route in ROUTES:
        candidates = [episode for episode in split["train"] if catalog[episode] == route]
        if len(candidates) < requested[route]:
            raise ValueError(f"Requested {requested[route]} {route} episodes, only {len(candidates)} available")
        selected.extend(rng.sample(candidates, requested[route]))
    selected.sort()

    model = make_policy(configuration["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    maps = make_maps(configuration["model"]).to(device)

    recovery_root = output / "recovery_data"
    recovery_root.mkdir()
    write_json(output / "config.json", {
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
        "benchmark": benchmark, "model": configuration["model"], "eval": options,
        "source_partition": "train", "episodes": selected,
        "requested_episodes_by_route": requested,
        "privileged_action_map": True,
        "recovery_labels_use_expert_reference": True,
    })
    write_json(output / "splits.json", split)

    from tsn.evaluation.closed_loop import evaluate_rollouts

    summary = evaluate_rollouts(
        selected, catalog, root, output, model, maps, device, options, recovery_root
    )
    episode_counts = Counter(catalog[episode] for episode in selected)
    samples_by_route = Counter()
    sample_total = 0
    archive_ids = []
    for archive in sorted(recovery_root.glob("episode_*.npz")):
        with np.load(archive, allow_pickle=False) as data:
            route = str(data["route"])
            archive_ids.append(str(data["episode_id"]))
            samples_by_route[route] += len(data["depth"])
            sample_total += len(data["depth"])
    if sorted(archive_ids) != sorted(selected):
        raise RuntimeError("Recovery archive coverage differs from selected training episodes")

    write_json(recovery_root / "source_split.json", split)
    write_json(recovery_root / "manifest.json", {
        "schema_version": "tsn-1k-onpolicy-recovery-v1",
        "source_dataset": str(root), "source_partition": "train",
        "episode_ids": selected,
        "episodes": len(selected), "episodes_by_route": dict(episode_counts),
        "samples": sample_total,
        "samples_by_route": {route: samples_by_route[route] for route in ROUTES},
        "parameters": {
            "source_checkpoint": str(checkpoint_path),
            "execute_horizon": int(options["execute_horizon"]),
            "max_control_steps": int(options["max_control_steps"]),
            "seed": seed + 1,
        },
        "rollout_summary": summary["overall"],
    })
    write_json(output / "summary.json", {
        **summarize_rollouts(summary["results"]),
        "recovery_samples": sample_total,
        "recovery_episodes": len(archive_ids),
        "recovery_samples_by_route": {route: samples_by_route[route] for route in ROUTES},
        "source_checkpoint": str(checkpoint_path),
    })
    print(f"Collected {sample_total} recovery observations from {len(selected)} training episodes")
    print(f"Recovery data: {recovery_root}")


if __name__ == "__main__":
    main()
