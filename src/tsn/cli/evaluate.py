"""Usage: python -m tsn.cli.evaluate --checkpoint /run/user/1016/experiments/RUN/best.pt."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import default_config, experiment_path, read_json, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.loaders import make_loader
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.open_loop import evaluate_predictions
from tsn.models.factory import make_maps, make_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, default=default_config("eval", "closed_loop.json"))
    parser.add_argument("--mode", choices=("open-loop", "closed-loop", "both"), default="both")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--episode", action="append", help="Restrict evaluation to listed checkpoint test episodes")
    parser.add_argument("--render-videos", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    options = read_json(args.eval_config)
    if args.device:
        options["device"] = args.device
    if args.render_videos is not None:
        options["render_videos"] = args.render_videos
    checkpoint = load_checkpoint(args.checkpoint)
    configuration = checkpoint["config"]
    seed_everything(int(configuration["train"]["seed"]))
    device = require_device(options["device"])
    root = args.dataset_root or Path(configuration["benchmark"]["root"])
    catalog = episode_catalog(root)
    split = checkpoint["splits"]
    validate_splits(split, catalog)
    ids = args.episode or split["test"]
    if len(ids) != len(set(ids)) or set(ids) - set(split["test"]):
        raise ValueError("Evaluation episodes must be unique members of the checkpoint's test split")
    model = make_policy(configuration["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    maps = make_maps(configuration["model"]).to(device)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = experiment_path(args.output_dir or args.checkpoint.resolve().parent / f"evaluation_{timestamp}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"checkpoint": str(args.checkpoint.resolve()),
                                       "checkpoint_epoch": checkpoint["epoch"], "mode": args.mode,
                                       "dataset_root": str(root), "episodes": ids, "eval": options,
                                       "model": configuration["model"],
                                       "full_test_split": set(ids) == set(split["test"]),
                                       "privileged_action_map": True})
    write_json(output / "splits.json", split)
    if args.mode in ("open-loop", "both"):
        dataset = FrameDataset(root, ids, catalog, model.chunk_size, int(options["frame_stride"]))
        try:
            loader = make_loader(dataset, options, False, int(configuration["train"]["seed"]))
            metrics = evaluate_predictions(model, maps, loader, device)
            write_json(output / "open_loop.json", metrics)
            print(f"Test open-loop RMSE: {metrics['rmse_rad']:.6f} rad", flush=True)
        finally:
            dataset.close()
    if args.mode in ("closed-loop", "both"):
        # Keep simulator/renderer imports out of open-loop-only evaluations.
        from tsn.evaluation.closed_loop import evaluate_rollouts
        evaluate_rollouts(ids, catalog, root, output, model, maps, device, options)
    print(f"Evaluation artifacts: {output}")


if __name__ == "__main__":
    main()

