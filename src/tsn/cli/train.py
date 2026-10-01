"""Usage: python -m tsn.cli.train [--run-dir /run/user/1016/experiments/NAME]."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tsn.common.config import default_config, read_json
from tsn.training.runner import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, default=default_config("benchmark", "tsn-1k.json"))
    parser.add_argument("--model-config", type=Path, default=default_config("model", "geometry_policy.json"))
    parser.add_argument("--train-config", type=Path, default=default_config("train", "baseline.json"))
    parser.add_argument("--run-dir", type=Path, help="New external run directory; existing directories are rejected")
    parser.add_argument("--device", help="Override configured device, e.g. cuda:0 or cpu")
    args = parser.parse_args()
    benchmark, model, options = (read_json(path) for path in (
        args.benchmark_config, args.model_config, args.train_config))
    if args.device:
        options["device"] = args.device
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.run_dir or Path(options["experiment_root"]) / f"{options['run_name']}_{timestamp}"
    checkpoint = train(benchmark, model, options, output)
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

