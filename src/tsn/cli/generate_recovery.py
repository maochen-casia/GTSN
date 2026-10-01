"""Generate simulator-rendered recovery frames from training demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsn.common.config import default_config, read_json
from tsn.data.recovery_generation import generate_recovery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, default=default_config("benchmark", "tsn-1k.json"))
    parser.add_argument("--eval-config", type=Path, default=default_config("eval", "closed_loop.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--samples-per-episode", type=int, default=4)
    parser.add_argument("--noise-std-rad", type=float, default=0.035)
    parser.add_argument("--noise-max-rad", type=float, default=0.10)
    parser.add_argument("--minimum-clearance-m", type=float, default=0.002)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episode", action="append", help="Limit generation to selected training episodes")
    args = parser.parse_args()
    generate_recovery(
        read_json(args.benchmark_config), read_json(args.eval_config), args.output_dir,
        args.chunk_size, args.samples_per_episode, args.noise_std_rad, args.noise_max_rad,
        args.minimum_clearance_m, args.max_attempts, args.workers,
        args.episode,
    )


if __name__ == "__main__":
    main()
