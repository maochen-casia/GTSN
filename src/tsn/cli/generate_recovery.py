"""Generate RGB/depth perturbation-only recovery data from training episodes."""
import argparse
from pathlib import Path

from tsn.common.config import PROJECT_ROOT, output_path, read_json
from tsn.data.recovery_generation import generate_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/sim2real.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--samples-per-episode', type=int, default=4)
    parser.add_argument('--noise-std-rad', type=float, default=.035)
    parser.add_argument('--noise-max-rad', type=float, default=.10)
    parser.add_argument('--minimum-clearance-m', type=float, default=.002)
    parser.add_argument('--max-attempts', type=int, default=30)
    parser.add_argument('--episode', action='append')
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    args = parser.parse_args()
    config = read_json(args.config)
    generate_recovery(config['benchmark'], config['eval'], output_path(args.output_dir), 30,
                      args.samples_per_episode, args.noise_std_rad, args.noise_max_rad,
                      args.minimum_clearance_m, args.max_attempts, args.workers, args.episode)


if __name__ == '__main__':
    main()
