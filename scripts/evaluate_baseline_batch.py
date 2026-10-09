"""Bound simulator lifetime to a short episode batch without changing a policy."""
import argparse
from pathlib import Path
import torch
from tsn.baselines.runner import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--partition', choices=('validation', 'test'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.1)
    evaluate(args.checkpoint, args.partition, args.output_dir, torch.device('cuda'), args.episodes)


if __name__ == '__main__':
    main()
