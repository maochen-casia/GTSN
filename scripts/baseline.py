"""Baseline commands; run this script inside the project Docker runtime."""
import argparse
from pathlib import Path
import torch
from tsn.baselines.data import prepare_cache
from tsn.baselines.runner import train, evaluate
from tsn.common.config import output_path, read_json
from tsn.common.seed import require_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('cache', 'train', 'evaluate'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--partition', choices=('validation', 'test'))
    parser.add_argument('--episode', action='append')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = output_path(args.output_dir)
    if args.command == 'cache':
        if not args.config:
            parser.error('cache requires --config')
        prepare_cache(read_json(args.config), output)
    elif args.command == 'train':
        if not args.config or not args.cache:
            parser.error('train requires --config and --cache')
        train(read_json(args.config), args.cache, output, require_device(args.device))
    else:
        if not args.checkpoint or not args.partition:
            parser.error('evaluate requires --checkpoint and --partition')
        evaluate(args.checkpoint, args.partition, output, require_device(args.device), args.episode)


if __name__ == '__main__':
    main()
