"""Train only the compact memory head, using an existing frozen Pi3 checkpoint."""
import argparse
from pathlib import Path

import torch

from tsn.common.config import default_config, output_path, read_json
from tsn.training.runner import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Compact export or the reference epoch-15 baseline.pt')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for this run')
    parser.add_argument('--config', type=Path, default=default_config('train', 'compact.json'))
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    options = read_json(args.config)
    if args.device:
        options['device'] = args.device
    for key in ('epochs', 'batch_size', 'cache_batch_size'):
        if options[key] <= 0:
            parser.error(f'{key} must be positive')
    if options['num_workers'] < 0 or options['learning_rate'] <= 0 or options['weight_decay'] < 0:
        parser.error('Invalid worker count or optimizer settings')
    torch.set_num_threads(1)
    train(args.checkpoint, output_path(args.output_dir), options)


if __name__ == '__main__':
    main()
