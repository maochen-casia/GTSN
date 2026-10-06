"""Collect verified corrections or train the retained current-view residual."""
import argparse
import math
from pathlib import Path

import torch

from tsn.common.config import output_path
from tsn.common.seed import require_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('collect', 'train'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--initial-head', type=Path)
    parser.add_argument('--initial-residual', type=Path)
    parser.add_argument('--head', choices=('current_view', 'compact'), default='current_view')
    parser.add_argument('--learning-rate', type=float, default=3e-5,
                        help='Learning rate for compact-head fine-tuning')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--recovery-cache', type=Path, action='append', default=[])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--draws', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--collection-episodes', type=int, default=800)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if args.stage == 'train' and (not args.cache or not args.recovery_cache or
                                 (args.head == 'current_view' and not args.initial_head)):
        parser.error('Training requires caches and, for current_view, --initial-head')
    if args.head == 'compact' and (args.stage != 'train' or args.initial_head or args.initial_residual):
        parser.error('Compact-head training initializes from --checkpoint; omit residual initialization')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Learning rate must be finite and positive')
    if min(args.epochs, args.draws, args.batch_size) < 1:
        parser.error('Training sizes must be positive')
    torch.set_num_threads(1)
    device = require_device(args.device)
    output = output_path(args.output_dir)
    if args.stage == 'collect':
        from tsn.training.sequence_recovery import collect
        collect(args.checkpoint, output, device, args.collection_episodes, args.shard,
                args.shards, args.seed, args.initial_head)
    elif args.head == 'compact':
        from tsn.training.compact_corrective import train
        train(args.checkpoint, args.cache, args.recovery_cache, output, device,
              args.epochs, args.draws, args.batch_size, args.seed, args.learning_rate)
    else:
        from tsn.training.corrective import train
        train(args.checkpoint, args.initial_head, args.initial_residual, args.cache,
              args.recovery_cache, output, device, args.epochs, args.draws, args.batch_size, args.seed)


if __name__ == '__main__':
    main()
