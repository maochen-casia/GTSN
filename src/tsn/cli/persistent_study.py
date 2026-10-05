"""Prepare frozen features and compare persistent scene-memory architectures."""
import argparse
from pathlib import Path

import torch

from tsn.common.config import output_path
from tsn.common.seed import require_device
from tsn.training.persistent import prepare_cache, train_study, analyze_study, report_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cache', 'train', 'analyze', 'report'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--draws', type=int, default=2048)
    parser.add_argument('--seed', type=int, action='append')
    parser.add_argument('--variant', choices=('current', 'four', 'scene', 'scene_points'), action='append')
    parser.add_argument('--resume', action='store_true', help='Reuse completed models and restart an incomplete model')
    parser.add_argument('--study', type=Path, action='append', help='Completed training study, for analysis')
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--rollouts', type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.epochs < 1 or args.draws < 1:
        parser.error('epochs and draws must be positive')
    if args.stage in ('cache', 'train') and args.checkpoint is None:
        parser.error('cache/train requires --checkpoint')
    if args.stage in ('analyze', 'report') and not args.study:
        parser.error('analyze requires --study')
    if args.stage == 'report' and (args.analysis is None or args.rollouts is None):
        parser.error('report requires --analysis and --rollouts')
    torch.set_num_threads(1)
    if args.stage == 'cache':
        prepare_cache(args.checkpoint, args.cache, output_path(args.output_dir),
                      require_device(args.device), args.batch_size)
    elif args.stage == 'train':
        train_study(args.checkpoint, args.cache, output_path(args.output_dir), require_device(args.device),
                    args.epochs, args.seed or [20261005],
                    args.variant or ['current', 'four', 'scene', 'scene_points'], args.batch_size, args.draws, args.resume)
    elif args.stage == 'analyze':
        analyze_study(args.study, args.cache, output_path(args.output_dir), require_device(args.device))
    else:
        report_study(args.analysis, args.study, args.rollouts, output_path(args.output_dir), args.cache)


if __name__ == '__main__':
    main()
