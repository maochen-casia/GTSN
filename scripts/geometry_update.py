"""Cache frozen RGB predictions or train an independent learned C1/C2 adapter."""
import argparse
from pathlib import Path
import torch
from tsn.common.config import output_path
from tsn.training.geometry_update import cache_examples, train_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('cache', 'train'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--component', choices=('c1', 'c2'))
    parser.add_argument('--draws', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=4)
    args = parser.parse_args()
    if args.draws <= 0 or args.epochs <= 0:parser.error('Draws and epochs must be positive')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.12)
    args.output_dir = output_path(args.output_dir)
    if args.stage == 'cache':cache_examples(args.checkpoint, args.output_dir, args.draws)
    else:
        if args.cache is None or args.component is None:parser.error('Training requires cache and component')
        train_adapter(args.checkpoint, args.cache, args.output_dir, args.component, args.epochs)


if __name__ == '__main__':main()
