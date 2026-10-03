"""Evaluate the standalone compact model in the project Docker image."""
import argparse
from pathlib import Path

import torch

from tsn.common.config import write_json
from tsn.common.seed import seed_everything
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.models.simplified_consensus import load_compact_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episode', action='append')
    args = parser.parse_args()
    if args.output_dir.parent.resolve() != Path('/run/user/1016/experiments'):
        parser.error('Use a new direct child of /run/user/1016/experiments')
    torch.set_num_threads(1)
    seed_everything(20261002)
    policy, maps, checkpoint = load_compact_policy(args.checkpoint, 'cuda')
    config, splits = checkpoint['config'], checkpoint['splits']
    root = Path(config['benchmark']['root'])
    catalog = episode_catalog(root)
    validate_splits(splits, catalog)
    ids = args.episode or splits['test']
    if len(ids) != len(set(ids)) or set(ids)-set(splits['test']):
        parser.error('Episodes must be unique members of the checkpoint test split')
    args.output_dir.mkdir(exist_ok=False)
    evaluate_rollouts(ids, catalog, root, args.output_dir, policy, maps,
                      torch.device('cuda'), config['eval'])
    write_json(args.output_dir/'complete.json', dict(episodes=len(ids), variant=checkpoint['variant']))


if __name__ == '__main__':
    main()
