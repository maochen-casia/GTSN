"""Run closed-loop validation or testing with the single-memory compact model."""
import argparse
from pathlib import Path

import torch

from tsn.common.config import create_output, output_path, read_json, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.models.compact_policy import load_compact_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--partition', choices=('validation', 'test'), required=True)
    parser.add_argument('--episode', action='append', help='Optional subset of the selected partition')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--eval-config', type=Path, help='Override checkpoint rollout settings')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--render-videos', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    output = output_path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output directory already exists; use a new directory')
    torch.set_num_threads(1)
    device = require_device(args.device)
    policy, maps, checkpoint = load_compact_policy(args.checkpoint, device)
    config, splits = checkpoint['config'], checkpoint['splits']
    seed_everything(config['train'].get('seed', 20261002))
    root = args.dataset_root or Path(config['benchmark']['root'])
    catalog = episode_catalog(root)
    validate_splits(splits, catalog)
    ids = args.episode or splits[args.partition]
    if not ids or len(ids) != len(set(ids)) or set(ids) - set(splits[args.partition]):
        parser.error('Episodes must be unique members of the selected checkpoint partition')
    options = read_json(args.eval_config) if args.eval_config else dict(config['eval'])
    if options['execute_horizon'] != policy.execute:
        parser.error('The compact controller requires execute_horizon=15')
    options['device'] = str(device)
    if args.render_videos is not None:
        options['render_videos'] = args.render_videos
    create_output(output)
    write_json(output / 'config.json', dict(checkpoint=str(args.checkpoint.resolve()),
               partition=args.partition, episodes=ids, dataset_root=str(root), eval=options,
               full_partition=set(ids) == set(splits[args.partition]), variant='single_memory'))
    evaluate_rollouts(ids, catalog, root, output, policy, maps, device, options)
    write_json(output / 'complete.json', dict(episodes=len(ids), partition=args.partition, variant='single_memory'))


if __name__ == '__main__':
    main()
