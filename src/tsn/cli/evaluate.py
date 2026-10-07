"""Closed-loop validation or testing of the main C1/C2/C3 model."""
import argparse
from pathlib import Path

import torch

from tsn.common.config import create_output, output_path, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.models.policy import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--partition', choices=('validation', 'test'), required=True)
    parser.add_argument('--episode', action='append', help='Optional subset for a short evaluation')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--render-videos', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    output = output_path(args.output_dir)
    if output.exists() and any(output.iterdir()):parser.error('Use a new output directory')
    torch.set_num_threads(1)
    device = require_device(args.device)
    policy, maps, checkpoint = load_policy(args.checkpoint, device)
    config, splits = checkpoint['config'], checkpoint['splits']
    seed_everything(config['train']['seed'])
    root = args.dataset_root or Path(config['benchmark']['root'])
    catalog = episode_catalog(root); validate_splits(splits, catalog)
    ids = args.episode or splits[args.partition]
    if not ids or len(ids) != len(set(ids)) or set(ids)-set(splits[args.partition]):
        parser.error('Episodes must be unique members of the selected partition')
    options = dict(config['eval'])
    if options['execute_horizon'] != policy.execute:parser.error('The policy executes 15 steps')
    if args.render_videos is not None:options['render_videos'] = args.render_videos
    create_output(output)
    write_json(output/'config.json', dict(checkpoint=str(args.checkpoint.resolve()), partition=args.partition,
        episodes=ids, dataset_root=str(root), eval=options, full_partition=set(ids) == set(splits[args.partition])))
    evaluate_rollouts(ids, catalog, root, output, policy, maps, device, options)
    write_json(output/'complete.json', dict(episodes=len(ids), partition=args.partition, architecture='gtsn_main'))


if __name__ == '__main__':main()
