"""Evaluate a frozen geometry adapter in short, isolated simulator batches."""
import argparse
import hashlib
from pathlib import Path
import torch
from tsn.common.config import create_output, write_json
from tsn.common.seed import seed_everything
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.models.policy import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--partition', choices=('validation', 'test'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', required=True)
    parser.add_argument('--strength', type=float)
    args = parser.parse_args()
    if args.strength is not None and not 0 <= args.strength <= 1:parser.error('Strength must lie in [0,1]')
    if args.partition == 'test' and args.strength is not None:parser.error('Test uses the selected checkpoint without overrides')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.12)
    model, maps, saved = load_policy(args.checkpoint, 'cuda')
    if args.strength is not None and saved['config']['model'].get('learned_geometry', {}).get('mode') == 'replacement':
        parser.error('Replacement modules do not accept strength overrides')
    if args.strength is not None:
        if model.learned_c1:model.memory.strength = args.strength
        if model.learned_c2:model.embodiment.strength = args.strength
    root = Path(saved['config']['benchmark']['root'])
    catalog = episode_catalog(root)
    validate_splits(saved['splits'], catalog)
    if len(set(args.episodes)) != len(args.episodes) or set(args.episodes)-set(saved['splits'][args.partition]):
        parser.error('Episodes must be unique members of the requested partition')
    seed_everything(saved['config']['train']['seed'])
    options = dict(saved['config']['eval']); options['render_videos'] = False
    create_output(args.output_dir)
    write_json(args.output_dir/'config.json', dict(checkpoint=str(args.checkpoint), partition=args.partition,
        episodes=args.episodes, eval=options, strengths=dict(c1=model.memory.strength,
            c2=getattr(model.embodiment, 'strength', 0)), override=args.strength))
    evaluate_rollouts(args.episodes, catalog, root, args.output_dir, model, maps, torch.device('cuda'), options)
    write_json(args.output_dir/'complete.json', dict(episodes=len(args.episodes), partition=args.partition))


if __name__ == '__main__':main()
