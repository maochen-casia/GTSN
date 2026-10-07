"""Run closed-loop validation or testing with compact or clearance control."""
import argparse
import math
from pathlib import Path

import torch

from tsn.common.config import create_output, output_path, read_json, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.models.clearance_policy import load_clearance_policy
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
    parser.add_argument('--controller', choices=('clearance', 'compact', 'geometric_energy', 'adaptive_geometry', 'uncertain_clearance', 'embodied_clearance'), default='clearance',
                        help='compact executes the learned route without clearance corrections')
    parser.add_argument('--render-videos', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--clearance-margin', type=float, default=.04)
    parser.add_argument('--clearance-penalty', type=float, default=.08)
    parser.add_argument('--energy-ablation', choices=('full', 'no_history', 'current_frame', 'tcp_only', 'no_trust', 'fixed_trust'), default='full')
    parser.add_argument('--memory-mode', choices=('persistent', 'recent', 'current', 'unconfirmed', 'persistent_visual',
                                                'persistent_geometry', 'recent_geometry', 'unconfirmed_geometry'), default='unconfirmed')
    parser.add_argument('--refinement-mode', choices=('adaptive', 'uniform', 'fixed', 'none'), default='adaptive')
    parser.add_argument('--uncertainty-padding', type=float, default=.02)
    parser.add_argument('--body-mode', choices=('tcp','axial','hand','tool'), default='tool')
    parser.add_argument('--body-weight', type=float, default=.7)
    parser.add_argument('--body-self-mask', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--body-pose', choices=('fixed','roll','wide_roll'), default='fixed')
    parser.add_argument('--body-field', choices=('gaussian','signed'), default='gaussian')
    parser.add_argument('--body-representation', choices=('union','parts'), default='union')
    args = parser.parse_args()
    if (not math.isfinite(args.clearance_margin) or args.clearance_margin <= 0 or
            not math.isfinite(args.clearance_penalty) or args.clearance_penalty < 0):
        parser.error('Clearance margin must be finite and positive; penalty must be finite and nonnegative')
    output = output_path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output directory already exists; use a new directory')
    torch.set_num_threads(1)
    device = require_device(args.device)
    if args.controller == 'embodied_clearance':
        from tsn.models.embodied_clearance import load_embodied_clearance
        policy,maps,checkpoint = load_embodied_clearance(args.checkpoint,device,args.body_mode,args.body_weight,
            args.uncertainty_padding,args.body_self_mask,args.body_pose,args.body_field,args.body_representation)
        clearance = dict(mode='embodied_clearance',embodiment=policy.embodiment_config(),
                         refinement=policy.refinement_config(),memory=policy.memory_config(),
                         training=checkpoint['uncertainty_training'])
    elif args.controller == 'uncertain_clearance':
        from tsn.models.uncertain_clearance import load_uncertain_clearance
        policy, maps, checkpoint = load_uncertain_clearance(args.checkpoint, device, args.refinement_mode, args.uncertainty_padding)
        clearance = dict(mode='uncertain_clearance', refinement=policy.refinement_config(),
                         memory=policy.memory_config(), training=checkpoint['uncertainty_training'])
    elif args.controller == 'adaptive_geometry':
        from tsn.models.adaptive_geometry import load_adaptive_geometry
        policy, maps, checkpoint = load_adaptive_geometry(args.checkpoint, device, args.memory_mode)
        clearance = dict(mode='adaptive_geometry', memory_mode=args.memory_mode,
                         training=checkpoint['geometric_energy'], memory=policy.memory_config())
    elif args.controller == 'geometric_energy':
        from tsn.models.geometric_energy import load_geometric_energy
        policy, maps, checkpoint = load_geometric_energy(args.checkpoint, device, args.energy_ablation)
        clearance = dict(mode='geometric_energy', ablation=args.energy_ablation, training=checkpoint['geometric_energy'])
    elif args.controller == 'compact':
        policy, maps, checkpoint = load_compact_policy(args.checkpoint, device)
        clearance = dict(mode='disabled')
    else:
        policy, maps, checkpoint = load_clearance_policy(
            args.checkpoint, device, margin=args.clearance_margin, penalty=args.clearance_penalty)
        clearance = dict(mode='deterministic', margin=policy.margin, penalty=policy.penalty)
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
        parser.error('The route controller requires execute_horizon=15')
    options['device'] = str(device)
    if args.render_videos is not None:
        options['render_videos'] = args.render_videos
    create_output(output)
    write_json(output / 'config.json', dict(checkpoint=str(args.checkpoint.resolve()),
               partition=args.partition, episodes=ids, dataset_root=str(root), eval=options,
               full_partition=set(ids) == set(splits[args.partition]), variant=policy.schedule,
               controller=args.controller, clearance=clearance))
    evaluate_rollouts(ids, catalog, root, output, policy, maps, device, options)
    write_json(output / 'complete.json', dict(episodes=len(ids), partition=args.partition,
                                             variant=policy.schedule, controller=args.controller))


if __name__ == '__main__':
    main()
