"""Validation-only development followed by frozen test evaluation, inside Docker."""
import argparse
import hashlib
from pathlib import Path
import shutil

import torch

from run_gtsn import backbone, SEED, CHECKPOINT, source_hashes
from run_cartesian import ROOT as BASE_ROOT
from tsn.common.config import read_json, write_json
from tsn.common.seed import seed_everything
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.models.cartesian_policy import CartesianHead, PandaKinematics
from tsn.models.consensus_policy import ConsensusHead, ConsensusPolicy
from tsn.models.factory import make_maps

ROOT = Path('/run/user/1016/experiments/gtsn_consensus_20261003')


def hashes():
    return {**source_hashes(), 'scripts/run_consensus.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--name', required=True)
    p.add_argument('--modes', nargs='+', default=['state', 'memory'])
    p.add_argument('--weights', nargs='+', type=float)
    p.add_argument('--threshold', type=float, default=1e9)
    p.add_argument('--temporal', type=float, default=0.)
    p.add_argument('--execute', type=int, default=15)
    p.add_argument('--servo', type=float, default=.08)
    p.add_argument('--screen', action='store_true')
    p.add_argument('--partition', choices=['validation', 'test'], default='validation')
    a = p.parse_args()
    torch.set_num_threads(1)
    seed_everything(SEED)
    specification = dict(modes=a.modes, weights=a.weights, threshold=a.threshold,
                         temporal=a.temporal, execute=a.execute, servo=a.servo)
    if a.partition == 'test':
        selection = read_json(a.root/'selection.json')
        if selection['test_conditions'].get(a.name) != specification:
            raise ValueError('Test condition must be frozen in selection.json')
    out = a.root/'rollouts'/a.partition/(a.name+('_screen' if a.screen else ''))
    if (out/'complete.json').exists():
        return
    if out.exists():
        raise RuntimeError(f'Incomplete output exists; preserve it and use a new name: {out}')
    model, cfg, splits = backbone()
    catalog = episode_catalog(Path(cfg['benchmark']['root']))
    validate_splits(splits, catalog)
    weights = {}
    heads = []
    for mode in a.modes:
        head = CartesianHead(mode).cuda()
        checkpoint = BASE_ROOT/'heads'/mode/'best.pt'
        head.load_state_dict(torch.load(checkpoint, weights_only=True))
        weights[str(checkpoint)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        heads.append(head)
    weights[str(CHECKPOINT)] = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    policy = ConsensusPolicy(model, ConsensusHead(heads, a.weights), PandaKinematics(),
        servo_radius=a.servo, execute=a.execute, orientation='baseline',
        disagreement_threshold=a.threshold, temporal_weight=a.temporal).cuda().eval()
    ids = splits[a.partition]
    if a.screen:
        ids = [ep for route, n in [('direct',4),('over',8),('side',8)]
               for ep in [x for x in ids if catalog[x] == route][:n]]
    provenance = dict(specification=specification, weight_sha256=weights, ids=ids,
                      source_sha256_at_start=hashes(), seed=SEED)
    write_json(out/'protocol.json', provenance)
    for filename in provenance['source_sha256_at_start']:
        dest = out/'source'/filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path('/workspace')/filename, dest)
    options = read_json('/workspace/configs/eval/pi3_small.json')
    options['execute_horizon'] = a.execute
    result = evaluate_rollouts(ids, catalog, Path(cfg['benchmark']['root']), out,
                               policy, make_maps(cfg['model']).cuda(), torch.device('cuda'), options)
    assert not result['privileged_action_map']
    end_hashes = hashes()
    write_json(out/'complete.json', dict(episodes=len(ids), source_sha256_at_end=end_hashes,
        sources_unchanged=provenance['source_sha256_at_start']==end_hashes))


if __name__ == '__main__':
    main()
