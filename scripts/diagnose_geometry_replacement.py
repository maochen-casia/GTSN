"""Check learned decisions on training geometry, without held-out labels."""
import argparse
import gc
from pathlib import Path

import torch

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import write_json
from tsn.models.c1_memory import PersistentGeometry
from tsn.models.c2_embodiment import AttentionEmbodimentGeometry, EmbodimentGeometry
from tsn.models.c3_clearance import route_candidates


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--examples', type=int, default=128)
    args = parser.parse_args(); root = args.root
    torch.set_num_threads(1)
    examples = torch.load('/run/user/1016/experiments/learned_geometry_20261009/cache/examples.pt',
                          map_location='cpu', weights_only=True)[:args.examples]
    result = dict(partition='train', examples=len(examples), trials={})
    for name in ('c1_d2', 'c2_d2', 'joint_d4'):
        path = root/name/'training/best.pt'
        if not path.exists():continue
        saved = load_checkpoint(path); config = saved['config']['model']; settings = config['learned_geometry']
        memory = PersistentGeometry(**config['memory'], replacement=settings['c1'],
                                    hidden_dim=settings['width'], depth=settings['depth'],
                                    query_source=settings.get('query_source', 'bank'),
                                    query_capacity=settings.get('query_capacity'))
        if settings['c1']:
            memory.load_state_dict({k[len('memory.'):]: v for k, v in saved['model'].items() if k.startswith('memory.')})
        body = (AttentionEmbodimentGeometry(config['robot'], settings['width'], settings['depth'],
                                          settings.get('calibrated_risk', False)) if settings['c2']
                else EmbodimentGeometry(config['robot']))
        body.load_state_dict({k[len('embodiment.'):]: v for k, v in saved['model'].items() if k.startswith('embodiment.')})
        del saved; gc.collect()
        logits, sizes, ranges, errors, conditional, bank_changes = [], [], [], [], [], []
        for index, example in enumerate(examples):
            memory.reset(); original = PersistentGeometry(**config['memory'])
            for frame in example['history']:
                p = frame['points']; valid = torch.ones(len(p), dtype=torch.bool)
                for bank in (memory, original):
                    bank.update(p, valid, frame['radius'], frame['step'], frame['tcp'][:3, 3], example['goal'], frame['state'])
            points, radius, weights = memory.query(True)
            sizes.append(len(points))
            if not len(points):continue
            if settings['c1']:
                score = memory.scores(memory.points, memory.radius, memory.support, memory.scatter, memory.step-memory.last)
                logits.append(score)
                changed = (torch.cdist(memory.points, original.points).amin(-1) > 1e-6).float().mean()
                bank_changes.append(float(changed))
                other = examples[(index+1)%len(examples)]
                shifted = memory.selector(memory.points, memory.radius, memory.support, memory.scatter,
                    memory.step-memory.last, other['tcp'][:3, 3], other['goal'], other['state'])
                conditional.append(float((score-shifted).abs().mean()))
                assert torch.equal(weights, torch.ones_like(weights))
            if settings['c2']:
                candidates, _ = route_candidates(example['positions'][None], example['tcp'][None], example['goal'][None])
                padding = config['clearance']['max_padding']*((radius-.015)/.085).clamp(0, 1)
                context = {k: example[k] for k in ('state', 'goal', 'arm_points')}
                risk = body.contact_risk(candidates[0], example['rotations'], example['tcp'], points, padding,
                    example['fingers'], camera_extrinsic=example['mount'], context=context)
                teacher = EmbodimentGeometry.contact_risk(body, candidates[0], example['rotations'], example['tcp'],
                    points, padding, example['fingers'], camera_extrinsic=example['mount'])
                ranges.append(float(risk.max()-risk.min())); errors.append((risk-teacher).square())
        record = dict(mean_queried_points=sum(sizes)/len(sizes))
        if logits:
            values = torch.cat(logits)
            record.update(score_min=float(values.min()), score_max=float(values.max()), score_std=float(values.std()),
                mean_context_logit_change=sum(conditional)/len(conditional),
                mean_fraction_bank_coordinates_changed_vs_legacy=sum(bank_changes)/len(bank_changes), point_weights='unit')
        if errors:
            record.update(risk_rmse_vs_training_teacher=float(torch.cat(errors).mean().sqrt()),
                          mean_candidate_risk_range=sum(ranges)/len(ranges))
        result['trials'][name] = record
    result['note'] = 'Training-only diagnostic of learned behavior; it does not establish a causal navigation gain.'
    write_json(root/'replacement_diagnostics.json', result)


if __name__ == '__main__':main()
