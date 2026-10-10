"""Inspect learned response variation on training-only cached geometry."""
import argparse
from pathlib import Path
import torch
from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import write_json
from tsn.models.c1_memory import PersistentGeometry
from tsn.models.c2_embodiment import LearnedEmbodimentGeometry
from tsn.models.c3_clearance import route_candidates


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.root
    examples = torch.load(root/'cache/examples.pt', map_location='cpu', weights_only=True)[:32]
    c1 = load_checkpoint(root/'c1/selected.pt')
    c1_strength = c1['config']['model']['learned_geometry']['c1_strength']
    memory = PersistentGeometry(**c1['config']['model']['memory'], learned=True, strength=c1_strength)
    memory.load_state_dict({key[len('memory.'):]: value for key, value in c1['model'].items() if key.startswith('memory.')})
    del c1
    c2 = load_checkpoint(root/'c2/selected.pt')
    c2_strength = c2['config']['model']['learned_geometry']['c2_strength']
    body = LearnedEmbodimentGeometry(c2['config']['model'].get('robot', 'panda'), strength=c2_strength)
    body.load_state_dict({key[len('embodiment.'):]: value for key, value in c2['model'].items() if key.startswith('embodiment.')})
    del c2
    memory.eval(); body.eval()
    point_weights, region_weights, region_spreads = [], [], []
    for example in examples:
        memory.reset()
        for frame in example['history']:
            p = frame['points']
            memory.update(p, torch.ones(len(p), dtype=torch.bool), frame['radius'], frame['step'],
                          frame['tcp'][:3, 3], example['goal'], frame['state'])
        points, radius, weights = memory.query(True)
        if not len(points):continue
        point_weights.append(weights)
        tcp, rotations = example['tcp'], example['rotations']
        candidates, _ = route_candidates(example['positions'][None], tcp[None], example['goal'][None])
        valid = torch.isfinite(points).all(-1) & ((points-tcp[:3, 3]).norm(dim=-1) > .07)
        if not valid.any():continue
        local = torch.einsum('ctni,tij->ctnj', points[None, None]-candidates[0, :, 2:15:3, None], rotations[2:15:3])
        padding = .03*((radius-.015)/.085).clamp(0, 1)
        context = {key: example[key] for key in ('state', 'goal', 'arm_points')}
        weights = body.regional_weights(local, valid, padding, candidates[0], rotations, tcp,
                                       example['fingers'], example['mount'], context)
        region_weights.append(weights.flatten())
        region_spreads.append(float(weights.max()-weights.min()))
    def describe(values):
        values = torch.cat(values)
        return dict(min=float(values.min()), max=float(values.max()), mean=float(values.mean()),
                    std=float(values.std(unbiased=False)), count=len(values))
    result = dict(partition='train', cached_examples=len(examples), strengths=dict(c1=c1_strength, c2=c2_strength),
        c1_weights=describe(point_weights), c2_weights=describe(region_weights),
        c2_mean_within_example_range=sum(region_spreads)/len(region_spreads),
        note='Response variation is a training-cache diagnostic, not evidence of a navigation gain from conditioning.')
    write_json(root/'learned_response_diagnostics.json', result)
    print(result)


if __name__ == '__main__':main()
