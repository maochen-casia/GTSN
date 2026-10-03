"""Estimate a constant-margin control using training observations only."""
import argparse
from pathlib import Path

import numpy as np
import torch

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import create_output, write_json
from tsn.models.compact_policy import CompactRouteHead, compact_state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    create_output(args.output_dir)
    torch.set_num_threads(1)
    _, state = compact_state(load_checkpoint(args.checkpoint))
    head = CompactRouteHead().to(args.device).eval()
    head.load_state_dict(state)
    tokens = np.load(args.cache/'train/tokens.npy', mmap_mode='r')
    geometry = np.load(args.cache/'train/geometry.npy', mmap_mode='r')
    scales = []
    scale = torch.tensor([.55, .55, .5], device=args.device)
    for start in range(0, len(tokens), 512):
        visual = torch.from_numpy(np.array(tokens[start:start+512], copy=True)).to(args.device)
        raw = head.distribution(head.visual(visual.float())).float()
        std = (-5+4*raw[..., 3:].tanh()).exp().sqrt()*scale
        sigma = std.square().mean(-1).sqrt().clamp(max=.08)
        points = torch.from_numpy(np.array(geometry[start:start+512, ..., :3], copy=True)).to(args.device)*scale+scale.new_tensor([.65, 0, .22])
        valid = ((points[..., 0] > .10) & (points[..., 0] < 1.05) &
                 (points[..., 1].abs() < .60) & (points[..., 2] > .04) & (points[..., 2] < .65))
        scales.append(sigma[valid].cpu().numpy())
    values = np.concatenate(scales)
    if not len(values):
        raise ValueError('No training points in workspace')
    result = dict(partition='train', samples=len(tokens), valid_pooled_points=len(values),
                  mean_sigma_m=float(values.mean()), median_sigma_m=float(np.median(values)),
                  constant_radius_m=float(.04+.5*values.mean()), multiplier=.5,
                  note='Mean pooled-cell scale on training expert and recovery observations; no validation/test calibration')
    write_json(args.output_dir/'calibration.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
