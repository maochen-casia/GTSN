"""Fit a metric translation using training-only paired geometric supervision."""
import argparse
from pathlib import Path

import numpy as np
import torch

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import create_output, read_json, write_json
from tsn.models.compact_policy import CompactRouteHead, compact_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--evaluate-learned-mean', action='store_true')
    args = parser.parse_args()
    create_output(args.output_dir)
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(args.checkpoint)
    scale, center = np.array([.55, .55, .5]), np.array([.65, 0., .22])
    residuals, eligible = {}, {}
    for split in ('train', 'validation'):
        root = args.cache/split
        meta = read_json(root/'metadata.json')
        if meta['episode_ids'] != checkpoint['splits'][split]:
            raise ValueError('Cache episode IDs disagree with checkpoint split')
        predicted = np.asarray(np.load(root/'geometry.npy', mmap_mode='r')[..., :3], dtype=np.float64)*scale+center
        target = np.asarray(np.load(root/'teacher.npy', mmap_mode='r'), dtype=np.float64)*scale+center
        valid = np.load(root/'geometry_valid.npy')
        valid = valid & ((predicted[..., 0] > .10) & (predicted[..., 0] < 1.05) &
                         (np.abs(predicted[..., 1]) < .60) & (predicted[..., 2] > .04) & (predicted[..., 2] < .65))
        residuals[split] = (target-predicted)[valid]
        eligible[split] = valid
        if not len(residuals[split]):
            raise ValueError('No eligible paired geometry')
    bias = residuals['train'].mean(0)
    stats = {split:dict(points=len(values), rmse_before_m=float(np.sqrt(np.mean(values**2))),
                       rmse_after_m=float(np.sqrt(np.mean((values-bias)**2))),
                       residual_mean_before_m=values.mean(0).tolist()) for split,values in residuals.items()}
    if args.evaluate_learned_mean:
        _, weights = compact_state(checkpoint)
        head = CompactRouteHead().to(args.device).eval()
        head.load_state_dict(weights)
        with torch.inference_mode():
            for split in ('train', 'validation'):
                root = args.cache/split
                tokens = np.load(root/'tokens.npy', mmap_mode='r')
                geometry = np.load(root/'geometry.npy', mmap_mode='r')
                teacher = np.load(root/'teacher.npy', mmap_mode='r')
                squared, count = 0., 0
                for start in range(0, len(tokens), 512):
                    visual = torch.from_numpy(np.array(tokens[start:start+512])).to(args.device)
                    with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                        enabled=torch.device(args.device).type == 'cuda'):
                        raw = head.distribution(head.visual(visual.float())).float()
                    correction = (.25*raw[..., :3].tanh()).cpu().numpy()
                    error = (np.asarray(teacher[start:start+512], dtype=np.float64)-
                             np.asarray(geometry[start:start+512, ..., :3], dtype=np.float64)-correction)*scale
                    error = error[eligible[split][start:start+512]]
                    squared += float((error**2).sum())
                    count += error.size
                stats[split]['rmse_learned_mean_m'] = float(np.sqrt(squared/count))
    result = dict(translation_m=bias.tolist(), fit_partition='train', checkpoint=str(args.checkpoint.resolve()),
                  cache=str(args.cache.resolve()), statistics=stats,
                  selection='One mean residual fitted on valid training cells within the predicted-point workspace; no validation fitting or test geometry.',
                  learned_mean_evaluated=args.evaluate_learned_mean,
                  limitation='Paired 4x4 pooled geometry calibrates a translation later applied to dense surfaces; it does not measure dense surface accuracy.')
    write_json(args.output_dir/'calibration.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
