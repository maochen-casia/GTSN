"""Compare recorded and reconstructed RGB at fixed validation initial states.

These three predetermined examples diagnose appearance mismatch, not causality
or full-distribution generalization. No model parameters are updated.
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from run_gtsn import backbone
from tsn.common.config import read_json, write_json
from tsn.data.splits import ROUTES, episode_catalog
from tsn.features.state import policy_state
from tsn.models.factory import make_maps
from tsn.models.gtsn_policy import GTSNHead
from tsn.simulation.episode import EpisodeSimulation


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, cfg, splits = backbone()
    root = Path(cfg['benchmark']['root'])
    catalog = episode_catalog(root)
    maps = make_maps(cfg['model']).cuda()
    head = GTSNHead('memory').cuda().eval()
    head.load_state_dict(torch.load(args.root / 'heads/memory/best.pt', weights_only=True))
    options = read_json('/workspace/configs/eval/pi3_small.json')
    fig, axes = plt.subplots(3, 2, figsize=(8, 8))
    records = []
    for row, route in enumerate(ROUTES):
        episode = next(ep for ep in splits['validation'] if catalog[ep] == route)
        with h5py.File(root / episode / 'episode.h5', 'r') as f:
            q, ee, goal = f['qpos'][0], f['ee_pose'][0], f['goal_pose_xyz_wxyz'][:]
            K, extrinsic, pose = f['intrinsics'][:], f['T_ee_camera_cv'][:], f['T_base_camera_cv'][0]
            original, original_depth = f['rgb'][0], f['depth_m'][0]
            names = tuple(x.decode() if isinstance(x, bytes) else str(x) for x in f['joint_names'][:])
        with EpisodeSimulation(read_json(root / episode / 'scene.json'), K, extrinsic,
                               original_depth.shape, q, ee, names, options) as simulation:
            rendered, depth = simulation.render()
            measured = simulation.snapshot()
        rgb = torch.as_tensor(np.stack([original, rendered]), device='cuda')
        poses = torch.as_tensor(np.stack([pose, measured['T_B_C']]), device='cuda', dtype=torch.float32)
        calibration = torch.as_tensor(K, device='cuda', dtype=torch.float32)[None].expand(2, -1, -1)
        state = policy_state(torch.as_tensor(q, device='cuda', dtype=torch.float32)[None].expand(2, -1),
                             torch.as_tensor(goal, device='cuda', dtype=torch.float32)[None].expand(2, -1), maps.settings)
        base, tokens, geometry = model(rgb, state, calibration, poses, return_features=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, aux = head(tokens[:, None], geometry[:, None], poses[:, None], torch.zeros(2, 1, device='cuda'),
                          torch.ones(2, 1, dtype=torch.bool, device='cuda'), state, base)
        records.append({'episode': episode, 'route': route, 'frame': 0,
                        'normalized_rgb_mean_absolute_difference': float(np.abs(original.astype(float)-rendered).mean()/255),
                        'mean_token_cosine_similarity': float(F.cosine_similarity(tokens[0].float(), tokens[1].float(), dim=-1).mean()),
                        'recorded_rgb_uncertainty_risk': float(aux['risk'][0]),
                        'rendered_rgb_uncertainty_risk': float(aux['risk'][1]),
                        'camera_transform_max_abs_difference': float(np.abs(pose-measured['T_B_C']).max())})
        for column, (label, image) in enumerate([('Recorded', original), ('Reconstructed', rendered)]):
            axes[row, column].imshow(image)
            axes[row, column].set_title(f'{route}: {episode}, {label}', fontsize=10)
            axes[row, column].axis('off')
    fig.tight_layout()
    figures = args.root / 'figures'
    figures.mkdir(exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(figures / f'appearance_comparison.{extension}', dpi=180)
    write_json(args.root / 'appearance_diagnostics.json', {'selection': 'first validation episode per route, frame zero',
        'examples': records, 'model_updated': False,
        'limitation': 'Three fixed examples; appearance effects are not isolated from other perception errors'})
    print(records)


if __name__ == '__main__':
    main()
