"""Validation-only hold-action and CARP tokenizer reconstruction diagnostics.

Expert future labels enter the tokenizer diagnostic only. This is an offline
reconstruction check, never a deployment policy or a closed-loop upper bound.
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from tsn.baselines.policies import BaselinePolicy
from tsn.common.config import write_json
from tsn.models.kinematics import PandaKinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.1)
    device = torch.device('cuda')
    saved = torch.load(root/'carp/training/best.pt', map_location='cpu', weights_only=True)
    model = BaselinePolicy(saved['config']).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    kinematics = PandaKinematics().to(device)
    targets = np.load(root/'cache/validation/target.npy', mmap_mode='r', allow_pickle=False)
    qpos = np.load(root/'cache/validation/qpos.npy', mmap_mode='r', allow_pickle=False)
    zero_squared, reconstruction_squared, count = 0., 0., 0
    zero_joint_squared, reconstruction_joint_squared, joint_count = 0., 0., 0
    with torch.inference_mode():
        for start in range(0, len(targets), 64):
            target = torch.tensor(np.array(targets[start:start+64]), device=device)
            q = torch.tensor(np.array(qpos[start:start+64]), device=device)[:, None, :7]
            latent, _, _, _ = model.tokenizer.encode(model.normalize(target))
            reconstructed = model.tokenizer.decode(latent)*model.action_scale+model.action_center
            actual = kinematics(q+target[:, 4::5])[..., :3, 3]
            current = kinematics(q)[..., :3, 3]
            predicted = kinematics(q+reconstructed[:, 4::5])[..., :3, 3]
            zero_squared += float((actual-current).square().sum())
            reconstruction_squared += float((actual-predicted).square().sum())
            count += actual.numel()
            zero_joint_squared += float(target.square().sum())
            reconstruction_joint_squared += float((target-reconstructed).square().sum())
            joint_count += target.numel()
    result = {'partition': 'validation', 'samples': len(targets), 'test_used': False,
        'used_for_checkpoint_selection': False, 'zero_action_waypoint_rmse_m': (zero_squared/count)**.5,
        'zero_action_joint_rmse_rad': (zero_joint_squared/joint_count)**.5,
        'carp_tokenizer_teacher_action_reconstruction_waypoint_rmse_m': (reconstruction_squared/count)**.5,
        'carp_tokenizer_teacher_action_reconstruction_joint_rmse_rad': (reconstruction_joint_squared/joint_count)**.5,
        'reconstruction_uses_expert_future_labels': True,
        'reconstruction_is_deployment_policy': False, 'reconstruction_is_closed_loop_upper_bound': False}
    write_json(root/'validation_diagnostics.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
