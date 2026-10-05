"""Visualize actual RGB geometry and its action query at a saved rollout state.

This offline illustration re-renders measured saved robot states. It does not
perform a new rollout or use scene geometry as input to the policy.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from tsn.common.config import create_output, read_json, write_json
from tsn.features.state import policy_state
from tsn.models.clearance_policy import ClearancePolicy, load_clearance_policy
from tsn.simulation.episode import EpisodeSimulation


class IllustratedPolicy(ClearancePolicy):
    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        result = super().refine_waypoints(waypoints, rotation, tcp, near, pose)
        self.illustration = dict(proposal=waypoints.cpu().numpy()[0],
                                 corrected=result.cpu().numpy()[0], rotation=rotation.cpu().numpy()[0])
        return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    create_output(args.output_dir)
    torch.set_num_threads(1)
    config = read_json(args.run/'config.json')
    cfg = config['clearance']
    if (cfg['mode'] != 'deterministic' or cfg.get('geometry_update') or
            cfg.get('body') or cfg.get('trigger')):
        raise ValueError('This illustration reconstructs the fixed-margin union-memory method')
    source = args.run/'episodes'/args.episode
    diagnostics = read_json(source/'policy_diagnostics.json')
    chosen = next((x for x in diagnostics if x['step'] >= 45 and x['correction_m'] > 0), None)
    if chosen is None:
        raise ValueError('No corrected observation with four available views')
    step = chosen['step']
    trace = np.load(source/'trajectory.npz')
    device = torch.device(args.device)
    base, maps, _ = load_clearance_policy(Path(config['checkpoint']), device)
    policy = IllustratedPolicy(base.backbone, base.head, base.kinematics,
                               margin=cfg['margin'], penalty=cfg['penalty']).to(device).eval()
    dataset = Path(config['dataset_root'])/args.episode
    with h5py.File(dataset/'episode.h5', 'r') as data:
        q0, ee0 = data['qpos'][0], data['ee_pose'][0]
        calibration, extrinsic = data['intrinsics'][:], data['T_ee_camera_cv'][:]
        source_hw = data['depth_m'].shape[1:]
        goal = data['goal_pose_xyz_wxyz'][:]
        names = tuple(x.decode() if isinstance(x, bytes) else str(x) for x in data['joint_names'][:])
    K = torch.as_tensor(calibration, device=device, dtype=torch.float32)[None]
    goal_tensor = torch.as_tensor(goal, device=device, dtype=torch.float32)[None]
    replay = list(range(step-45, step+1, 15))
    with EpisodeSimulation(read_json(dataset/'scene.json'), calibration, extrinsic,
                           source_hw, q0, ee0, names, config['eval']) as sim:
        for index in replay:
            sim.robot.set_qpos(trace['qpos'][index].astype(np.float32))
            measured = sim.snapshot()
            if not np.allclose(measured['T_B_E'], trace['T_B_E'][index], atol=1e-5):
                raise ValueError('Saved pose and re-rendered robot state disagree')
            rgb, _ = sim.render()
            state = policy_state(torch.as_tensor(trace['qpos'][index], device=device)[None],
                                 goal_tensor, maps.settings)
            policy.observe_step(index)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
                policy(torch.as_tensor(rgb, device=device)[None], state, K,
                       torch.as_tensor(trace['T_B_C'][index], device=device, dtype=torch.float32)[None])
    actual = policy.diagnostics[-1]
    if actual['choice'] != chosen['choice']:
        raise ValueError('Illustrated correction does not reproduce the archived decision')
    points = torch.cat([c[0] for c in policy.clouds], 1)[0].cpu().numpy()
    valid = torch.cat([c[1] for c in policy.clouds], 1)[0].cpu().numpy()
    valid &= np.linalg.norm(points-trace['T_B_E'][step, :3, 3], axis=-1) > .07
    ages = np.repeat([45, 30, 15, 0], 400)
    points, ages = points[valid], ages[valid]
    proposal, corrected, rotation = [policy.illustration[k] for k in ('proposal', 'corrected', 'rotation')]
    original_hand = proposal-.05*rotation[..., 2]
    corrected_hand = corrected-.05*rotation[..., 2]
    slice_z = float(np.median(original_hand[:15, 2]))
    x = np.linspace(.15, 1.05, 180)
    y = np.linspace(-.55, .55, 180)
    xx, yy = np.meshgrid(x, y)
    queries = np.stack((xx.ravel(), yy.ravel(), np.full(xx.size, slice_z)), -1)
    field = np.zeros(xx.size)
    for start in range(0, len(queries), 512):
        distance = ((queries[start:start+512, None]-points[None])**2).sum(-1)
        if len(points):
            field[start:start+512] = np.exp(-.5*distance/cfg['margin']**2).max(-1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title(f'Live wrist RGB: {args.episode}, step {step}')
    axes[0].axis('off')
    axes[1].contourf(xx, yy, field.reshape(xx.shape), levels=np.linspace(0, 1, 11), cmap='YlOrRd', alpha=.7)
    axes[1].set_title(f'Predicted proximity field, z={slice_z:.2f} m')
    axes[1].set(xlabel='Base x (m)', ylabel='Base y (m)')
    scatter = axes[2].scatter(points[:, 1], points[:, 2], c=ages, cmap='viridis_r', s=10, alpha=.65, vmin=0, vmax=45)
    fig.colorbar(scatter, ax=axes[2], label='Observation age (control steps)', shrink=.7)
    axes[2].set_title('Remembered surfaces and upcoming hand motion')
    axes[2].set(xlabel='Base y (m)', ylabel='Base z (m)')
    for ax, a, b in ((axes[1], 0, 1), (axes[2], 1, 2)):
        ax.plot(original_hand[:15, a], original_hand[:15, b], '--', color='#285a9f', lw=2, label='Proposed hand center')
        ax.plot(corrected_hand[:15, a], corrected_hand[:15, b], color='#0b7965', lw=2, label='Corrected hand center')
        ax.scatter([goal[a]], [goal[b]], marker='*', color='#232323', s=90, label='Goal TCP')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=8, loc='best')
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output_dir/f'geometry_grounding.{suffix}', dpi=180)
    np.savez_compressed(args.output_dir/'geometry_grounding.npz', rgb=rgb, points=points,
                        observation_age=ages, proposal=proposal, corrected=corrected,
                        rotation=rotation, field=field.reshape(xx.shape), x=x, y=y, slice_z=slice_z)
    write_json(args.output_dir/'provenance.json', dict(
        run=str(args.run.resolve()), episode=args.episode, step=step, replay_observations=replay,
        step_rule='first corrected observation with four available views',
        decision_reproduced=True, archived_choice=chosen['choice'], actual_choice=actual['choice'],
        geometry_source='Only RGB-predicted point coordinates; no true obstacle geometry or depth in the field',
        status='Offline illustration of archived execution, not an additional independent rollout'))


if __name__ == '__main__':
    main()
