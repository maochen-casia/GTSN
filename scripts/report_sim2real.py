"""Audit the camera-varied experiment and write its results inside Docker."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch
from safetensors import safe_open

from tsn.common.config import read_json, write_json
from tsn.data.hdf5_dataset import padded_future
from tsn.data.splits import episode_catalog, validate_splits
from tsn.models.kinematics import PandaKinematics


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_recovery(root, config, splits):
    manifest = read_json(root/'recovery/manifest.json')
    assert manifest['source_dataset'] == config['benchmark']['root']
    assert manifest['source_partition'] == 'train'
    assert set(manifest['requested_episode_ids']) == set(splits['train'])
    assert set(manifest['episode_ids']) <= set(splits['train'])
    archives = sorted((root/'recovery').glob('episode_*.npz'))
    assert {p.stem for p in archives} == set(manifest['episode_ids'])
    counts, minimum, maximum_pose_error = Counter(), float('inf'), 0.
    kin = PandaKinematics(config['model'].get('robot', 'panda'))
    for path in archives:
        with np.load(path, allow_pickle=False) as data, h5py.File(
                Path(config['benchmark']['root'])/path.stem/'episode.h5', 'r') as reference:
            count = len(data['qpos'])
            counts[str(data['route'])] += count
            assert data['rgb'].dtype == np.uint8 and data['rgb'].shape == (*data['depth'].shape, 3)
            assert data['valid_future'].all()
            np.testing.assert_allclose(data['K'], np.repeat(reference['intrinsics'][:][None], count, axis=0))
            np.testing.assert_allclose(data['T_ee_camera_cv'], reference['T_ee_camera_cv'][:], atol=1e-7)
            reconstructed = kin(torch.from_numpy(data['qpos'][:, :7])).detach().numpy()@data['T_ee_camera_cv']
            maximum_pose_error = max(maximum_pose_error, float(abs(reconstructed-data['T_B_C']).max()))
            minimum = min(minimum, float(data['clearance_m'].min()))
            q, ee = reference['qpos'][:], reference['ee_pose'][:, :3]
            for row, frame in enumerate(data['frame_index']):
                future, _ = padded_future(q[:, :7], int(frame), 30)
                xyz, _ = padded_future(ee, int(frame), 30)
                np.testing.assert_allclose(data['target'][row], future-data['qpos'][row, None, :7], atol=1e-6)
                np.testing.assert_allclose(data['future_ee'][row], xyz, atol=1e-6)
                assert np.linalg.norm(data['qpos'][row, :7]-q[frame, :7]) > 1e-6
            for key in data.files:
                if data[key].dtype.kind in 'fciu':
                    assert np.isfinite(data[key]).all(), (path, key)
    assert dict(counts) == manifest['samples_by_route'] and sum(counts.values()) == manifest['samples']
    assert minimum >= manifest['parameters']['minimum_clearance_m'] and maximum_pose_error < 3e-6
    return dict(episodes=len(archives), samples=manifest['samples'], samples_by_route=dict(counts),
                requested_episodes=800, rejected_episodes=manifest['rejected_episodes'],
                minimum_clearance_m=minimum, maximum_camera_pose_error=maximum_pose_error,
                training_partition_only=True, collection='independent perturbations only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--recovery-only', action='store_true')
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(1)
    config = read_json(root/'source/configs/sim2real.json')
    splits = read_json(root/'recovery/source_split.json')
    validate_splits(splits, episode_catalog(Path(config['benchmark']['root'])))
    recovery = audit_recovery(root, config, splits)
    write_json(root/'recovery_audit.json', recovery)
    if args.recovery_only:
        print(recovery, flush=True)
        return
    for folder, manifest in (('source', 'source_sha256.json'), ('generation_source', 'generation_source_sha256.json')):
        for name, expected in read_json(root/manifest).items():
            assert sha256(root/folder/name) == expected, (folder, name)
    checkpoint_path = root/'training/best.pt'
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    epochs = read_json(root/'training/epochs.json')
    initialization = read_json(root/'training/initialization.json')
    complete = read_json(root/'training/complete.json')
    assert len(epochs) == complete['epochs'] == 30
    assert initialization['parameters'] == initialization['trainable_parameters']
    assert initialization['navigation_initialized_from_scratch'] and not initialization['encoder_frozen']
    assert not initialization['test_used_for_selection']
    assert saved['splits'] == splits and saved['config'] == config
    assert config['model']['perception']['geometry_mode'] == 'camera_ray_depth'
    assert config['model']['robot'] == 'fr3' and not config['model']['maps']['clip_points']
    selected = min(epochs, key=lambda row:row['validation_rmse_m'])
    assert selected['epoch'] == saved['selected_epoch']
    assert selected['validation_rmse_m'] == saved['validation_rmse_m']
    assert all(torch.isfinite(value).all() for value in saved['model'].values())
    with safe_open(config['model']['perception']['pretrained_weights'], framework='pt', device='cpu') as official:
        keys = [key for key in official.keys() if key.startswith('encoder.')]
        changed = sum(not torch.equal(saved['model']['perception.'+key], official.get_tensor(key)) for key in keys)
    assert changed == len(keys)
    digest = sha256(checkpoint_path)
    assert digest == read_json(root/'selected_checkpoint.json')['sha256']
    partitions = {}
    for partition in ('validation', 'test'):
        output = root/partition
        settings, report = read_json(output/'config.json'), read_json(output/'closed_loop.json')
        rows = report['results']
        assert read_json(output/'complete.json')['episodes'] == len(rows) == 100
        assert settings['full_partition'] and settings['dataset_root'] == config['benchmark']['root']
        assert settings['checkpoint'] == str(checkpoint_path)
        assert {row['episode_id'] for row in rows} == set(splits[partition])
        assert dict(Counter(row['route'] for row in rows)) == dict(direct=20, over=40, side=40)
        for row in rows:
            directory = output/'episodes'/row['episode_id']
            assert row == read_json(directory/'metrics.json')
            assert row['robot'] == 'fr3' and not row['privileged_action_map']
            assert row['expert_progress_method'] is None
            assert row['success'] == (row['goal_reached'] and row['collision'] is None)
            assert row['execute_horizon'] == 15 and row['prediction_horizon'] == 30
            with np.load(directory/'trajectory.npz', allow_pickle=False) as trace:
                assert len(trace['qpos']) == row['control_steps']+1
                assert len(trace['predicted_joint_targets']) == row['control_steps']
                assert len(trace['reference_indices']) == 0 and (trace['execution_horizons'] == 15).all()
                for key in trace.files:
                    assert np.isfinite(trace[key]).all(), (partition, row['episode_id'], key)
        partitions[partition] = dict(successes=sum(row['success'] for row in rows),
                                     **report['overall'], by_route=report['by_route'])
    summary = dict(benchmark=config['benchmark']['root'], architecture='gtsn_main', robot='fr3',
                   geometry='camera-ray Z-depth transformed analytically with T_B_C', recovery=recovery,
                   initialization=initialization, epochs=30, selected_epoch=saved['selected_epoch'],
                   validation_waypoint_rmse_m=saved['validation_rmse_m'], checkpoint=str(checkpoint_path),
                   checkpoint_sha256=digest, partitions=partitions, test_used_for_selection=False,
                   audit=dict(source_hashes_verified=True, encoder_tensors_changed=changed,
                              encoder_tensors_compared=len(keys), finite_weights_and_trajectories=True,
                              complete_disjoint_partitions=True, fixed_execution_horizon=15))
    write_json(root/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    axes[0].plot([row['epoch'] for row in epochs], [row['loss'] for row in epochs])
    axes[0].set(xlabel='Epoch', ylabel='Training loss')
    axes[1].plot([row['epoch'] for row in epochs], [row['validation_rmse_m']*1000 for row in epochs])
    axes[1].axvline(selected['epoch'], linestyle='--', color='tab:orange')
    axes[1].set(xlabel='Epoch', ylabel='Validation waypoint RMSE (mm)')
    for suffix in ('png', 'pdf'):
        fig.savefig(root/f'training_curve.{suffix}', dpi=180)
    plt.close(fig)
    lines = ['# Camera-varied benchmark experiment', '',
             'Fresh navigation weights; official Pi3 encoder initialized and fully tuned. '
             'Camera-ray depth is transformed analytically into the robot base frame. '
             'FR3 kinematics and camera housing follow the episode calibration.', '',
             f"Training: 800 episodes, {recovery['samples']} independent perturbations, 30 epochs. "
             f"Selected epoch {saved['selected_epoch']} by validation waypoint RMSE "
             f"({saved['validation_rmse_m']*1000:.2f} mm).", '',
             '| Partition | Success | Collision | Direct | Over | Side |',
             '|---|---:|---:|---:|---:|---:|']
    for partition, values in partitions.items():
        routes = values['by_route']
        lines.append(f"| {partition} | {values['successes']}/100 | {values['collision_rate']*100:.0f}% | "+
                     ' | '.join(f"{routes[name]['success_rate']*100:.1f}%" for name in ('direct','over','side'))+' |')
    lines += ['', 'Success requires reaching XYZ within 10 mm without collision; orientation is reported separately. '
              'Both partitions contain 20 direct, 40 over and 40 side episodes. Test data was excluded from '
              'training, recovery generation and checkpoint selection.', '',
              'All code, generation provenance, calibration, encoder changes, complete splits and finite '
              'trajectories passed audit. Recovery labels return toward the next expert states; '
              'they do not come from collision-aware replanning. Simulation results do not measure real-robot transfer.', '',
              f'Checkpoint SHA-256: `{digest}`.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
