"""Audit matched Panda experiments and report complete paired results."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from tsn.common.config import read_json, write_json
from tsn.data.splits import episode_catalog, validate_splits
from report_sim2real import sha256, audit_recovery
from run_cam_var_pipeline import VARIANTS, files_digest


def paired_comparison(full, ablated):
    full = {row['episode_id']:row for row in full}
    ablated = {row['episode_id']:row for row in ablated}
    ids = sorted(full)
    changes = np.array([int(full[ep]['success'])-int(ablated[ep]['success']) for ep in ids])
    rng = np.random.default_rng(20261008)
    means = changes[rng.integers(len(ids), size=(20000, len(ids)))].mean(1)*100
    return dict(full_minus_ablation_success_pp=float(changes.mean()*100),
        paired_bootstrap_95_percent_interval_pp=np.quantile(means, [.025, .975]).tolist(),
        both_success=sum(full[ep]['success'] and ablated[ep]['success'] for ep in ids),
        full_only=sum(full[ep]['success'] and not ablated[ep]['success'] for ep in ids),
        ablation_only=sum(not full[ep]['success'] and ablated[ep]['success'] for ep in ids),
        neither=sum(not full[ep]['success'] and not ablated[ep]['success'] for ep in ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    torch.set_num_threads(1)
    config = read_json(root/'configs/full.json')
    protocol = read_json(root/'protocol.json')
    variants = {name: VARIANTS[name] for name in protocol['variants']}
    assert files_digest(root/'source') == read_json(root/'source_sha256.json')
    dataset = Path(config['benchmark']['root'])
    for name, expected in read_json(root/'benchmark_sha256.json').items():
        assert sha256(dataset/name) == expected, ('benchmark changed', name)
    splits = read_json(root/'recovery/source_split.json')
    validate_splits(splits, episode_catalog(dataset))
    recovery = audit_recovery(root, config, splits)
    if protocol.get('reference_root'):
        reference = Path(protocol['reference_root'])
        expected_source = read_json(root/'source_sha256.json')
        for name, expected in read_json(reference/'source_sha256.json').items():
            if name.startswith(('src/', 'vendor/', 'assets/')) or name in ('Dockerfile', 'pyproject.toml'):
                assert expected_source[name] == expected, ('numerical implementation changed', name)
        assert files_digest(root/'recovery') == read_json(root/'recovery_sha256.json')
        assert sha256(root/'previous_seed_summary.json') == protocol['previous_seed_summary_sha256']
        for variant in variants:
            compared = read_json(root/'configs'/f'{variant}.json')
            previous = read_json(reference/'configs'/f'{variant}.json')
            compared['train']['seed'] = previous['train']['seed']
            assert compared == previous, ('Only training seed may change', variant)
    assert read_json(root/'tests_complete.json')['passed']
    assert set(read_json(root/'smoke_complete.json')) == set(variants)
    models, episodes = {}, {}
    for variant, removed in variants.items():
        directory = root/variant
        expected_config = read_json(root/'configs'/f'{variant}.json')
        contributions = {key:key != removed for key in ('c1', 'c2', 'c3')}
        comparable = read_json(root/'configs'/f'{variant}.json')
        comparable['model']['contributions'] = config['model']['contributions']
        assert comparable == config, 'Only the declared contribution may differ'
        saved = torch.load(directory/'training/best.pt', map_location='cpu', weights_only=True)
        epochs = read_json(directory/'training/epochs.json')
        initialization = read_json(directory/'training/initialization.json')
        complete = read_json(directory/'training/complete.json')
        selected = min(epochs, key=lambda row:row['validation_rmse_m'])
        assert [row['epoch'] for row in epochs] == list(range(1, 31))
        assert complete['epochs'] == len(epochs) == protocol['epochs'] == 30
        assert saved['config'] == expected_config and saved['splits'] == splits
        assert saved['selected_epoch'] == selected['epoch']
        assert saved['validation_rmse_m'] == selected['validation_rmse_m']
        assert initialization['navigation_initialized_from_scratch'] and not initialization['encoder_frozen']
        assert initialization['contributions'] == contributions
        assert initialization['history_slots'] == (1 if removed == 'c1' else 4)
        assert initialization['draws_per_epoch'] == 8192 and initialization['global_batch_size'] == 8
        assert initialization['distributed_processes'] == 1
        assert not initialization['test_used_for_selection']
        assert expected_config['model']['robot'] == 'panda'
        assert expected_config['model']['perception']['geometry_mode'] == 'camera_ray_depth'
        assert not expected_config['model']['maps']['clip_points']
        assert all(torch.isfinite(value).all() for value in saved['model'].values())
        if removed == 'c2':
            assert not any(key.startswith('embodiment.') for key in saved['model'])
        if removed == 'c3':
            assert all(row['loss_components']['uncertainty'] == row['loss_components']['trust'] == 0 for row in epochs)
            assert initialization['trainable_parameters'] < initialization['parameters']
        else:
            assert initialization['trainable_parameters'] == initialization['parameters']
        with safe_open(config['model']['perception']['pretrained_weights'], framework='pt', device='cpu') as official:
            keys = [key for key in official.keys() if key.startswith('encoder.')]
            changed = sum(not torch.equal(saved['model']['perception.'+key], official.get_tensor(key)) for key in keys)
        assert changed == len(keys)
        digest = sha256(directory/'training/best.pt')
        receipt = read_json(directory/'selected_checkpoint.json')
        assert digest == receipt['sha256'] and not receipt['test_used_for_selection']
        assert receipt['selected_epoch'] == selected['epoch']
        partitions = {}
        episodes[variant] = {}
        for partition in ('validation', 'test'):
            output = directory/partition
            settings = read_json(output/'config.json')
            report = read_json(output/'closed_loop.json')
            rows = report['results']
            assert len(rows) == read_json(output/'complete.json')['episodes'] == 100
            assert settings['full_partition'] and settings['dataset_root'] == str(dataset)
            assert settings['checkpoint'] == str(directory/'training/best.pt')
            assert {row['episode_id'] for row in rows} == set(splits[partition])
            assert dict(Counter(row['route'] for row in rows)) == dict(direct=20, over=40, side=40)
            for row in rows:
                episode = output/'episodes'/row['episode_id']
                assert row == read_json(episode/'metrics.json')
                assert row['robot'] == 'panda' and not row['privileged_action_map']
                assert row['contributions'] == contributions
                assert row['max_history_slots'] <= (1 if removed == 'c1' else 4)
                assert row['expert_progress_method'] is None
                assert row['success'] == (row['goal_reached'] and row['collision'] is None)
                assert row['execute_horizon'] == 15 and row['prediction_horizon'] == 30
                with np.load(episode/'trajectory.npz', allow_pickle=False) as trace:
                    assert len(trace['qpos']) == row['control_steps']+1
                    assert len(trace['predicted_joint_targets']) == row['control_steps']
                    assert len(trace['reference_indices']) == 0 and (trace['execution_horizons'] == 15).all()
                    for key in trace.files:
                        assert np.isfinite(trace[key]).all(), (variant, partition, row['episode_id'], key)
            partitions[partition] = dict(successes=sum(row['success'] for row in rows),
                collisions=sum(row['collision'] is not None for row in rows),
                timeouts=sum(not row['success'] and row['collision'] is None for row in rows),
                **report['overall'], by_route=report['by_route'])
            episodes[variant][partition] = rows
        models[variant] = dict(contributions=contributions, selected_epoch=selected['epoch'],
            validation_waypoint_rmse_m=selected['validation_rmse_m'],
            checkpoint=str(directory/'training/best.pt'), checkpoint_sha256=digest,
            initialization=initialization, partitions=partitions,
            encoder_tensors_changed=changed, encoder_tensors_compared=len(keys))
        del saved
    comparisons = {variant: {partition: paired_comparison(episodes['full'][partition], episodes[variant][partition])
                             for partition in ('validation', 'test')}
                   for variant in variants if variant != 'full'}
    seed_comparison = None
    if protocol.get('reference_root'):
        previous = read_json(root/'previous_seed_summary.json')
        assert previous['audit']['all_800_rollouts_complete']
        assert previous['protocol']['seed'] != protocol['seed']
        seed_comparison = dict(seeds=[previous['protocol']['seed'], protocol['seed']], models={},
                               full_minus_c1_pp={})
        for variant in variants:
            seed_comparison['models'][variant] = {}
            for partition in ('validation', 'test'):
                first = previous['models'][variant]['partitions'][partition]['success_rate']*100
                second = models[variant]['partitions'][partition]['success_rate']*100
                seed_comparison['models'][variant][partition] = dict(
                    success_percent_by_seed=[first, second], mean_success_percent=(first+second)/2)
        if 'without_c1' in variants:
            for partition in ('validation', 'test'):
                full = seed_comparison['models']['full'][partition]['success_percent_by_seed']
                ablated = seed_comparison['models']['without_c1'][partition]['success_percent_by_seed']
                gains = [a-b for a,b in zip(full, ablated)]
                seed_comparison['full_minus_c1_pp'][partition] = dict(
                    gains_by_seed=gains, mean_gain_pp=sum(gains)/len(gains))
    summary = dict(benchmark=str(dataset), robot='panda', protocol=protocol, recovery=recovery,
        benchmark_audit=read_json(root/'benchmark_audit.json'), models=models, paired_comparisons=comparisons,
        seed_comparison=seed_comparison,
        test_used_for_selection=False, audit=dict(source_hashes_verified=True, benchmark_contents_verified=True,
            all_rollouts_complete=True, rollouts=200*len(variants),
            all_800_rollouts_complete=len(variants) == 4,
            identical_reference_numerical_code=bool(protocol.get('reference_root')),
            identical_reference_recovery_data=bool(protocol.get('reference_root')),
            complete_disjoint_partitions=True, finite_weights_and_trajectories=True,
            all_pi3_encoders_fully_tuned=True, fixed_execution_horizon=15))
    write_json(root/'summary.json', summary)
    labels = dict(full='Full C1/C2/C3', without_c1='w/o C1', without_c2='w/o C2', without_c3='w/o C3')
    lines = ['# Panda variable-camera experiment', '',
        f"All {len(variants)} models train from fresh navigation weights with the official Pi3 image encoder fully tuned. "
        'The 1,000 Panda scenes use independent camera intrinsics and physical mounts; camera-ray Z-depth '
        'is transformed analytically to the base frame and the physical housing follows the episode calibration.', '',
        f"Training uses 800 episodes and {recovery['samples']} independent perturbation samples from "
        f"{recovery['episodes']} episodes, a 65/35 expert/perturbation mixture, and 20/40/40 route sampling. "
        f"Each model trains for 30 epochs, 8,192 samples per epoch, batch size 8, AdamW at 1e-4, seed {protocol['seed']}. "
        'Minimum waypoint RMSE on the 100 validation episodes selects each checkpoint before all test rollouts.', '',
        '| Model | Epoch | Val. RMSE (mm) | Val. success | Test success | Val./test collisions |',
        '|---|---:|---:|---:|---:|---:|']
    for variant, row in models.items():
        val, test = row['partitions']['validation'], row['partitions']['test']
        lines.append(f"| {labels[variant]} | {row['selected_epoch']} | {row['validation_waypoint_rmse_m']*1000:.2f} | "
                     f"{val['successes']}/100 | {test['successes']}/100 | {val['collisions']}/{test['collisions']} |")
    lines += ['', '| Model | Test direct (20) | Test over (40) | Test side (40) | Test XYZ + orientation |',
              '|---|---:|---:|---:|---:|']
    for variant, row in models.items():
        test = row['partitions']['test']
        values = ' | '.join(f"{test['by_route'][route]['success_rate']*100:.1f}%" for route in ('direct', 'over', 'side'))
        lines.append(f"| {labels[variant]} | {values} | {test['xyz_orientation_success_rate']*100:.0f}% |")
    lines += ['', 'C1 removal uses only the current RGB frame during training and inference and clears geometry '
              'before every observation. C2 removal scores only the TCP point and omits body self filtering. '
              'C3 removal bypasses clearance refinement and omits uncertainty/trust losses. Every other configuration '
              'setting and the complete 800/100/100 scene split are identical.', '',
              '| Full minus ablation | Test success gain (pp) | Paired bootstrap 95% interval (pp) | Full-only / ablation-only successes |',
              '|---|---:|---:|---:|']
    for variant, values in comparisons.items():
        result = values['test']; low, high = result['paired_bootstrap_95_percent_interval_pp']
        lines.append(f"| {labels[variant]} | {result['full_minus_ablation_success_pp']:+.0f} | [{low:+.0f}, {high:+.0f}] | "
                     f"{result['full_only']} / {result['ablation_only']} |")
    if seed_comparison:
        first, second = seed_comparison['seeds']
        lines += ['', f'## Matched training seeds {first} and {second}', '',
            'The numerical model/data/training code, expert benchmark contents, recovery archives, '
            'scene split and every training/evaluation setting are identical. Only the training seed changes. '
            'Navigation heads initialize freshly for both models and the official Pi3 encoder is fully tuned.', '',
            f'| Model | Test seed {first} | Test seed {second} | Mean test success |',
            '|---|---:|---:|---:|']
        for variant in variants:
            values = seed_comparison['models'][variant]['test']
            a, b = values['success_percent_by_seed']
            lines.append(f"| {labels[variant]} | {a:.0f}% | {b:.0f}% | {values['mean_success_percent']:.1f}% |")
        if 'without_c1' in variants:
            values = seed_comparison['full_minus_c1_pp']['test']
            a, b = values['gains_by_seed']
            lines += ['', f"Full-minus-w/o-C1 test success differences are {a:+.0f} and {b:+.0f} points; "
                      f"the two-seed mean difference is {values['mean_gain_pp']:+.1f} points. "
                      'Two training seeds provide a repeat of the experiment, not a reliable estimate of training-seed variance.']
    lines += ['', 'Success requires reaching XYZ within 10 mm without collision, using live wrist RGB and measured '
              'state, 30-step proposals, fixed 15-step execution and a 400-control-step limit. Orientation-constrained '
              'success uses 0.15 rad and is recorded when the XYZ rollout ends. Every validation/test evaluation '
              'contains all 100 episodes (20 direct, 40 over, 40 side).', '',
              'Source and benchmark hashes, recovery provenance and held-out isolation, full Pi3 encoder updates, '
              f"checkpoint selection, and finite weights/trajectories passed audit. All {200*len(variants)} rollouts read no expert "
              'future action or depth input. Perturbation labels return to subsequent expert states and are not '
              'collision-aware recovery plans. These are descriptive results from one training seed; paired '
              'intervals resample test scenes and do not measure variation across training seeds or real-robot transfer.', '',
              'Artifacts: `configs/`, `source/`, `recovery/`, `benchmark_audit.json`, each model’s `training/`, '
              '`selected_checkpoint.json`, `validation/`, `test/`, and `summary.json`.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    for variant in variants:
        records = read_json(root/variant/'training/epochs.json')
        axes[0].plot([r['epoch'] for r in records], [r['loss'] for r in records], label=labels[variant])
        axes[1].plot([r['epoch'] for r in records], [r['validation_rmse_m']*1000 for r in records], label=labels[variant])
    axes[0].set(xlabel='Epoch', ylabel='Training objective')
    axes[1].set(xlabel='Epoch', ylabel='Validation waypoint RMSE (mm)')
    axes[1].legend(fontsize=8)
    x = np.arange(len(variants))
    for i, partition in enumerate(('validation', 'test')):
        axes[2].bar(x+(i-.5)*.35, [models[v]['partitions'][partition]['success_rate']*100 for v in variants],
                    width=.35, label=partition.title())
    axes[2].set(xticks=x, xticklabels=[labels[v] for v in variants], ylabel='Collision-free success (%)', ylim=(0, 100))
    axes[2].legend(fontsize=8)
    for suffix in ('png', 'pdf'):
        fig.savefig(root/f'experiment.{suffix}', dpi=180)
    plt.close(fig)
    print({name:row['partitions'] for name, row in models.items()}, flush=True)


if __name__ == '__main__':
    main()
