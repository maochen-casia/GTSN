"""Audit complete baseline experiments and compare paired scenes with GTSN."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import torch

from tsn.baselines.policies import METHODS
from tsn.common.config import read_json, write_json
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.metrics import summarize_rollouts
from run_baselines import files_digest


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def paired(main, baseline):
    first = {r['episode_id']: r for r in main}
    second = {r['episode_id']: r for r in baseline}
    assert set(first) == set(second)
    ids = sorted(first)
    differences = np.array([int(first[k]['success'])-int(second[k]['success']) for k in ids])
    rng = np.random.default_rng(20261009)
    means = differences[rng.integers(len(ids), size=(20000, len(ids)))].mean(1)*100
    return {'main_minus_baseline_pp': float(differences.mean()*100),
        'paired_scene_bootstrap_95_interval_pp': np.quantile(means, [.025, .975]).tolist(),
        'main_only': sum(first[k]['success'] and not second[k]['success'] for k in ids),
        'baseline_only': sum(second[k]['success'] and not first[k]['success'] for k in ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    torch.set_num_threads(1)
    protocol = read_json(root/'protocol.json')
    config = read_json(root/'base_config.json')
    dataset = Path(config['benchmark']['root'])
    reference = Path(protocol['main_root'])
    assert files_digest(root/'source') == read_json(root/'source_sha256.json'), 'Source changed'
    for name, expected in read_json(root/'benchmark_sha256.json').items():
        assert sha256(dataset/name) == expected, ('Benchmark changed', name)
    for name, expected in read_json(root/'main_reference_sha256.json').items():
        assert sha256(reference/name) == expected, ('Main reference changed', name)
    assert read_json(root/'tests_complete.json')['passed']
    smoke = read_json(root/'smoke_complete.json')
    assert set(smoke) == set(METHODS)
    assert all(row['smoke_only'] and not row['test_partition_used'] and row['control_steps'] > 0
               for row in smoke.values())
    splits = read_json(root/'cache/splits.json')
    validate_splits(splits, episode_catalog(dataset))
    assert splits == read_json(reference/'recovery/source_split.json')
    cache = read_json(root/'cache/complete.json')
    assert not cache['test_cached'] and cache['normalization_fit_partition'] == 'train'
    assert cache['recovery_partition'] == 'train' and cache['partitions']['train']['perturbation_samples'] > 0
    main_summary = read_json(root/'main_summary.json')
    assert main_summary['audit']['all_800_rollouts_complete']
    models, comparisons = {}, {}
    for method in METHODS:
        directory = root/method
        saved = torch.load(directory/'training/best.pt', map_location='cpu', weights_only=True)
        epochs = read_json(directory/'training/epochs.json')
        selected = min(epochs, key=lambda r: r['validation_rmse_m'])
        assert [r['epoch'] for r in epochs] == list(range(1, protocol['epochs']+1))
        assert read_json(directory/'training/complete.json')['epochs'] == protocol['epochs']
        assert saved['architecture'] == 'tsn_baseline' and saved['method'] == method
        assert saved['config'] == read_json(root/'configs'/f'{method}.json') and saved['splits'] == splits
        assert saved['selected_epoch'] == selected['epoch']
        assert saved['validation_rmse_m'] == selected['validation_rmse_m']
        assert not saved['test_used_for_selection']
        assert all(torch.isfinite(value).all() for value in saved['model'].values())
        initialization = read_json(directory/'training/initialization.json')
        assert initialization['from_scratch'] and not initialization['test_used_for_selection']
        assert initialization['draws_per_epoch'] == protocol['draws_per_epoch']
        assert initialization['global_batch_size'] == protocol['batch_size']
        checkpoint_digest = sha256(directory/'training/best.pt')
        assert checkpoint_digest == read_json(directory/'training/selected_checkpoint.json')['sha256']
        if method == 'carp':
            tokenizer = read_json(directory/'training/tokenizer_epochs.json')
            assert [r['epoch'] for r in tokenizer] == list(range(1, protocol['carp_additional_tokenizer_epochs']+1))
        values = {'selected_epoch': saved['selected_epoch'], 'validation_waypoint_rmse_m': saved['validation_rmse_m'],
            'checkpoint': str(directory/'training/best.pt'), 'checkpoint_sha256': checkpoint_digest,
            'depth_input_at_inference': method in ('dp3', 'flowpolicy'), 'initialization': initialization}
        comparisons[method] = {}
        for partition in ('validation', 'test'):
            output = directory/partition
            settings = read_json(output/'config.json')
            report = read_json(output/'closed_loop.json')
            rows = report['results']
            assert settings['full_partition'] and settings['eval'] == config['eval']
            assert settings['checkpoint_sha256'] == checkpoint_digest
            assert read_json(output/'splits.json') == splits
            assert len(rows) == read_json(output/'complete.json')['episodes'] == 100
            assert {r['episode_id'] for r in rows} == set(splits[partition])
            assert dict(Counter(r['route'] for r in rows)) == dict(direct=20, over=40, side=40)
            summary = summarize_rollouts(rows)
            assert summary['overall'] == report['overall'] and summary['by_route'] == report['by_route']
            for row in rows:
                episode = output/'episodes'/row['episode_id']
                assert row == read_json(episode/'metrics.json')
                assert row['baseline_method'] == method and row['robot'] == 'panda'
                assert row['depth_input_at_inference'] == (method in ('dp3', 'flowpolicy'))
                assert not row['privileged_action_map'] and row['expert_progress_method'] is None
                assert row['prediction_horizon'] == 30 and row['execute_horizon'] == 15
                assert row['success'] == (row['goal_reached'] and row['collision'] is None)
                with np.load(episode/'trajectory.npz', allow_pickle=False) as trace:
                    assert len(trace['qpos']) == row['control_steps']+1
                    assert len(trace['predicted_joint_targets']) == row['control_steps']
                    assert len(trace['reference_indices']) == 0 and (trace['execution_horizons'] == 15).all()
                    assert all(np.isfinite(trace[k]).all() for k in trace.files)
            values[partition] = {**summary['overall'], 'by_route': summary['by_route'],
                'successes': sum(r['success'] for r in rows),
                'collisions': sum(r['collision'] is not None for r in rows),
                'timeouts': sum(not r['success'] and r['collision'] is None for r in rows)}
            comparisons[method][partition] = paired(read_json(reference/'full'/partition/'closed_loop.json')['results'], rows)
        models[method] = values
        del saved
    result = {'protocol': protocol, 'models': models, 'main': main_summary['models']['full'],
        'paired_comparisons': comparisons, 'audit': {'passed': True, 'baseline_rollouts': 800,
        'matched_split': True, 'matched_simulator': True, 'test_used_for_selection': False,
        'source_and_benchmark_hashes_verified': True, 'finite_weights_and_trajectories': True},
        'limitations': ['One training seed; bootstrap intervals resample scenes, not training seeds.',
            'DP3 and FlowPolicy receive live depth; GTSN, DP and CARP receive RGB.',
            'GTSN uses published Pi3 encoder initialization; baselines initialize from scratch.',
            'Baselines use compact reimplementations and TSN goal/action adaptations, not official benchmark reproductions.',
            'CARP has an additional 30-epoch train-only tokenizer fitting stage.',
            'Uniform point sampling, compact encoders/U-Net, two observations, joint-residual actions and TSN calibration conditioning are adaptations.']}
    write_json(root/'summary.json', result)
    labels = {'diffusion_policy': 'Diffusion Policy (RSS 2023/IJRR 2024)', 'dp3': 'DP3 (RSS 2024)',
              'flowpolicy': 'FlowPolicy (AAAI 2025)', 'carp': 'CARP (ICCV 2025)'}
    lines = ['# tsn-1k-var baseline comparison', '',
        'All four TSN adaptations completed training and all 100 validation plus 100 test closed-loop episodes. '
        'The archived GTSN full model uses identical scene partitions, recovery data and numerical simulator. '
        'Results below describe these adaptations and their stated training budgets.', '',
        '| Model | Input | Epoch | Val. RMSE (mm) | Val. success | Test success | Test collisions |',
        '|---|---|---:|---:|---:|---:|---:|']
    main_row = result['main']
    lines.append(f"| GTSN | RGB, pretrained Pi3 | {main_row['selected_epoch']} | {main_row['validation_waypoint_rmse_m']*1000:.2f} | "
        f"{main_row['partitions']['validation']['successes']}/100 | {main_row['partitions']['test']['successes']}/100 | "
        f"{main_row['partitions']['test']['collisions']} |")
    for method, row in models.items():
        lines.append(f"| {labels[method]} | {'Depth XYZ' if row['depth_input_at_inference'] else 'RGB'} | "
            f"{row['selected_epoch']} | {row['validation_waypoint_rmse_m']*1000:.2f} | {row['validation']['successes']}/100 | "
            f"{row['test']['successes']}/100 | {row['test']['collisions']} |")
    lines += ['', '| Baseline | Test direct (20) | Test over (40) | Test side (40) | XYZ + orientation | Main minus baseline (pp) | Paired 95% interval (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method, row in models.items():
        rates = ' | '.join(f"{row['test']['by_route'][route]['success_rate']*100:.1f}%" for route in ('direct', 'over', 'side'))
        comparison = comparisons[method]['test']
        lo, hi = comparison['paired_scene_bootstrap_95_interval_pp']
        lines.append(f"| {labels[method]} | {rates} | {row['test']['xyz_orientation_success_rate']*100:.0f}% | "
            f"{comparison['main_minus_baseline_pp']:+.0f} | [{lo:+.0f}, {hi:+.0f}] |")
    lines += ['', 'Training: 800 scenes, train-only existing independent perturbations, 65/35 expert/perturbation '
        'mixture, 20/40/40 route mixture, 30 epochs × 8,192 sampled chunks, batch size 8, AdamW 1e-4, '
        'seed 20261002. Minimum validation waypoint RMSE selects EMA checkpoints; the validation sampling seed '
        'is fixed across epochs. CARP first fits its independent-joint residual VQ tokenizer for 30 epochs '
        'on training chunks only. No test frames participate in caching, normalization, tokenization, training '
        'or checkpoint selection.', '',
        'Evaluation: live wrist observations and measured state, two causal observations, 30 future joint-residual '
        'targets, fixed 15-step execution, collision checks at 100 Hz, 400-control-step limit, collision-free '
        'XYZ reaching within 10 mm. Full XYZ plus 0.15-rad orientation success is reported separately. '
        'The main result is reused from its completed audited archive; it was not retrained.', '',
        '## Interpretation limits', '', *[f'- {line}' for line in result['limitations']], '',
        'Audit passed: all 800 baseline rollouts, complete 20/40/40 held-out partitions, disjoint training/recovery '
        'membership, validation-only selection, checkpoint receipts, finite weights and trajectories, exact '
        'simulator identity and frozen source/benchmark hashes.', '',
        'Artifacts: `source/`, `source_sha256.json`, `configs/`, `cache/`, `protocol.json`, `tests.log`, '
        'method logs, each method’s `training/best.pt`, `training/latest.pt`, `validation/`, `test/`, '
        '`summary.json`, `comparison.png` and `comparison.pdf`.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['GTSN', 'DP', 'DP3\n(depth)', 'FlowPolicy\n(depth)', 'CARP']
    validation = [main_row['partitions']['validation']['success_rate']]+[models[m]['validation']['success_rate'] for m in METHODS]
    test = [main_row['partitions']['test']['success_rate']]+[models[m]['test']['success_rate'] for m in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    x = np.arange(5)
    axes[0].bar(x-.17, np.array(validation)*100, width=.34, label='Validation')
    axes[0].bar(x+.17, np.array(test)*100, width=.34, label='Test')
    axes[0].set(xticks=x, xticklabels=names, ylabel='Collision-free XYZ success (%)', ylim=(0, 100))
    axes[0].legend()
    for method in METHODS:
        epochs = read_json(root/method/'training/epochs.json')
        axes[1].plot([r['epoch'] for r in epochs], [r['validation_rmse_m']*1000 for r in epochs], label=method)
    axes[1].set(xlabel='Epoch', ylabel='Validation waypoint RMSE (mm)')
    axes[1].legend(fontsize=8)
    for extension in ('png', 'pdf'):
        fig.savefig(root/f'comparison.{extension}', dpi=180)
    plt.close(fig)
    print({method: row['test'] for method, row in models.items()}, flush=True)


if __name__ == '__main__':
    main()
