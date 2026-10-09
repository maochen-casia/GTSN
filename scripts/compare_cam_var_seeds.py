"""Compare audited Full and w/o C1 pairs across matched training seeds."""
import argparse
import copy
from pathlib import Path

import numpy as np

from tsn.common.config import read_json, write_json
from report_sim2real import sha256

VARIANTS = ('full', 'without_c1')


def descriptive(values):
    return dict(by_seed=values, mean=float(np.mean(values)),
                sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)


def compare(roots):
    summaries = [read_json(root/'summary.json') for root in roots]
    seeds = [summary['protocol']['seed'] for summary in summaries]
    assert len(seeds) == len(set(seeds)) and len(seeds) >= 2, 'Require distinct training seeds'
    benchmark = read_json(roots[0]/'benchmark_sha256.json')
    sources = read_json(roots[0]/'source_sha256.json')
    numerical = {name:value for name,value in sources.items()
                 if name.startswith(('src/', 'vendor/', 'assets/')) or name in ('Dockerfile', 'pyproject.toml')}
    reference_configs = {variant:read_json(roots[0]/'configs'/f'{variant}.json') for variant in VARIANTS}
    splits = read_json(roots[0]/'recovery/source_split.json')
    recovery_root = Path(reference_configs['full']['train']['recovery_root']).resolve()
    recovery_hashes = None
    records = []
    for root,summary,seed in zip(roots, summaries, seeds):
        audit = summary['audit']
        assert audit.get('all_rollouts_complete', audit.get('all_800_rollouts_complete', False))
        assert audit['source_hashes_verified'] and audit['benchmark_contents_verified']
        assert audit['complete_disjoint_partitions'] and audit['finite_weights_and_trajectories']
        assert audit['all_pi3_encoders_fully_tuned'] and not summary['test_used_for_selection']
        assert read_json(root/'benchmark_sha256.json') == benchmark
        other_sources = read_json(root/'source_sha256.json')
        other_numerical = {name:value for name,value in other_sources.items()
                          if name.startswith(('src/', 'vendor/', 'assets/')) or name in ('Dockerfile', 'pyproject.toml')}
        assert other_numerical == numerical
        assert read_json(root/'recovery/source_split.json') == splits
        if (root/'recovery_sha256.json').exists():
            hashes = read_json(root/'recovery_sha256.json')
            if recovery_hashes is not None:
                assert hashes == recovery_hashes
            recovery_hashes = hashes
        for variant in VARIANTS:
            config = copy.deepcopy(read_json(root/'configs'/f'{variant}.json'))
            assert config['train']['seed'] == seed
            assert Path(config['train']['recovery_root']).resolve() == recovery_root
            config['train']['seed'] = reference_configs[variant]['train']['seed']
            assert config == reference_configs[variant], ('Only training seed may differ', root, variant)
            for partition in ('validation', 'test'):
                metrics = summary['models'][variant]['partitions'][partition]
                assert len(read_json(root/variant/partition/'closed_loop.json')['results']) == 100
                assert np.isclose(metrics['success_rate'], metrics['successes']/100)
        records.append(dict(seed=seed, root=str(root), summary_sha256=sha256(root/'summary.json'),
                            paired_comparison=summary['paired_comparisons']['without_c1']))
    models = {variant:{partition:descriptive([
        summary['models'][variant]['partitions'][partition]['success_rate']*100
        for summary in summaries]) for partition in ('validation', 'test')} for variant in VARIANTS}
    gains = {partition:descriptive([a-b for a,b in zip(models['full'][partition]['by_seed'],
                models['without_c1'][partition]['by_seed'])]) for partition in ('validation', 'test')}
    return dict(seeds=seeds, records=records, models=models, full_minus_c1_pp=gains,
                episodes_per_partition_per_seed=100, audit=dict(matched_settings=True,
                matched_numerical_source=True, matched_benchmark=True, matched_splits=True,
                shared_recovery=True, all_input_experiments_audited=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.roots)
    write_json(args.output_dir/'seed_comparison.json', result)
    lines = ['# Full versus w/o C1 across training seeds', '',
             'The Panda benchmark, split, recovery pool, numerical code and training/evaluation settings '
             'match across all runs. Each model trains for 30 epochs with the Pi3 encoder fully tuned. '
             'Validation waypoint RMSE selects each checkpoint before test evaluation.', '',
             '| Training seed | Full test success | w/o C1 test success | Full minus w/o C1 (pp) |',
             '|---|---:|---:|---:|']
    for i,seed in enumerate(result['seeds']):
        full = result['models']['full']['test']['by_seed'][i]
        ablated = result['models']['without_c1']['test']['by_seed'][i]
        lines.append(f'| {seed} | {full:.0f}% | {ablated:.0f}% | {full-ablated:+.0f} |')
    lines += ['', '| Model | Validation success, mean ± SD | Test success, mean ± SD |',
              '|---|---:|---:|']
    for variant,label in zip(VARIANTS, ('Full', 'w/o C1')):
        val,test = (result['models'][variant][partition] for partition in ('validation', 'test'))
        lines.append(f"| {label} | {val['mean']:.1f} ± {val['sample_sd']:.1f}% | "
                     f"{test['mean']:.1f} ± {test['sample_sd']:.1f}% |")
    gain = result['full_minus_c1_pp']['test']
    lines += ['', f"The paired test success difference across seeds is {gain['mean']:+.1f} ± "
              f"{gain['sample_sd']:.1f} percentage points (mean ± sample SD). "
              f"These are descriptive results across {len(result['seeds'])} training seeds. "
              'Each uses the same 100 test scenes; repeated evaluations are not additional independent scenes. '
              'Per-seed paired scene bootstrap intervals remain in each experiment report and the JSON records.', '',
              'All input experiments passed their source/data, split, checkpoint and rollout audits. '
              'This comparison additionally checks matching numerical source, benchmark fingerprints, '
              'recovery provenance, scene splits, settings and distinct seeds.', '', 'Experiment directories:']
    lines.extend(f'- `{record["root"]}`' for record in result['records'])
    (args.output_dir/'SEED_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(dict(seeds=result['seeds'], test_difference_pp=gain), flush=True)


if __name__ == '__main__':
    main()
