"""Summarize completed GTSN runs with paired validation uncertainty and figures."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tsn.common.config import read_json, write_json
from tsn.evaluation.map_diagnostics import episode_bootstrap
from tsn.models.gtsn_policy import MODES


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [float(center-radius), float(center+radius)]


def paired_difference(a, b, seed=20261002, metric='success'):
    by_id = {x['episode_id']: x for x in b}
    assert set(by_id) == {x['episode_id'] for x in a}
    differences, groups = [], []
    for route in ('direct', 'over', 'side'):
        group = [x for x in a if x['route'] == route]
        def value(item):
            return int(item['collision'] is not None) if metric == 'collision' else int(item['success'])
        groups.append(np.array([value(x)-value(by_id[x['episode_id']]) for x in group]))
        differences.extend(groups[-1])
    rng = np.random.default_rng(seed)
    samples = sum(g[rng.integers(0, len(g), (5000, len(g)))].sum(1) for g in groups) / len(a)
    return {f'{metric}_rate_difference': float(np.mean(differences)),
            'paired_stratified_bootstrap_95': np.quantile(samples, [.025, .975]).tolist(),
            'bootstrap_samples': 5000, 'multiple_comparison_adjustment': False}


def sensing_summary(directory, results):
    horizons = []
    for item in results:
        with np.load(directory / 'episodes' / item['episode_id'] / 'trajectory.npz') as trajectory:
            horizons.extend(trajectory['execution_horizons'].tolist())
    return {'observation_calls': len(horizons), 'fraction_five_step_requests': float(np.mean(np.array(horizons) == 5)),
            'mean_requested_execution_horizon': float(np.mean(horizons)),
            'mean_replans_per_episode': float(np.mean([x['replans'] for x in results]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    assert read_json(root / 'status.json')['state'] == 'complete'
    assert read_json(root / 'audit.json')['passed']
    diagnostic_script = Path(__file__).with_name('diagnose_gtsn_uncertainty.py')
    if not (root / 'uncertainty_diagnostics.json').exists():
        subprocess.run([sys.executable, str(diagnostic_script), '--root', str(root)], check=True)
    appearance_script = Path(__file__).with_name('diagnose_gtsn_domain.py')
    if not (root / 'appearance_diagnostics.json').exists():
        subprocess.run([sys.executable, str(appearance_script), '--root', str(root)], check=True)
    selection = read_json(root / 'selection.json')
    selected = selection['selected']
    heads = {mode: read_json(root / 'heads' / mode / 'results.json') for mode in MODES}
    validation = {p.parent.name: read_json(p) for p in (root / 'rollouts/validation').glob('*/closed_loop.json')}
    test = read_json(root / 'rollouts/test' / selected / 'closed_loop.json')
    test_open = read_json(root / 'rollouts/test' / selected / 'open_loop.json')
    baseline_dir = Path(read_json(root / 'protocol.json')['backbone_checkpoint']).parent.parent / 'test'
    original_test = read_json(baseline_dir / 'closed_loop.json')
    pairs = [('visual_fixed15', 'state_correction_fixed15'), ('uncertainty_fixed15', 'visual_fixed15'),
             ('memory_fixed15', 'uncertainty_fixed15'), ('memory_adaptive', 'memory_fixed15'),
             ('memory_adaptive', 'memory_fixed5')]
    comparisons = {a+' minus '+b: paired_difference(validation[a]['results'], validation[b]['results']) for a, b in pairs}
    collision_comparisons = {a+' minus '+b: paired_difference(validation[a]['results'], validation[b]['results'], metric='collision')
                            for a, b in pairs}
    episode = np.load(root / 'cache/validation/episode.npy')
    route = np.load(root / 'cache/validation/route.npy')
    _, first = np.unique(episode, return_index=True)
    episode_routes = route[first]
    counts = np.bincount(episode)
    squared = {mode: np.bincount(episode, weights=np.load(root / 'heads' / mode / 'validation_frame_mse.npy'))
               for mode in MODES}
    open_pairs = [('visual', 'state_correction'), ('uncertainty', 'visual'), ('memory', 'uncertainty')]
    open_comparisons = {a+' minus '+b: episode_bootstrap(squared[a], counts, squared[b], episode_routes)
                        for a, b in open_pairs}
    successes = sum(x['success'] for x in test['results'])
    sensing = {name: sensing_summary(root / 'rollouts/validation' / name, result['results'])
               for name, result in validation.items()}
    summary = {'selected': selected, 'heads': heads, 'validation': selection['validation'],
               'paired_validation_comparisons': comparisons, 'validation_sensing': sensing,
               'paired_validation_collisions': collision_comparisons,
               'paired_validation_open_loop': open_comparisons,
               'uncertainty_diagnostics': read_json(root / 'uncertainty_diagnostics.json'),
               'appearance_diagnostics': read_json(root / 'appearance_diagnostics.json'),
               'original_pi3_test': original_test['overall'],
               'test_vs_original_pi3': paired_difference(test['results'], original_test['results']),
               'test_collisions_vs_original_pi3': paired_difference(test['results'], original_test['results'], metric='collision'),
               'test': {**test['overall'], 'by_route': test['by_route'],
               'success_wilson_95': wilson(successes, len(test['results'])), 'open_loop': test_open,
               'sensing': sensing_summary(root / 'rollouts/test' / selected, test['results'])},
               'report_source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (Path(__file__), diagnostic_script, appearance_script)},
               'interpretation_limits': ['One training seed', 'Frozen perception pilot',
                   'Active timing only; no active viewpoint search', 'Intervals do not include training-seed uncertainty',
                   'Prior baseline test results were already inspected', 'State correction retains the visual baseline']}
    write_json(root / 'results.json', summary)
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for mode in MODES:
        records = read_json(root / 'heads' / mode / 'epochs.json')
        ax.plot([0]+[x['epoch'] for x in records],
                [heads[mode]['initial_validation']['rmse_rad']]+[x['validation']['rmse_rad'] for x in records],
                marker='.', label=mode.replace('_', ' '))
    ax.set(xlabel='Training epoch', ylabel='Validation joint RMSE (rad)')
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(figures / f'training.{extension}', dpi=180)
    plt.close(fig)
    names = list(selection['validation'])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    success = [selection['validation'][n]['success_rate'] for n in names]
    collision = [selection['validation'][n]['collision_rate'] for n in names]
    x = np.arange(len(names))
    success_bars = ax.bar(x-.18, success, .36, label='Success')
    collision_bars = ax.bar(x+.18, collision, .36, label='Collision')
    ax.bar_label(success_bars, labels=[f'{100*v:.0f}%' for v in success], padding=3, fontsize=8)
    ax.bar_label(collision_bars, labels=[f'{100*v:.0f}%' for v in collision], padding=3, fontsize=8)
    ax.set_xticks(x, [n.replace('_', '\n') for n in names], fontsize=8)
    ax.set(ylabel='Episode fraction', ylim=(0, 1), title='100 validation episodes per condition')
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(figures / f'validation_rollouts.{extension}', dpi=180)
    plt.close(fig)
    print(f'Saved {root / "results.json"}; selected {selected}, test successes {successes}/100')


if __name__ == '__main__':
    main()
