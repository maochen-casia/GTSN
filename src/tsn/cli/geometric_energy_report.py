"""Consolidate the complete geometric-energy study and export paper figures."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

from tsn.cli.report import paired
from tsn.common.config import create_output, read_json, write_json


CONDITIONS = ('full', 'no_history', 'tcp_only', 'no_trust', 'fixed_trust')
LABELS = ('Full model', 'No geometric history', 'TCP queries only', 'No route preservation', 'Fixed trust (baseline)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    args = parser.parse_args()
    create_output(args.output_dir)
    data, receipts = {}, {}
    for partition in ('validation', 'test'):
        data[partition] = {}
        audit = read_json(args.root/'reports'/partition/'audit.json')
        if set(audit) != set(CONDITIONS):
            raise ValueError('Missing audited condition')
        for condition in CONDITIONS:
            directory = args.root/partition/condition
            if not (directory/'complete.json').exists():
                raise ValueError('Incomplete rollout condition')
            result = read_json(directory/'closed_loop.json')
            if Counter(x['route'] for x in result['results']) != dict(direct=20, over=40, side=40):
                raise ValueError('Incorrect route proportions')
            if len(set(x['episode_id'] for x in result['results'])) != 100:
                raise ValueError('Expected 100 distinct episodes')
            if result['privileged_action_map']:
                raise ValueError('Privileged deployment input')
            data[partition][condition] = result
            receipts[partition+'/'+condition] = hashlib.sha256((directory/'closed_loop.json').read_bytes()).hexdigest()
    effects = {p: {c: paired(data[p][c], data[p]['full']) for c in CONDITIONS if c != 'full'} for p in data}
    criteria = dict(
        overall_target_met=all(data[p]['full']['overall']['success_rate'] >= .70 for p in data),
        each_component_has_observed_success_gain=all(
            any(effects[p][c]['success_difference_pp'] > 0 for p in data)
            for c in ('no_history', 'tcp_only', 'no_trust')),
        all_three_improve_validation=all(effects['validation'][c]['success_difference_pp'] > 0
            for c in ('no_history', 'tcp_only', 'no_trust')),
        inference_ablation_note='Same trained checkpoint; component removal without retraining',
        statistical_note='Observed gains; paired confidence intervals and uncorrected exact McNemar p values are descriptive, not three decisive independent effects',
        new_rollouts=1000, new_training_epochs=20, test_used_for_checkpoint_selection=False)
    summary = {p: {c: dict(overall=data[p][c]['overall'], by_route=data[p][c]['by_route'],
        timeouts=sum(x['termination']=='timeout' for x in data[p][c]['results'])) for c in CONDITIONS} for p in data}
    # Baseline evidence is historical and explicitly identified as reused.
    compact = {}
    base = Path('/run/user/1016/experiments/gtsn_simplification_20261003/rollouts')
    for p in data:
        paths = list((base/p).glob('*/closed_loop.json'))
        path = next((x for x in paths if x.parent.name == 'single_memory'), None)
        if path is not None:
            original = read_json(path)
            compact[p] = dict(source=str(path), reused=True, overall=original['overall'],
                              full_minus_compact=paired(original, data[p]['full']))
    write_json(args.output_dir/'summary.json', summary)
    write_json(args.output_dir/'effects.json', effects)
    write_json(args.output_dir/'criteria.json', criteria)
    write_json(args.output_dir/'historical_compact.json', compact)
    write_json(args.output_dir/'result_receipts.json', receipts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for ax, p in zip(axes, data):
        values = [summary[p][c]['overall']['success_rate']*100 for c in CONDITIONS]
        bars = ax.barh(LABELS, values, color=['#167d8d']+['#99b5bf']*3+['#c28454'])
        ax.invert_yaxis(); ax.set(xlim=(0, 105), xlabel='Collision-free XYZ success (%)', title=p.capitalize()+' · 100 episodes')
        for bar, value in zip(bars, values):
            ax.text(value+1, bar.get_y()+bar.get_height()/2, str(round(value)), va='center')
    for ext in ('png', 'pdf'):
        fig.savefig(args.output_dir/f'ablation_success.{ext}', dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    names = ('no_history', 'tcp_only', 'no_trust')
    for i, (p, color) in enumerate((('validation', '#167d8d'), ('test', '#c28454'))):
        for row, c in enumerate(names):
            effect = effects[p][c]
            center = effect['success_difference_pp']
            low, high = effect['paired_stratified_95_ci_pp']
            ax.errorbar(center, row+(i-.5)*.22, xerr=[[center-low], [high-center]], fmt='o',
                        capsize=4, color=color, label=p.capitalize() if row == 0 else None)
    ax.axvline(0, color='#888', ls='--')
    ax.set_yticks(range(3), ['C1: surface persistence', 'C2: swept-hand queries', 'C3: route preservation'])
    ax.invert_yaxis(); ax.set(xlabel='Full model minus removal (percentage points)',
        title='Paired route-stratified 95% bootstrap intervals'); ax.legend()
    for ext in ('png', 'pdf'):
        fig.savefig(args.output_dir/f'contribution_effects.{ext}', dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    ax.set(xlim=(0, 12), ylim=(0, 4)); ax.axis('off')
    from matplotlib.patches import FancyBboxPatch
    def box(x, y, width, text, color='#e7eff2'):
        ax.add_patch(FancyBboxPatch((x, y), width, .8, boxstyle='round,pad=.08', facecolor=color, edgecolor='#396478'))
        ax.text(x+width/2, y+.4, text, ha='center', va='center', fontsize=10)
    def arrow(a, b):
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color='#396478', lw=1.5))
    box(.1, 1.6, 1.2, 'RGB +\nrobot state')
    box(1.8, 1.6, 1.6, 'Frozen metric\nperception')
    box(4., 2.7, 2., 'C1: four-cloud\nsurface memory')
    box(4., .5, 2., 'Frozen route +\n14 candidates')
    box(6.7, 2.7, 2., 'C2: swept-hand\nproximity energy')
    box(6.7, .5, 2., 'C3: learned\npositive route trust', '#f4eadf')
    box(9.4, 1.6, 2.2, 'Minimum cost → IK\nExecute 15 steps')
    arrow((1.3, 2), (1.8, 2)); arrow((3.4, 2), (4., 3.1)); arrow((3.4, 2), (4., .9))
    arrow((6., 3.1), (6.7, 3.1)); arrow((6., .9), (6.7, 2.7)); arrow((6., .9), (6.7, .9))
    arrow((7.7, 2.7), (7.7, 1.3)); arrow((8.7, 3.1), (9.4, 2.1)); arrow((8.7, .9), (9.4, 1.9))
    ax.text(6, .05, 'Training: existing expert + perturbation-only recovery · Deployment: RGB-predicted geometry only', ha='center', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(args.output_dir/f'architecture.{ext}', dpi=220)
    plt.close(fig)
    write_json(args.output_dir/'complete.json', dict(complete=True, criteria=criteria, source='Existing completed rollouts; no new model selection'))
    print(criteria, flush=True)


if __name__ == '__main__':
    main()
