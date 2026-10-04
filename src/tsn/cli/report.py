"""Paired episode statistics, completion audit, and exportable research figures."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from tsn.common.config import create_output, read_json, write_json


def paired(reference, candidate, seed=20261003):
    baseline = {x['episode_id']: x for x in reference['results']}
    results = {x['episode_id']: x for x in candidate['results']}
    if baseline.keys() != results.keys():
        raise ValueError('Paired comparisons require exactly the same episode IDs')
    groups = [[ep for ep in baseline if baseline[ep]['route'] == route] for route in ('direct', 'over', 'side')]
    generator = np.random.default_rng(seed)
    draws = np.zeros(20000)
    differences = []
    for group in groups:
        values = np.array([int(results[ep]['success'])-int(baseline[ep]['success']) for ep in group])
        differences.extend(values.tolist())
        draws += values[generator.integers(0, len(values), size=(20000, len(values)))].sum(1)
    differences = np.asarray(differences)
    wins, losses = int((differences > 0).sum()), int((differences < 0).sum())
    return dict(success_difference_pp=float(differences.mean()*100),
                paired_stratified_95_ci_pp=(np.quantile(draws/len(baseline), [.025, .975])*100).tolist(),
                wins=wins, losses=losses,
                exact_mcnemar_p=float(binomtest(wins, wins+losses).pvalue) if wins+losses else 1.)


def diagnostics(directory):
    observations = []
    for path in sorted((directory/'episodes').glob('*/policy_diagnostics.json')):
        observations.extend(read_json(path))
    if not observations:
        return None
    summary = dict(observations=len(observations),
                corrected_fraction=float(np.mean([x['choice'] != 0 and x['correction_m'] > 0 for x in observations])),
                mean_correction_m=float(np.mean([x['correction_m'] for x in observations])),
                mean_initial_risk=float(np.mean([x['risk'] for x in observations])),
                mean_retained_points=float(np.mean([x['points'] for x in observations])))
    if 'removed_history_points' in observations[0]:
        removed = sum(x['removed_history_points'] for x in observations)
        checked = sum(x['historical_valid_before'] for x in observations)
        summary.update(removed_history_point_checks=removed, historical_valid_point_checks=checked,
                       removed_fraction_of_history_checks=removed/checked if checked else None,
                       mean_removed_history_points_per_observation=removed/len(observations))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=('cpu',))
    parser.add_argument('--candidate', help='Condition to illustrate in paired trajectory figures')
    args = parser.parse_args()
    create_output(args.output_dir)
    reference = read_json(args.reference/'closed_loop.json')
    results, raw, audits = {}, {}, {}
    for path in sorted(args.root.glob('*/closed_loop.json')):
        directory = path.parent
        if not (directory/'complete.json').is_file():
            raise ValueError(f'Incomplete evaluation: {directory}')
        condition = read_json(path)
        episode_ids = [x['episode_id'] for x in condition['results']]
        if len(episode_ids) != 100 or len(set(episode_ids)) != 100:
            raise ValueError(f'Expected one full split: {directory}')
        for episode in condition['results']:
            trace = np.load(directory/'episodes'/episode['episode_id']/'trajectory.npz')
            if len(trace['reference_indices']) or not all(np.isfinite(trace[k]).all() for k in trace.files):
                raise ValueError('Nonfinite trace or privileged expert progress')
            if not np.all(trace['execution_horizons'] == 15):
                raise ValueError('Execution schedule changed')
        results[directory.name] = dict(overall=condition['overall'], by_route=condition['by_route'],
                paired_vs_reference=paired(reference, condition), diagnostics=diagnostics(directory),
                config=read_json(directory/'config.json'))
        raw[directory.name] = condition
        audits[directory.name] = dict(episodes=100, finite_trajectories=True, no_expert_progress=True,
                fixed_execution=True, summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(args.output_dir/'results.json', results)
    write_json(args.output_dir/'audit.json', audits)
    write_json(args.output_dir/'paired_comparisons.json', {
        f'{first}_minus_{second}': paired(raw[second], raw[first])
        for first in raw for second in raw if first != second})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(results)
    friendly = {'baseline': 'Compact baseline', 'method': 'Route-preserving clearance',
                'without_correction_penalty': 'No correction penalty',
                'without_hand_extent': 'TCP only',
                'without_surface_memory': 'Current surfaces only'}
    friendly.update(visibility_004='Visibility update (4 cm)', visibility_008='Visibility update (8 cm)',
                    shift_pos='Geometry shifted +10 cm', shift_neg='Geometry shifted −10 cm',
                    calibrated='Training bias calibration', pooled_mean='Learned mean correction')
    labels = [friendly.get(name, name) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].barh(labels, [results[x]['overall']['success_rate']*100 for x in names], color='#237a98')
    axes[0].axvline(reference['overall']['success_rate']*100, color='#d4773c', ls='--', label='Compact baseline')
    axes[0].set(xlim=(0, 100), xlabel='Success (%)', title='Fixed 100-episode split')
    axes[0].legend()
    x = np.arange(len(names))
    for j, route in enumerate(('direct', 'over', 'side')):
        axes[1].bar(x+(j-1)*.25, [results[n]['by_route'][route]['success_rate']*100 for n in names], width=.25, label=route)
    axes[1].set(ylim=(0, 105), ylabel='Success (%)', title='Route breakdown')
    axes[1].set_xticks(x, labels, rotation=35, ha='right')
    axes[1].legend()
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output_dir/f'closed_loop.{suffix}', dpi=180)
    plt.close(fig)
    compared = [name for name in names if (args.root/name).resolve() != args.reference.resolve()]
    if compared:
        fig, ax = plt.subplots(figsize=(9, max(3., .65*len(compared))), constrained_layout=True)
        for row, name in enumerate(compared):
            effect = results[name]['paired_vs_reference']
            center = effect['success_difference_pp']
            low, high = effect['paired_stratified_95_ci_pp']
            ax.errorbar(center, row, xerr=[[center-low], [high-center]],
                        fmt='o', color='#237a98', capsize=4)
            ax.annotate(f" {center:+.0f} pp; {effect['wins']} gained / {effect['losses']} lost",
                        (high, row), xytext=(5, 0), textcoords='offset points', va='center', fontsize=9)
        ax.axvline(0, color='#888888', ls='--')
        ax.set_yticks(range(len(compared)), [friendly.get(name, name) for name in compared])
        ax.set(xlabel='Success difference from baseline (percentage points)',
               title='Paired 95% bootstrap intervals: 100 shared episodes')
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi+max(10, (hi-lo)*.55))
        for suffix in ('png', 'pdf'):
            fig.savefig(args.output_dir/f'paired_effects.{suffix}', dpi=180)
        plt.close(fig)
    candidate_name = args.candidate or next((name for name in ('method', 'full', 'full_p008') if name in raw), None)
    if candidate_name and candidate_name not in raw:
        raise ValueError(f'Unknown paired-figure candidate: {candidate_name}')
    if candidate_name:
        from matplotlib.patches import Polygon
        original = {x['episode_id']: x for x in reference['results']}
        chosen = {x['episode_id']: x for x in raw[candidate_name]['results']}
        cases = []
        for label, gain in [('First gained episode', True), ('First lost episode', False)]:
            ids = sorted(ep for ep in chosen if chosen[ep]['success'] == gain and original[ep]['success'] != gain)
            if ids:
                cases.append((label, ids[0]))
        if cases:
            dataset_root = Path(results[candidate_name]['config']['dataset_root'])
            fig, axes = plt.subplots(len(cases), 2, figsize=(10, 4*len(cases)), squeeze=False, constrained_layout=True)
            for row, (label, ep) in enumerate(cases):
                scene = read_json(dataset_root/ep/'scene.json')
                main = next(item for item in scene['objects'] if item.get('main'))
                center, half = np.asarray(main['center']), np.asarray(main['half_size'])
                yaw = main.get('yaw', 0.)
                rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*half[:2]
                polygon = square@rotation.T+center[:2]
                axes[row, 0].add_patch(Polygon(polygon, facecolor='#cccccc', edgecolor='#777777', label='Obstacle envelope'))
                lo, hi = polygon[:, 1].min(), polygon[:, 1].max()
                axes[row, 1].add_patch(Polygon([[lo, center[2]-half[2]], [hi, center[2]-half[2]],
                                              [hi, center[2]+half[2]], [lo, center[2]+half[2]]],
                                             facecolor='#cccccc', edgecolor='#777777'))
                for directory, name, color in [(args.reference, 'Baseline', '#cb6843'),
                                               (args.root/candidate_name, 'Clearance', '#237a98')]:
                    trace = np.load(directory/'episodes'/ep/'trajectory.npz')['T_B_E']
                    tcp, hand = trace[:, :3, 3], trace[:, :3, 3]-.05*trace[:, :3, 2]
                    for column, (a, b) in enumerate(((0, 1), (1, 2))):
                        axes[row, column].plot(tcp[:, a], tcp[:, b], color=color, lw=1., ls=':', label=name+' TCP')
                        axes[row, column].plot(hand[:, a], hand[:, b], color=color, lw=1.8, label=name+' hand center')
                        axes[row, column].scatter([scene['goal'][a]], [scene['goal'][b]], marker='*', color='#35895b', s=70)
                for column, names_xy in enumerate((('x (m)', 'y (m)'), ('y (m)', 'z (m)'))):
                    axes[row, column].set(xlabel=names_xy[0], ylabel=names_xy[1], title=f'{label}: {ep}')
                    axes[row, column].set_aspect('equal', adjustable='datalim')
                axes[row, 0].legend(fontsize=8)
            for suffix in ('png', 'pdf'):
                fig.savefig(args.output_dir/f'paired_cases.{suffix}', dpi=180)
            plt.close(fig)
            write_json(args.output_dir/'case_selection.json', dict(cases=cases,
                       rule='lexicographically first gained and first lost episode; no model selection',
                       obstacle_geometry='Ground-truth envelopes used only in this offline figure, never by the policy'))
    print({name: value['paired_vs_reference'] for name, value in results.items()}, flush=True)


if __name__ == '__main__':
    main()
