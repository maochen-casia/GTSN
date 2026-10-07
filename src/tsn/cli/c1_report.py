"""Consolidate C1 outcomes, paired effects, memory-use audits and figures."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from tsn.cli.report import paired
from tsn.common.config import create_output, read_json, write_json


def memory_audit(directory, mode):
    rows, episode_usage = [], []
    for path in sorted((directory/'episodes').glob('*/policy_diagnostics.json')):
        episode = read_json(path)
        if not episode or any(b['step'] <= a['step'] for a, b in zip(episode, episode[1:])):
            raise ValueError('Missing diagnostics or noncausal timestamps')
        if episode[0]['route_feature_frames'] != 1 or episode[0]['old_only_points']:
            raise ValueError('Memory did not reset at an episode boundary')
        episode_usage.append(dict(episode_id=path.parent.name,
            reads_old=any(r['old_only_points'] > 0 for r in episode),
            old_changes_choice=any(r['old_changed_choice'] for r in episode)))
        rows.extend(episode)
    if len(episode_usage) != 100:
        raise ValueError('Expected diagnostics for all 100 episodes')
    if any(r['map_points'] > 1600 or r['persistent_geometry_bytes'] > 57600 for r in rows):
        raise ValueError('Persistent map exceeded its fixed bound')
    if mode == 'current' and any(r['route_feature_frames'] != 1 or r['points'] > 400 or r['map_points'] for r in rows):
        raise ValueError('Current-frame control retained earlier evidence')
    ages = [r['oldest_read_age_steps'] for r in rows]
    return dict(observations=len(rows), current_only_verified=mode == 'current',
        episodes_reading_old=sum(x['reads_old'] for x in episode_usage),
        episodes_with_old_changed_choice=sum(x['old_changes_choice'] for x in episode_usage),
        observations_reading_old=sum(r['old_only_points'] > 0 for r in rows),
        observations_with_old_changed_choice=sum(r['old_changed_choice'] for r in rows),
        maximum_read_age_steps=max(ages), maximum_map_points=max(r['map_points'] for r in rows),
        maximum_map_bytes=max(r['persistent_geometry_bytes'] for r in rows),
        maximum_visual_feature_frames=max(r['route_feature_frames'] for r in rows),
        mean_old_only_points=float(np.mean([r['old_only_points'] for r in rows])),
        maximum_old_only_points=max(r['old_only_points'] for r in rows),
        by_episode=episode_usage,
        bytes_scope='Persistent geometry tensors only; excludes recent clouds, visual features and temporary queries')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    args = parser.parse_args()
    create_output(args.output_dir)
    results, effects, audits, raw = {}, {}, {}, {}
    for partition in ('validation', 'test'):
        results[partition], effects[partition], audits[partition], raw[partition] = {}, {}, {}, {}
        for path in sorted((args.root/partition).glob('*/closed_loop.json')):
            mode, directory = path.parent.name, path.parent
            if not (directory/'complete.json').exists():
                raise ValueError('Incomplete condition: '+str(directory))
            data = read_json(path)
            if len(data['results']) != 100 or sorted(v['episodes'] for v in data['by_route'].values()) != [20, 40, 40]:
                raise ValueError('Full stratified partition required')
            for episode in data['results']:
                trace = np.load(directory/'episodes'/episode['episode_id']/'trajectory.npz')
                if len(trace['reference_indices']) or not all(np.isfinite(trace[k]).all() for k in trace.files):
                    raise ValueError('Nonfinite values or privileged expert progress')
                if not np.all(trace['execution_horizons'] == 15):
                    raise ValueError('Execution schedule changed')
            results[partition][mode] = dict(overall=data['overall'], by_route=data['by_route'],
                timeouts=sum(r['termination'] == 'timeout' for r in data['results']),
                mean_inference_ms=float(np.mean([r['mean_inference_ms'] for r in data['results']])),
                success_count=sum(r['success'] for r in data['results']))
            raw[partition][mode] = data
            audits[partition][mode] = memory_audit(directory, mode)
        for first in raw[partition]:
            for second in raw[partition]:
                if first != second:
                    effects[partition][first+'_minus_'+second] = paired(raw[partition][second], raw[partition][first])
    criteria = {}
    for partition in ('validation', 'test'):
        effect = effects[partition][args.candidate+'_minus_current']
        score = results[partition][args.candidate]['overall']['success_rate']
        criteria[partition] = dict(candidate_success=score,
            current_success=results[partition]['current']['overall']['success_rate'],
            gain_pp=effect['success_difference_pp'], at_least_3_pp=effect['success_difference_pp'] >= 3-1e-8,
            at_least_70_percent=score >= .70,
            significant_at_95_percent=effect['paired_stratified_95_ci_pp'][0] > 0)
    criteria['target_met'] = all(criteria[p]['at_least_3_pp'] and criteria[p]['at_least_70_percent']
                                 for p in ('validation', 'test'))
    write_json(args.output_dir/'summary.json', results)
    write_json(args.output_dir/'paired_effects.json', effects)
    write_json(args.output_dir/'memory_audit.json', audits)
    write_json(args.output_dir/'criteria.json', criteria)
    write_json(args.output_dir/'receipt.json', dict(candidate=args.candidate,
        protocol=read_json(args.root/'protocol.json'), test_freeze=read_json(args.root/'test_freeze.json'),
        summary_sha256={str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(args.root.glob('*/*/closed_loop.json'))},
        total_rollouts=sum(v['overall']['episodes'] for part in results.values() for v in part.values()),
        interpretation='Point-estimate target on the fixed, historically reused exploratory test set; inference-time removals'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions = ['current', 'recent_geometry' if args.candidate in ('persistent_geometry', 'unconfirmed_geometry') else 'recent', args.candidate]
    conditions = list(dict.fromkeys(c for c in conditions if c in results['test']))
    labels = {'current': 'Current frame only', 'recent': 'Four recent frames',
              'persistent': 'Supported persistent map', 'persistent_visual': 'Map + long visual history',
              'persistent_geometry': 'Persistent geometry; one visual frame',
              'recent_geometry': 'Four clouds; one visual frame',
              'unconfirmed': 'Anchored persistent map',
              'unconfirmed_geometry': 'Persistent map; one visual frame'}
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    x = np.arange(len(conditions))
    for offset, partition, color in ((-.18, 'validation', '#237a98'), (.18, 'test', '#d4773c')):
        scores = [results[partition][c]['success_count'] for c in conditions]
        bars = ax.bar(x+offset, scores, width=.34, label=partition.title(), color=color)
        ax.bar_label(bars, labels=[f'{s}/100' for s in scores], padding=3)
    ax.axhline(70, color='#888888', linestyle=':', linewidth=1)
    ax.set(ylim=(0, 105), ylabel='Collision-free XYZ success (%)', title='C1: memory versus a strict current-frame control')
    ax.set_xticks(x, [labels.get(c, c) for c in conditions])
    ax.legend()
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output_dir/f'c1_success.{suffix}', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for row, partition in enumerate(('validation', 'test')):
        effect = effects[partition][args.candidate+'_minus_current']
        center = effect['success_difference_pp']
        low, high = effect['paired_stratified_95_ci_pp']
        axes[0].errorbar(center, row, xerr=[[center-low], [high-center]], fmt='o', capsize=5, color='#237a98')
        axes[0].annotate(f" {center:+.0f} pp; {effect['wins']} gained, {effect['losses']} lost", (high, row), fontsize=9)
        audit = audits[partition][args.candidate]
        axes[1].bar(row, audit['maximum_read_age_steps'], color=['#237a98', '#d4773c'][row])
    axes[0].axvline(0, color='#888888', linestyle='--')
    axes[0].axvline(3, color='#35895b', linestyle=':', label='Requested gain')
    axes[0].set_yticks([0, 1], ['Validation', 'Test'])
    axes[0].set(xlabel='Paired success difference (percentage points)', title='95% route-stratified bootstrap')
    lo, hi = axes[0].get_xlim()
    axes[0].set_xlim(lo, hi+18)
    axes[0].legend(fontsize=8)
    axes[1].axhline(45, color='#888888', linestyle=':', label='Old four-frame maximum age')
    axes[1].set_xticks([0, 1], ['Validation', 'Test'])
    axes[1].set(ylabel='Maximum queried geometry age (control steps)', title='Measured persistent geometry use')
    axes[1].legend(fontsize=8)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output_dir/f'c1_effects_memory.{suffix}', dpi=200)
    plt.close(fig)
    write_json(args.output_dir/'complete.json', dict(candidate=args.candidate, criteria=criteria,
                                                  finite_traces=True, no_expert_progress=True, fixed_execution=True))
    print(criteria, flush=True)


if __name__ == '__main__':
    main()
