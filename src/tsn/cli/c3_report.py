"""Summarize uncertainty-clearance outcomes, calibration and action diagnostics."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from tsn.cli.report import paired
from tsn.common.config import create_output, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    args = parser.parse_args()
    create_output(args.output_dir)
    results, effects, audits, raw = {}, {}, {}, {}
    for partition in ('validation','test'):
        results[partition], effects[partition], audits[partition], raw[partition] = {}, {}, {}, {}
        for path in sorted((args.root/partition).glob('*/closed_loop.json')):
            condition, directory = path.parent.name, path.parent
            if not (directory/'complete.json').exists():
                raise ValueError('Incomplete rollout condition')
            data = read_json(path)
            if len(data['results']) != 100 or sorted(v['episodes'] for v in data['by_route'].values()) != [20,40,40]:
                raise ValueError('Require the full fixed stratified split')
            diagnostics = []
            for episode in data['results']:
                trace = np.load(directory/'episodes'/episode['episode_id']/'trajectory.npz')
                if len(trace['reference_indices']) or not all(np.isfinite(trace[k]).all() for k in trace.files):
                    raise ValueError('Nonfinite trajectory or privileged expert progress')
                if not np.all(trace['execution_horizons']==15):
                    raise ValueError('Execution schedule changed')
                rows = read_json(directory/'episodes'/episode['episode_id']/'policy_diagnostics.json')
                if any(r.get('route_feature_frames',4)>4 for r in rows):
                    raise ValueError('Proposal history changed')
                if condition == 'none' and any(r['correction_m'] != 0 or r['choice'] != 0 for r in rows):
                    raise ValueError('No-refinement control applied corrections')
                diagnostics.extend(rows)
            raw[partition][condition] = data
            results[partition][condition] = dict(overall=data['overall'],by_route=data['by_route'],
                successes=sum(r['success'] for r in data['results']),
                timeouts=sum(r['termination']=='timeout' for r in data['results']),
                mean_correction_m=float(np.mean([r['correction_m'] for r in diagnostics])),
                mean_padding_m=float(np.mean([r.get('mean_padding_m',0.) for r in diagnostics])),
                maximum_padding_m=max(r.get('max_padding_m',0.) for r in diagnostics),
                mean_predicted_error_m=float(np.mean([r['mean_predicted_error_m'] for r in diagnostics if 'mean_predicted_error_m' in r])) if any('mean_predicted_error_m' in r for r in diagnostics) else None)
            audits[partition][condition] = dict(episodes=100, finite_trajectories=True,
                no_expert_progress=True,fixed_execution=True,no_refinement_verified=condition=='none',
                summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for first in raw[partition]:
            for second in raw[partition]:
                if first != second:
                    effects[partition][first+'_minus_'+second] = paired(raw[partition][second],raw[partition][first])
    criteria = {}
    for partition in ('validation','test'):
        effect = effects[partition][args.candidate+'_minus_none']
        score = results[partition][args.candidate]['overall']['success_rate']
        criteria[partition] = dict(candidate_success=score,no_refinement_success=results[partition]['none']['overall']['success_rate'],
            gain_pp=effect['success_difference_pp'],at_least_3_pp=effect['success_difference_pp']>=3-1e-8,
            at_least_70_percent=score>=.70,paired_95_ci_pp=effect['paired_stratified_95_ci_pp'])
    criteria['target_met'] = all(criteria[p]['at_least_3_pp'] and criteria[p]['at_least_70_percent'] for p in ('validation','test'))
    records = read_json(args.root/'training/epochs.json')
    calibration = dict(initial=records[0]['validation'],final=records[-1]['validation'],
                       training=read_json(args.root/'training/train_calibration.json'))
    write_json(args.output_dir/'summary.json',results)
    write_json(args.output_dir/'paired_effects.json',effects)
    write_json(args.output_dir/'audit.json',audits)
    write_json(args.output_dir/'criteria.json',criteria)
    write_json(args.output_dir/'calibration.json',calibration)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    uniform = 'uniform'+args.candidate.removeprefix('adaptive')
    names = [n for n in ('none','fixed',uniform,args.candidate) if n in results['test']]
    names = list(dict.fromkeys(names))
    labels = dict(none='No refinement',fixed='C1 fixed clearance',uniform10='Uniform inflation',
                  uniform20='Uniform inflation',uniform30='Uniform inflation',
                  adaptive10='Uncertainty: ≤10 mm',adaptive20='Uncertainty: ≤20 mm',adaptive30='Uncertainty: ≤30 mm')
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    x = np.arange(len(names))
    for offset, partition, color in ((-.18,'validation','#237a98'),(.18,'test','#d4773c')):
        bars = axes[0].bar(x+offset,[results[partition][n]['successes'] for n in names],width=.34,label=partition.title(),color=color)
        axes[0].bar_label(bars,padding=3)
    axes[0].set(ylim=(0,105),ylabel='Collision-free XYZ success (%)',title='C3: uncertainty-aware refinement')
    axes[0].set_xticks(x,[labels.get(n,n) for n in names],rotation=20,ha='right')
    axes[0].legend()
    bins = calibration['final']['prediction_bins']
    axes[1].plot([b['mean_prediction_m']*1000 for b in bins],[b['mean_error_m']*1000 for b in bins],'o-',color='#237a98')
    axes[1].set(xlabel='Predicted error quantile (mm), five bins',ylabel='Mean actual reconstruction error (mm)',
                title=f"Geometry-error ranking; coverage {calibration['final']['empirical_coverage']:.1%}")
    for suffix in ('png','pdf'):
        fig.savefig(args.output_dir/f'c3_results.{suffix}',dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.1),constrained_layout=True)
    for row, partition in enumerate(('validation','test')):
        effect = effects[partition][args.candidate+'_minus_none']
        center = effect['success_difference_pp']; low,high = effect['paired_stratified_95_ci_pp']
        ax.errorbar(center,row,xerr=[[center-low],[high-center]],fmt='o',capsize=5,color='#237a98')
        ax.annotate(f" {center:+.0f} pp; {effect['wins']} gained, {effect['losses']} lost",(high,row),fontsize=9)
    ax.axvline(0,color='#888888',linestyle='--')
    ax.axvline(3,color='#35895b',linestyle=':',label='Requested gain')
    ax.set_yticks([0,1],['Validation','Test'])
    ax.set(xlabel='Success difference from no refinement (percentage points)',title='Paired 95% route-stratified bootstrap')
    lo,hi = ax.get_xlim(); ax.set_xlim(lo,hi+15)
    ax.legend()
    for suffix in ('png','pdf'):
        fig.savefig(args.output_dir/f'c3_paired_effects.{suffix}',dpi=200)
    plt.close(fig)
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(12, 4.6), constrained_layout=True)
    ax.set(xlim=(0, 12), ylim=(0, 4.6))
    ax.axis('off')
    boxes = [
        (0.1, 2.8, 2.35, 1.2, 'Live RGB + measured pose\nFrozen visual/point model\n20×20 metric points', '#e4f1f5'),
        (3.05, 2.8, 2.45, 1.2, 'C1: persistent anchors\nFour fresh clouds + old map\nRetain error + disagreement', '#e4f1f5'),
        (0.1, 0.7, 2.35, 1.2, 'Geometry-error head\n74 → 64 → 1; 4,865 params\nExisting teacher supervision', '#fcebdc'),
        (3.05, 0.7, 2.45, 1.2, f"C3: uncertainty padding\nLarger predicted error\n→ 0–{args.candidate.removeprefix('adaptive')} mm inflation", '#fcebdc'),
        (6.1, 1.7, 2.55, 1.3, 'C2: swept-hand queries\nTCP + two hand probes\nInflated surface proximity', '#e7f0e5'),
        (9.25, 1.7, 2.55, 1.3, 'Route-preserving choice\n14 bounded corrections\nUnchanged IK / fixed 15', '#e7f0e5'),
    ]
    for x0, y0, width, height, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x0, y0), width, height, boxstyle='round,pad=0.04',
                                   facecolor=color, edgecolor='#61727a'))
        ax.text(x0+width/2, y0+height/2, label, ha='center', va='center', fontsize=10)
    for start, end in (((2.49,3.4),(3.0,3.4)), ((1.28,2.75),(1.28,1.95)),
                       ((2.49,1.3),(3.0,1.3)), ((4.28,2.75),(4.28,1.95)),
                       ((5.54,3.4),(6.05,2.65)), ((5.54,1.3),(6.05,2.0)),
                       ((8.69,2.35),(9.2,2.35))):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,color='#52636b'))
    ax.text(6,4.35,'Geometry-grounded navigation: persistent evidence → body motion → uncertain clearance',
            ha='center',fontsize=12)
    ax.text(6,0.25,f"Depth supervises training only. Predicted error is an uncertainty signal; measured coverage is {calibration['final']['empirical_coverage']:.1%}.",
            ha='center',fontsize=10)
    for suffix in ('png','pdf'):
        fig.savefig(args.output_dir/f'c3_architecture.{suffix}',dpi=200)
    plt.close(fig)
    write_json(args.output_dir/'complete.json',dict(candidate=args.candidate,criteria=criteria,
        protocol=read_json(args.root/'protocol.json'),test_freeze=read_json(args.root/'test_freeze.json'),
        calibration_limit='Head trained for q=.9 but measured validation coverage may be lower; not a collision probability or certified bound'))
    print(criteria,flush=True)


if __name__ == '__main__':
    main()
