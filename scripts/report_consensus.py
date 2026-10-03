"""Audit, select, and report the consensus study without test-driven tuning."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from tsn.common.config import read_json, write_json

ROOT = Path('/run/user/1016/experiments/gtsn_consensus_20261003')
BASE_ROOT = Path('/run/user/1016/experiments/gtsn_cartesian_20261003')


def paired(a, b):
    reference = {r['episode_id']: r for r in b}
    assert set(reference) == {r['episode_id'] for r in a}
    difference = np.array([int(r['success'])-int(reference[r['episode_id']]['success']) for r in a])
    rng = np.random.default_rng(20261003)
    totals = np.zeros(20000)
    for route in ('direct', 'over', 'side'):
        values = difference[[r['route'] == route for r in a]]
        totals += values[rng.integers(len(values), size=(20000,len(values)))].sum(1)
    wins, losses = int((difference==1).sum()), int((difference==-1).sum())
    return dict(gain=float(difference.mean()), bootstrap_95=np.quantile(totals/len(a),[.025,.975]).tolist(),
                wins=wins, losses=losses, mcnemar_exact_p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--select', nargs='+', help='Completed full-validation candidates, fixed before test')
    p.add_argument('--test-ablations', nargs='*', default=[])
    a = p.parse_args()
    records, summaries, audit = {}, {}, {}
    for part in ('validation','test'):
        records[part], summaries[part] = {}, {}
        for path in sorted((a.root/'rollouts'/part).glob('*/complete.json')):
            name = path.parent.name
            record = read_json(path.parent/'closed_loop.json')
            protocol = read_json(path.parent/'protocol.json')
            complete = read_json(path)
            assert complete['sources_unchanged'], path
            for filename, digest in protocol['source_sha256_at_start'].items():
                import hashlib
                assert hashlib.sha256((path.parent/'source'/filename).read_bytes()).hexdigest() == digest
            if part == 'test':
                selection = read_json(a.root/'selection.json')
                assert protocol['specification'] == selection['test_conditions'][name]
                assert protocol['source_sha256_at_start'] == selection['condition_sources'][name]
            assert [r['episode_id'] for r in record['results']] == protocol['ids']
            expected = dict(direct=4, over=8, side=8) if name.endswith('_screen') else dict(direct=20,over=40,side=40)
            assert Counter(r['route'] for r in record['results']) == expected
            observations, short = [], []
            for result in record['results']:
                assert not result['privileged_action_map'] and result['expert_progress_method'] is None
                assert result['goal_success_criterion'] == 'xyz'
                trajectory = np.load(path.parent/'episodes'/result['episode_id']/'trajectory.npz')
                assert trajectory['reference_indices'].size == 0
                for key in ('qpos','T_B_E','T_B_C','predicted_joint_targets','geometry_risk'):
                    assert np.isfinite(trajectory[key]).all()
                assert len(trajectory['qpos']) == result['control_steps']+1
                assert len(trajectory['predicted_joint_targets']) == result['control_steps']
                observations.append(result['replans'])
                short.extend(trajectory['execution_horizons'] == 5)
            records[part][name] = record['results']
            summaries[part][name] = {**record['overall'], 'by_route':record['by_route'],
                'mean_observations':float(np.mean(observations)), 'fraction_short_chunks':float(np.mean(short))}
            audit[f'{part}/{name}'] = dict(episodes=len(record['results']), passed=True)
    if a.select:
        if (a.root/'selection.json').exists():
            raise FileExistsError('Selection is immutable')
        assert not records['test'], 'Cannot select after test'
        # Selection candidates require full validation. Diagnostic interventions
        # may be predeclared while their validation finishes, since they cannot
        # change the frozen winner and are not used for selecting it.
        for name in a.select:
            assert summaries['validation'][name]['episodes'] == 100
        def rank(name):
            d = summaries['validation'][name]
            return (-d['success_rate'], d['collision_rate'], d['mean_observations'], name)
        winner = min(a.select, key=rank)
        conditions = {name:read_json(a.root/'rollouts/validation'/name/'protocol.json')['specification']
                      for name in dict.fromkeys([winner]+a.test_ablations)}
        condition_sources = {name:read_json(a.root/'rollouts/validation'/name/'protocol.json')['source_sha256_at_start']
                             for name in conditions}
        write_json(a.root/'selection.json',dict(selected=winner, candidates=a.select,
            rule='full validation success, collision rate, mean observations, name',
            validation=summaries['validation'], test_conditions=conditions,
            condition_sources=condition_sources, test_used_for_selection=False))
        print('FROZEN SELECTION',winner,conditions,flush=True)
    comparisons = {}
    for part in ('validation','test'):
        baseline = read_json(BASE_ROOT/'rollouts'/part/'state_e15_s0.08_obaseline'/'closed_loop.json')['results']
        for name, values in records[part].items():
            if len(values) == 100:
                comparisons[f'{part}/{name} minus reference'] = paired(values, baseline)
        for name, values in records[part].items():
            if len(values) != 100: continue
            for other, control in records[part].items():
                if name > other and len(control)==100:
                    comparisons[f'{part}/{name} minus {other}'] = paired(values,control)
    write_json(a.root/'results.json',dict(rollouts=summaries, comparisons=comparisons))
    write_json(a.root/'audit.json',dict(passed=True, conditions=audit, episodes=sum(x['episodes'] for x in audit.values())))
    print(summaries, comparisons, flush=True)
    if records['test']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        display = {'reference':'Reference', 'state_memory':'Consensus (selected)',
                   'state_visual':'Deterministic visual', 'state_uncertainty':'Current uncertainty',
                   'memory_current_only':'No history', 'memory_no_geometry':'No explicit maps'}
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        for ax, part in zip(axes,('validation','test')):
            base = read_json(BASE_ROOT/'rollouts'/part/'state_e15_s0.08_obaseline'/'closed_loop.json')['overall']
            table = {'reference':base, **{k:v for k,v in summaries[part].items() if v['episodes']==100}}
            labels = list(table)
            s=np.array([table[k]['success_rate'] for k in labels])*100
            c=np.array([table[k]['collision_rate'] for k in labels])*100
            shown = [display.get(k,k) for k in labels]
            ax.bar(shown,s,label='success',color='#298c68')
            ax.bar(shown,c,bottom=s,label='collision',color='#bb5149')
            ax.bar(shown,100-s-c,bottom=s+c,label='timeout',color='#aaaeb7')
            for i,x in enumerate(s): ax.text(i,x/2,f'{x:.0f}%',ha='center',color='white')
            ax.set_ylim(0,100); ax.set_title(part); ax.tick_params(axis='x',labelrotation=35)
        axes[0].set_ylabel('Episodes (%)'); axes[1].legend(fontsize=8)
        fig.tight_layout(); (a.root/'figures').mkdir(exist_ok=True)
        for ext in ('png','pdf'): fig.savefig(a.root/'figures'/f'closed_loop.{ext}',dpi=180)
        plt.close(fig)
        base = read_json(BASE_ROOT/'rollouts/test/state_e15_s0.08_obaseline/closed_loop.json')
        table = {'reference':base, **summaries['test']}
        fig,ax=plt.subplots(figsize=(10,4))
        width=.8/len(table)
        for i,(name,record) in enumerate(table.items()):
            ax.bar(np.arange(3)-.4+(i+.5)*width,
                   [100*record['by_route'][route]['success_rate'] for route in ('direct','over','side')],
                   width,label=display.get(name,name))
        ax.set_xticks(range(3),['Direct (20)','Over (40)','Side (40)'])
        ax.set_ylim(0,105);ax.set_ylabel('Test success (%)');ax.legend(fontsize=8,ncol=2)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(a.root/'figures'/f'test_routes.{ext}',dpi=180)


if __name__ == '__main__': main()
