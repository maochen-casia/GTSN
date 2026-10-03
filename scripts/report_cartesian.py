"""Persist revision provenance and paired closed-loop comparisons in Docker."""
import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import binomtest
from tsn.common.config import read_json,write_json

ROOT=Path('/run/user/1016/experiments/gtsn_cartesian_20261003')


def figures(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'figures';out.mkdir(exist_ok=True)
    def label(name):
        mode=name.split('_e15_')[0]
        if mode=='baseline':return 'Pi3'
        if mode.startswith('pilot_'):return 'Joint '+mode.removeprefix('pilot_').replace('_correction','')
        return 'Cartesian '+mode+(' (fixed wrist)' if not name.endswith('_obaseline') else '')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,part in zip(axes,('validation','test')):
        table={k:v for k,v in results[part].items() if not k.endswith('_screen')}
        names=list(table)
        if names:
            x=np.arange(len(names))
            success=np.array([table[n]['success_rate'] for n in names])*100
            collision=np.array([table[n]['collision_rate'] for n in names])*100
            ax.bar(x,success,label='Success',color='#298c68')
            for position,rate in zip(x,success):
                ax.text(position,rate/2,f'{rate:.0f}%',ha='center',va='center',color='white',fontsize=8)
            ax.bar(x,collision,bottom=success,label='Collision',color='#bb5149')
            ax.bar(x,100-success-collision,bottom=success+collision,label='Timeout',color='#aaaeb7')
            ax.set_xticks(x,[label(n) for n in names],rotation=45,ha='right',fontsize=8)
        ax.set_ylim(0,100);ax.set_ylabel('Episodes (%)');ax.set_title(part.title()+' (shared goal servo)')
    axes[0].legend(fontsize=8);fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(out/f'closed_loop.{extension}',dpi=180)
    plt.close(fig)

    if results['test']:
        fig,ax=plt.subplots(figsize=(9,4))
        names=list(results['test']);x=np.arange(3);width=.8/len(names)
        for i,name in enumerate(names):
            table=results['test'][name]['by_route']
            ax.bar(x-.4+(i+.5)*width,[100*table[r]['success_rate'] for r in ('direct','over','side')],width,label=label(name))
        ax.set_xticks(x,['Direct (20)','Over (40)','Side (40)']);ax.set_ylim(0,105);ax.set_ylabel('Test success (%)');ax.legend(fontsize=8,ncol=2);fig.tight_layout()
        for extension in ('png','pdf'):fig.savefig(out/f'test_routes.{extension}',dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for mode in ('state','visual','uncertainty','memory'):
        path=root/'heads'/mode/'epochs.json'
        if path.exists():
            epochs=read_json(path)
            ax.plot([e['epoch'] for e in epochs],[1000*e['rmse_m'] for e in epochs],label=mode)
    ax.set_xlabel('Epoch');ax.set_ylabel('Validation waypoint RMSE (mm)');ax.legend();fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(out/f'cartesian_training.{extension}',dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--snapshot',action='store_true');p.add_argument('--tag',default='revision2b');p.add_argument('--select',action='store_true');a=p.parse_args()
    if a.snapshot:
        provenance=a.root/'provenance'/a.tag
        if provenance.exists():raise FileExistsError(provenance)
        paths=list(Path('/workspace/src/tsn').rglob('*.py'))+[Path('/workspace/scripts')/x for x in ['run_cartesian.py','cartesian_docker.py','report_cartesian.py','run_gtsn.py']]
        hashes={}
        for path in paths:
            relative=path.relative_to('/workspace');out=provenance/'source'/relative
            out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,out)
            hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
        old=read_json('/run/user/1016/experiments/gtsn_pilot_20261002/protocol.json')
        write_json(provenance/'protocol.json',dict(source_sha256=hashes,splits=old['splits'],backbone_checkpoint=old['backbone_checkpoint'],cached_features='/run/user/1016/experiments/gtsn_pilot_20261002/cache',seed=20261002,epochs=30,batch_size=256,learning_rate=.0003,weight_decay=.0001,optimizer='AdamW',scheduler='cosine',sampling=dict(expert=.65,recovery=.35,route=[.2,.4,.4],samples_per_epoch=57335),waypoint_horizons=[5,10,15,20,25,30],loss='Huber .02 metres + .001 geometry NLL where applicable',screen='first 4 direct, 8 over, 8 side validation episodes',selection='validation success, then collision rate, then waypoint RMSE',test_used_for_selection=False,renderer='unchanged OSMesa benchmark reconstruction',known_limitations=['frozen perception','current orientation preserved by tracking','near-goal servo does not certify collision-free space','test split already used in earlier pilot; confirmatory new split needed for publication']))
    results={}
    for part in ('validation','test'):
        results[part]={}
        for path in sorted((a.root/'rollouts'/part).glob('*/complete.json')):
            record=read_json(path.parent/'closed_loop.json')
            results[part][path.parent.name]={**record['overall'],'by_route':record['by_route']}
            n=record['overall']['episodes'];rate=record['overall']['success_rate'];z=1.959963984540054
            center=(rate+z*z/(2*n))/(1+z*z/n)
            radius=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
            results[part][path.parent.name]['success_wilson_95']=[float(center-radius),float(center+radius)]
    comparisons={}
    tests=list((a.root/'rollouts/test').glob('*/complete.json'))
    records={p.parent.name:read_json(p.parent/'closed_loop.json')['results'] for p in tests}
    oldroot=Path('/run/user/1016/experiments/gtsn_pilot_20261002/rollouts/test/state_correction_fixed15/closed_loop.json')
    if oldroot.exists():records['pilot_selected']=read_json(oldroot)['results']
    original=Path('/run/user/1016/experiments/pi3_small_perturbation10_20261002/test/closed_loop.json')
    if original.exists():records['pi3_original']=read_json(original)['results']
    rng=np.random.default_rng(20261003)
    for name,rec in records.items():
        for reference,control in records.items():
            if name==reference or name in ('pilot_selected','pi3_original'):continue
            c={x['episode_id']:x for x in control}
            if set(c)!={x['episode_id'] for x in rec}:continue
            x=np.array([r['success'] for r in rec],dtype=int);y=np.array([c[r['episode_id']]['success'] for r in rec],dtype=int)
            diff=x-y;boot=diff[rng.integers(0,len(diff),(20000,len(diff)))].mean(1)
            wins=int((diff==1).sum());losses=int((diff==-1).sum())
            comparisons[name+' minus '+reference]=dict(success_gain=float(diff.mean()),paired_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),wins=wins,losses=losses,mcnemar_exact_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.)
    write_json(a.root/'results.json',dict(rollouts=results,paired_test_comparisons=comparisons))
    if a.select:
        selection=a.root/'selection.json'
        if selection.exists():raise FileExistsError(selection)
        table={k:v for k,v in results['validation'].items() if not k.endswith('_screen')}
        assert len(table)==10 and all(v['episodes']==100 for v in table.values())
        # Different action representations have incomparable imitation units.
        # Prefer fewer newly trained parameters on an exact rollout tie.
        def parameters(name):
            mode=name.split('_e15_')[0]
            return 0 if mode=='baseline' else 259672 if mode.startswith('pilot_') else 691768
        selected=min(table,key=lambda n:(-table[n]['success_rate'],table[n]['collision_rate'],parameters(n),n))
        write_json(selection,dict(selected=selected,criterion='full validation success, then collision rate, then fewer newly trained parameters, then name',validation=table,test_metrics_used_for_this_selection=False))
        print('SELECTED',selected)
    figures(a.root,results)
    for part,table in results.items():
        for name,metrics in table.items():
            print(part,name,metrics['episodes'],metrics['success_rate'],metrics['collision_rate'])
    selected=read_json(a.root/'selection.json')['selected'] if (a.root/'selection.json').exists() else ''
    for name,metrics in comparisons.items():
        if selected and name.startswith(selected+' minus '):print(name,metrics)

if __name__=='__main__':main()
