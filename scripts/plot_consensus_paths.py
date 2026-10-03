"""Transparent validation examples: first recovery and first regression by ID."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tsn.common.config import read_json, write_json
from run_consensus import ROOT, BASE_ROOT


def main():
    selection = read_json(ROOT/'selection.json')
    name = selection['selected']
    base = BASE_ROOT/'rollouts/validation/state_e15_s0.08_obaseline'
    candidate = ROOT/'rollouts/validation'/name
    old = {r['episode_id']:r for r in read_json(base/'closed_loop.json')['results']}
    new = {r['episode_id']:r for r in read_json(candidate/'closed_loop.json')['results']}
    chosen=[]
    for route in ('over','side'):
        for outcome, fn in [('recovery',lambda x,y: x and not y), ('regression',lambda x,y: y and not x)]:
            ids = sorted(ep for ep in old if new[ep]['route']==route and fn(new[ep]['success'],old[ep]['success']))
            if ids: chosen.append((route,outcome,ids[0]))
    if not chosen: return
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,len(chosen),figsize=(4*len(chosen),7),squeeze=False)
    details=[]
    for column,(route,outcome,ep) in enumerate(chosen):
        for path,label,color in [(base,'reference','#bb5149'),(candidate,'selected','#298c68')]:
            xyz=np.load(path/'episodes'/ep/'trajectory.npz')['T_B_E'][:,:3,3]
            axes[0,column].plot(xyz[:,0],xyz[:,1],label=label,color=color)
            axes[0,column].scatter(*xyz[-1,:2],color=color,marker='x')
            axes[1,column].plot(np.arange(len(xyz))/20,xyz[:,2],label=label,color=color)
        axes[0,column].set_title(f'{ep}: {route} {outcome}')
        axes[0,column].set_xlabel('Base-frame x (m)');axes[0,column].set_ylabel('Base-frame y (m)')
        axes[0,column].axis('equal');axes[0,column].legend()
        axes[1,column].set_xlabel('Control time (s)');axes[1,column].set_ylabel('TCP height (m)')
        details.append(dict(episode=ep,route=route,comparison=outcome,
                            reference=old[ep]['termination'], selected=new[ep]['termination']))
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'validation_paths.{ext}',dpi=180)
    write_json(out/'validation_paths.json',dict(rule='First episode by ID for each route/outcome; validation only',examples=details))


if __name__ == '__main__': main()
