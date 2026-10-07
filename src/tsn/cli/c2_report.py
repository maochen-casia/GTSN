"""Audit C2 body geometry, paired effects, primary/secondary targets and figures."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from tsn.cli.report import paired
from tsn.common.config import create_output,read_json,write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--secondary')
    parser.add_argument('--related-root',type=Path,action='append',default=[],
                        help='Also audit completed selection panels, deduplicating reused rollout paths')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu',),default='cpu')
    args = parser.parse_args()
    freeze = read_json(args.root/'test_freeze.json')
    if args.candidate!=freeze['candidate']:
        raise ValueError('Primary candidate must match the pre-test freeze')
    if args.secondary and args.secondary not in freeze['conditions']:
        raise ValueError('Secondary comparison was not prespecified')
    create_output(args.output_dir)
    protocol = read_json(args.root/'protocol.json')
    reference = protocol.get('reference','tcp')
    results,effects,audits,raw = {},{},{},{}
    for split in ('validation','test'):
        results[split],effects[split],audits[split],raw[split] = {},{},{},{}
        for path in sorted((args.root/split).glob('*/closed_loop.json')):
            directory = path.parent;name = directory.name
            if not (directory/'complete.json').exists():raise ValueError('Incomplete evaluation')
            data = read_json(path)
            if len(data['results'])!=100 or {k:v['episodes'] for k,v in data['by_route'].items()}!=dict(direct=20,over=40,side=40):
                raise ValueError('Require full fixed stratified split')
            rows = []
            for episode in data['results']:
                trace = np.load(directory/'episodes'/episode['episode_id']/'trajectory.npz')
                if len(trace['reference_indices']) or not all(np.isfinite(trace[k]).all() for k in trace.files):
                    raise ValueError('Nonfinite trajectory or privileged progress')
                if not np.all(trace['execution_horizons']==15):raise ValueError('Execution changed')
                diagnostics = read_json(directory/'episodes'/episode['episode_id']/'policy_diagnostics.json')
                if any(r.get('route_feature_frames',4)>4 for r in diagnostics):raise ValueError('Proposal history changed')
                if name==reference and any(r.get('body_mode')!='tcp' or r.get('body_weight')!=0 or
                                          r.get('self_masked_points',0)!=0 or r.get('roll_angle_deg',0)!=0 for r in diagnostics):
                    raise ValueError('TCP control applied embodiment information')
                rows.extend(diagnostics)
            raw[split][name] = data
            results[split][name] = dict(overall=data['overall'],by_route=data['by_route'],
                mean_correction_m=float(np.mean([r['correction_m'] for r in rows])),
                mean_tcp_cost=float(np.mean([r['tcp_risk'] for r in rows])) if all('tcp_risk' in r for r in rows) else None,
                mean_body_cost=float(np.mean([r['body_risk'] for r in rows])) if all('body_risk' in r for r in rows) else None,
                mean_geometry_classified_self_points=float(np.mean([r.get('self_masked_points',0) for r in rows])),
                self_count_scope='Raw geometry classification; overlaps prior workspace/TCP masks',
                timeouts=sum(r['termination']=='timeout' for r in data['results']))
            audits[split][name] = dict(episodes=100,finite=True,no_expert_progress=True,fixed15=True,
                tcp_only_verified=name==reference,summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for first in raw[split]:
            for second in raw[split]:
                if first!=second:effects[split][first+'_minus_'+second] = paired(raw[split][second],raw[split][first])
    criteria = {}
    for name in [args.candidate]+([args.secondary] if args.secondary else []):
        criteria[name] = {}
        for split in ('validation','test'):
            effect = effects[split][name+'_minus_'+reference]
            score = results[split][name]['overall']['success_rate']
            criteria[name][split] = dict(success=score,reference_success=results[split][reference]['overall']['success_rate'],
                gain_pp=effect['success_difference_pp'],at_least_3_pp=effect['success_difference_pp']>=3-1e-8,
                at_least_70_percent=score>=.70,paired_95_ci_pp=effect['paired_stratified_95_ci_pp'])
        criteria[name]['target_met'] = all(criteria[name][s]['at_least_3_pp'] and criteria[name][s]['at_least_70_percent'] for s in ('validation','test'))
    criteria['primary_target_met'] = criteria[args.candidate]['target_met']
    if args.secondary:criteria['secondary_target_met'] = criteria[args.secondary]['target_met']
    for filename,value in (('summary.json',results),('paired_effects.json',effects),('audit.json',audits),('criteria.json',criteria)):
        write_json(args.output_dir/filename,value)
    roots = list(dict.fromkeys([args.root.resolve()]+[p.resolve() for p in args.related_root]))
    verified = {};instances = 0
    for root in roots:
        saved = read_json(root/'protocol.json')
        if saved['checkpoint_sha256']!=protocol['checkpoint_sha256']:
            raise ValueError('Related panel changed the checkpoint')
        for relative,digest in read_json(root/'source_sha256.json').items():
            if hashlib.sha256((root/'source'/relative).read_bytes()).hexdigest()!=digest:
                raise ValueError('Frozen panel source changed')
        reference_name = saved.get('reference','tcp')
        for split in ('validation','test'):
            for path in sorted((root/split).glob('*/closed_loop.json')):
                instances += 100;directory = path.parent;canonical = directory.resolve()
                if canonical in verified:continue
                if not (directory/'complete.json').exists():raise ValueError('Incomplete related panel')
                data = read_json(path)
                if len(data['results'])!=100 or {k:v['episodes'] for k,v in data['by_route'].items()}!=dict(direct=20,over=40,side=40):
                    raise ValueError('Related panel split differs')
                for episode in data['results']:
                    trace = np.load(directory/'episodes'/episode['episode_id']/'trajectory.npz')
                    if len(trace['reference_indices']) or not all(np.isfinite(trace[k]).all() for k in trace.files) or not np.all(trace['execution_horizons']==15):
                        raise ValueError('Related panel trajectory audit failed')
                    rows = read_json(directory/'episodes'/episode['episode_id']/'policy_diagnostics.json')
                    if any(r.get('route_feature_frames',4)>4 for r in rows):raise ValueError('Related panel history changed')
                    if directory.name==reference_name and any(r.get('body_mode')!='tcp' or r.get('body_weight')!=0 or
                            r.get('self_masked_points',0)!=0 or r.get('roll_angle_deg',0)!=0 for r in rows):
                        raise ValueError('Related TCP control used body information')
                verified[canonical] = dict(episodes=100,partition=split,
                    new_c2_rollouts=any(canonical.is_relative_to(p) for p in roots),
                    summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(args.output_dir/'study_audit.json',dict(roots=[str(p) for p in roots],
        condition_episode_instances=instances,unique_rollouts=sum(v['episodes'] for v in verified.values()),
        new_c2_rollouts=sum(v['episodes'] for v in verified.values() if v['new_c2_rollouts']),
        all_finite=True,no_expert_progress=True,fixed15=True,full_stratified_splits=True,
        frozen_source_hashes_verified=True,rollouts={str(p):v for p,v in verified.items()}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,FancyBboxPatch
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull,HalfspaceIntersection
    from tsn.models.embodied_clearance import PandaToolGeometry
    names = list(dict.fromkeys([reference,'axial',args.candidate]+([args.secondary] if args.secondary else [])))
    names = [n for n in names if n in results['test']]
    x = np.arange(len(names))
    fig,ax = plt.subplots(figsize=(9,4.9),constrained_layout=True)
    for offset,split,color in ((-.18,'validation','#237a98'),(.18,'test','#d4773c')):
        bars = ax.bar(x+offset,[results[split][n]['overall']['success_rate']*100 for n in names],width=.34,color=color,label=split.title())
        ax.bar_label(bars,padding=3)
    ax.set(ylim=(0,105),ylabel='Collision-free XYZ success (%)',title='C2: explicit hand/tool geometry')
    labels = dict(tcp='TCP only',tcp_signed='TCP only\n(signed)',axial='Three axial\nprobes',
                  tool_parts100='Selected regional\nhand + rigid tool',
                  hand_signed35='Primary signed\nhand',tool_signed35='Secondary signed\nhand + rigid tool')
    ax.set_xticks(x,[labels.get(n,n.replace('_','\n')) for n in names]);ax.legend()
    for suffix in ('png','pdf'):fig.savefig(args.output_dir/f'c2_results.{suffix}',dpi=200)
    plt.close(fig)
    geometry = PandaToolGeometry()
    planes = geometry.palm_planes.numpy()
    center = linprog([0,0,0,-1],A_ub=np.c_[planes[:,:3],np.ones(len(planes))],
        b_ub=-planes[:,3],bounds=[(None,None)]*3+[(0,None)],method='highs')
    if not center.success:raise ValueError('Cannot derive palm hull for figure')
    palm = HalfspaceIntersection(planes,center.x[:3]).intersections
    centers = geometry.box_centers.numpy()+np.einsum('f,kfi->ki',np.array([.02,.02]),geometry.box_motion.numpy())
    signs = np.array([[i,j,k] for i in (-1,1) for j in (-1,1) for k in (-1,1)])
    fig,axes = plt.subplots(1,2,figsize=(10,4.8),constrained_layout=True)
    for ax,indices,title in zip(axes,((0,1),(1,2)),('Palm width and finger opening','Extent behind the TCP')):
        projected = palm[:,indices]*1000
        ax.add_patch(Polygon(projected[ConvexHull(projected).vertices],facecolor='#f0c295',edgecolor='#985a24',alpha=.7,label='Palm hull'))
        for i in range(8):
            vertices = (signs*geometry.box_half[i].numpy())@geometry.box_rotations[i].numpy().T+centers[i]
            projected = vertices[:,indices]*1000
            ax.add_patch(Polygon(projected[ConvexHull(projected).vertices],facecolor='#92c0cd',edgecolor='#237a98',alpha=.45,label='Finger boxes' if i==0 else None))
        axial = np.array([[0,0,0],[0,0,-50],[0,0,-100]])
        ax.plot(axial[:,indices[0]],axial[:,indices[1]],'o:',color='#555555',label='Old axial probes')
        ax.scatter([0],[0],marker='+',s=100,color='#a42d34',label='TCP only')
        ax.set_aspect('equal');ax.autoscale_view();ax.set(title=title,
            xlabel=f"TCP-frame {'xyz'[indices[0]]} (mm)",ylabel=f"TCP-frame {'xyz'[indices[1]]} (mm)")
    axes[0].legend(fontsize=8)
    fig.suptitle('Known collision geometry; example measured finger positions 20 mm each')
    for suffix in ('png','pdf'):fig.savefig(args.output_dir/f'c2_geometry.{suffix}',dpi=200)
    plt.close(fig)
    planes = geometry.wrist_planes.numpy()
    center = linprog([0,0,0,-1],A_ub=np.c_[planes[:,:3],np.ones(len(planes))],
        b_ub=-planes[:,3],bounds=[(None,None)]*3+[(0,None)],method='highs')
    if not center.success:raise ValueError('Cannot derive wrist hull for figure')
    wrist = HalfspaceIntersection(planes,center.x[:3]).intersections
    fig,axes = plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    for ax,indices in zip(axes,((0,1),(1,2))):
        for vertices,color,edge,label in ((palm,'#f0c295','#985a24','Palm'),(wrist,'#c6b0d4','#72528b','Rigid wrist')):
            projected = vertices[:,indices]*1000
            ax.add_patch(Polygon(projected[ConvexHull(projected).vertices],facecolor=color,edgecolor=edge,alpha=.65,label=label))
        for i in range(9):
            vertices = (signs*geometry.box_half[i].numpy())@geometry.box_rotations[i].numpy().T+centers[i]
            projected = vertices[:,indices]*1000
            ax.add_patch(Polygon(projected[ConvexHull(projected).vertices],facecolor='#92c0cd' if i<8 else '#a2bf89',
                edgecolor='#237a98' if i<8 else '#4f7033',alpha=.5,label='Fingers' if i==0 else 'Camera housing' if i==8 else None))
        axial = np.array([[0,0,0],[0,0,-50],[0,0,-100]])
        ax.plot(axial[:,indices[0]],axial[:,indices[1]],'o:',color='#555555',label='Old axial probes')
        ax.scatter([0],[0],marker='+',s=100,color='#a42d34',label='TCP')
        ax.set_aspect('equal');ax.autoscale_view();ax.set(
            xlabel=f"TCP-frame {'xyz'[indices[0]]} (mm)",ylabel=f"TCP-frame {'xyz'[indices[1]]} (mm)")
    axes[0].legend(fontsize=8)
    fig.suptitle('Six clearance regions: TCP, palm, left/right fingers, rigid wrist, camera')
    for suffix in ('png','pdf'):fig.savefig(args.output_dir/f'c2_tool_geometry.{suffix}',dpi=200)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(11,5),constrained_layout=True)
    ax.set(xlim=(0,11),ylim=(0,5));ax.axis('off')
    blocks = [(0.1,3.2,'RGB geometry + C1 map\nC3 point-specific padding'),
              (3.9,3.2,'Known collision geometry\nMeasured TCP + finger opening'),
              (7.7,3.2,'Frozen route proposal\n14 positional candidates'),
              (3.9,1.1,'Future body poses, 5 times\nOne contact feature per region'),
              (7.7,1.1,'Geometry cost + route trust\nSelect, IK, execute 15 steps')]
    for left,bottom,label in blocks:
        ax.add_patch(FancyBboxPatch((left,bottom),3.15,1.1,boxstyle='round,pad=0.08',
            facecolor='#e1eff2' if bottom>3 else '#f4e5d8',edgecolor='#53727b'))
        ax.text(left+1.575,bottom+.55,label,ha='center',va='center',fontsize=10)
    for start,end in (((1.7,3.1),(4.4,2.3)),((5.5,3.1),(5.5,2.3)),((9.3,3.1),(6.6,2.3)),((7.15,1.65),(7.55,1.65))):
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#53727b',lw=1.8))
    ax.text(.1,.2,'Deployment: RGB and measured robot/camera/goal state; no scene depth, contacts or expert progress.',fontsize=10)
    fig.suptitle('C2: candidate-conditioned regional embodiment geometry')
    for suffix in ('png','pdf'):fig.savefig(args.output_dir/f'c2_architecture.{suffix}',dpi=200)
    plt.close(fig)
    write_json(args.output_dir/'complete.json',dict(primary=args.candidate,secondary=args.secondary,reference=reference,
        criteria=criteria,protocol=protocol,test_freeze=freeze,
        interpretation='Prespecified secondary comparisons retain that label; the primary nominee is not changed after test'))
    print(criteria,flush=True)


if __name__=='__main__':main()
