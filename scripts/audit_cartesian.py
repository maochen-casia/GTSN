"""Audit splits, matched training, causal inputs, and completed rollout artifacts."""
import hashlib
from pathlib import Path
import shutil
import numpy as np
from tsn.common.config import read_json, write_json

root=Path('/run/user/1016/experiments/gtsn_cartesian_20261003')
protocol=read_json(root/'protocol.json')
splits=protocol['splits']
assert len(set(splits['train']) & set(splits['validation']))==0
assert len(set(splits['train']) & set(splits['test']))==0
assert len(set(splits['validation']) & set(splits['test']))==0
for part,counts in [('train',[160,320,320]),('validation',[20,40,40]),('test',[20,40,40])]:
    actual=[0,0,0]
    for ep in splits[part]:
        number=int(ep.split('_')[-1]);actual[0 if number<200 else 1 if number<600 else 2]+=1
    assert actual==counts
histories=[read_json(root/'heads'/mode/'epochs.json') for mode in ('state','visual','uncertainty','memory')]
assert all(len(h)==30 for h in histories)
assert all([x['sampling_sha256'] for x in h]==[x['sampling_sha256'] for x in histories[0]] for h in histories)
completed=[]
repeated_screens=[]
for path in (root/'rollouts').glob('*/*/complete.json'):
    part=path.parent.parent.name
    result=read_json(path.parent/'closed_loop.json')
    records=result['results'];ids=[r['episode_id'] for r in records]
    completion=read_json(path)
    assert len(ids)==len(set(ids))==completion['episodes']
    if part=='test':
        assert completion['sources_unchanged']
        assert completion['source_sha256_at_start']==completion['source_sha256_at_end']
    assert set(ids)<=set(splits[part])
    if not path.parent.name.endswith('_screen'):assert set(ids)==set(splits[part])
    for record in records:
        assert record['privileged_action_map'] is False
        assert record['expert_progress_method'] is None
        assert record['control_steps']<=400
        assert record['goal_success_criterion']=='xyz'
        if record['success']:
            assert record['final_position_error_m']<=.01 and record['collision'] is None
        data=np.load(path.parent/'episodes'/record['episode_id']/'trajectory.npz')
        assert data['reference_indices'].size==0
        for key in ('qpos','T_B_E','T_B_C','predicted_joint_targets'):assert np.isfinite(data[key]).all()
        assert len(data['qpos'])==record['control_steps']+1
    completed.append(dict(partition=part,condition=path.parent.name,episodes=len(ids)))
    if path.parent.name.endswith('_screen'):
        full=path.parent.with_name(path.parent.name.removesuffix('_screen'))
        if (full/'complete.json').exists():
            comparison={r['episode_id']:r for r in read_json(full/'closed_loop.json')['results']}
            for record in records:
                other=comparison[record['episode_id']]
                assert record['success']==other['success'] and record['termination']==other['termination']
                assert record['control_steps']==other['control_steps']
                original=np.load(path.parent/'episodes'/record['episode_id']/'trajectory.npz')['qpos']
                repeated=np.load(full/'episodes'/record['episode_id']/'trajectory.npz')['qpos']
                np.testing.assert_allclose(original,repeated,atol=1e-5,rtol=1e-5)
            repeated_screens.append(dict(condition=full.name,episodes=len(ids)))
test_conditions=[x for x in completed if x['partition']=='test']
study_complete=len(completed)==23 and len(test_conditions)==6
selection=read_json(root/'selection.json') if (root/'selection.json').exists() else None
if study_complete:
    assert selection and selection['test_metrics_used_for_this_selection'] is False
    assert selection['selected'] in {x['condition'] for x in test_conditions}
write_json(root/'audit.json',dict(passed=True,study_complete=study_complete,split_counts_correct=True,disjoint_splits=True,matched_training_order=True,training_epochs_each=30,completed=completed,total_rollout_episodes=sum(x['episodes'] for x in completed),repeated_screens_match=repeated_screens,finite_trajectories=True,no_expert_progress_inputs=True,test_sources_unchanged=True,original_goal_and_collision_criterion=True,kinematics=read_json(root/'kinematics_audit.json')))
print('AUDIT PASSED',len(completed),'conditions',sum(x['episodes'] for x in completed),'episodes')
if study_complete:
    results=read_json(root/'results.json')
    selected=selection['selected']
    gain=results['paired_test_comparisons'][selected+' minus baseline_e15_s0.08']
    assert gain['paired_bootstrap_95'][0]>0
    write_json(root/'status.json',dict(state='complete',selected=selected,training_runs=4,epochs_each=30,validation_rollouts=1140,test_rollouts=600,test=results['rollouts']['test'][selected],gain_over_controller_matched_pi3=gain,audit_passed=True,test_selection_changed=False))
    archive=root/'provenance/final'
    if not archive.exists():
        paths=list(Path('/workspace/src/tsn').rglob('*.py'))
        paths+=list(Path('/workspace/scripts').glob('*cartesian*.py'))
        paths+=[Path('/workspace/scripts/run_gtsn.py'),Path('/workspace/tests/test_cartesian.py'),Path('/workspace/tests/test_gtsn.py'),Path('/workspace/documents/gtsn_methodology.md'),Path('/workspace/documents/gtsn_experiments.md'),Path('/workspace/configs/eval/pi3_small.json')]
        hashes={}
        for path in paths:
            relative=path.relative_to('/workspace');destination=archive/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination)
            hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(archive/'sha256.json',hashes)
