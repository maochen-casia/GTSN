"""Export the validation-selected composition without copying frozen weights."""
import hashlib
from pathlib import Path
from tsn.common.config import read_json,write_json
from run_cartesian import ROOT,PILOT
from run_gtsn import CHECKPOINT

selection=read_json(ROOT/'selection.json')
condition=selection['selected']
mode=condition.split('_e15_')[0]
if mode.startswith('pilot_'):
    head=PILOT/'heads'/mode.removeprefix('pilot_')/'best.pt'
    family='joint_residual'
elif mode=='baseline':
    head=None;family='pi3'
else:
    head=ROOT/'heads'/mode/'best.pt';family='cartesian'
weights=[CHECKPOINT]+([head] if head else [])
write_json(ROOT/'selected_policy.json',dict(condition=condition,family=family,mode=mode.removeprefix('pilot_'),backbone_checkpoint=str(CHECKPOINT),head_checkpoint=str(head) if head else None,weight_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in weights},servo_radius_m=.08,chunk_size=30,execute_horizon=15,orientation='baseline' if condition.endswith('_obaseline') else 'current',input_contract=['wrist RGB uint8','measured qpos','goal XYZ+wxyz','camera intrinsics','measured camera-to-base transform'],forbidden_inputs=['measured depth','scene objects','route label','expert trajectory or progress'],selection=selection['criterion'],docker_image_id='sha256:5716cc81a619a9d00e6d08edecedff42a5b2b57fe10c8ff22835920cb480e009'))
print(condition)
