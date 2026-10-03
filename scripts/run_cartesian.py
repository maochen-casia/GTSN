"""Revision 2 numerical experiments; run only inside project Docker."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from run_gtsn import backbone, load_data, inputs, SEED, source_hashes as backbone_source_hashes
from tsn.common.config import read_json, write_json
from tsn.common.seed import seed_everything
from tsn.models.cartesian_policy import CartesianHead, CartesianPolicy, PandaKinematics, GoalServoPolicy
from tsn.models.gtsn_policy import geometry_nll, GTSNHead, GTSNPolicy
from tsn.models.factory import make_maps
from tsn.data.splits import episode_catalog
from tsn.evaluation.closed_loop import evaluate_rollouts

PILOT = Path('/run/user/1016/experiments/gtsn_pilot_20261002')
ROOT = Path('/run/user/1016/experiments/gtsn_cartesian_20261003')


def source_hashes():
    return {**backbone_source_hashes(), 'scripts/run_cartesian.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def data(partition, kin):
    d = load_data(PILOT/'cache'/partition)
    with torch.no_grad():
        d['tcp'] = kin(d['state'][:, :7]*np.pi)
        targets = []
        for idx in torch.arange(len(d['state']), device='cuda').split(512):
            q = d['state'][idx, None, :7]*np.pi + d['target'][idx][:, 4::5]
            targets.append(kin(q)[..., :3, 3]-d['tcp'][idx, None, :3, 3])
        d['waypoint'] = torch.cat(targets)
    return d


def forward(head, d, idx):
    args = inputs(d, idx)
    return head(*args[:6], d['tcp'][idx])


@torch.inference_mode()
def evaluate(head, d):
    head.eval()
    errors=[]
    for idx in torch.arange(len(d['state']), device='cuda').split(256):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, aux = forward(head,d,idx)
        errors.append((pred-d['waypoint'][idx]).square().mean((1,2)))
    return float(torch.cat(errors).mean().sqrt())


def train(args):
    kin = PandaKinematics().cuda()
    tr, val = data('train',kin), data('validation',kin)
    head = CartesianHead(args.mode).cuda()
    optimizer = torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    routes,sources = tr['route'].cpu().numpy(),tr['source'].cpu().numpy()
    weights = np.zeros(len(routes))
    for s,fraction in enumerate((.65,.35)):
        for r,p in enumerate((.2,.4,.4)):
            mask=(sources==s)&(routes==r)
            weights[mask]=fraction*p/mask.sum()
    out=args.root/'heads'/args.mode
    out.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator().manual_seed(SEED)
    best=float('inf');records=[]
    for epoch in range(1,args.epochs+1):
        start=time.monotonic();head.train()
        order=torch.multinomial(torch.from_numpy(weights),57335,replacement=True,generator=generator)
        losses=[]
        for idx in order.cuda().split(256):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred,aux=forward(head,tr,idx)
                loss=F.huber_loss(pred,tr['waypoint'][idx],delta=.02)
                if args.mode in ('uncertainty','memory'):
                    loss=loss+.001*geometry_nll(aux,tr['teacher'][idx],tr['geometry_valid'][idx])
            if not torch.isfinite(loss):raise FloatingPointError('loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        rmse=evaluate(head,val)
        if rmse<best:
            best=rmse;torch.save(head.state_dict(),out/'best.pt')
            selected=epoch
        scheduler.step()
        records.append(dict(epoch=epoch,rmse_m=rmse,loss=float(np.mean(losses)),seconds=time.monotonic()-start,sampling_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest()))
        write_json(out/'epochs.json',records)
        print(f'{args.mode} epoch {epoch} RMSE={rmse:.6f} best={best:.6f} seconds={records[-1]["seconds"]:.1f}',flush=True)
    write_json(out/'results.json',dict(selected_epoch=selected,rmse_m=best,epochs=args.epochs,parameters=sum(p.numel() for p in head.parameters())))


def rollout(args):
    hashes=source_hashes()
    if args.partition=='test' and not (args.root/'selection.json').exists():
        raise RuntimeError('Freeze validation selection before running test comparisons')
    model,cfg,splits=backbone()
    kin=PandaKinematics().cuda()
    if args.mode.startswith('pilot_'):
        mode=args.mode.removeprefix('pilot_')
        head=GTSNHead(mode).cuda()
        head.load_state_dict(torch.load(PILOT/'heads'/mode/'best.pt',weights_only=True))
        policy=GoalServoPolicy(GTSNPolicy(model,head),kin,args.servo).cuda().eval()
    else:
        head=CartesianHead(args.mode if args.mode!='baseline' else 'state').cuda()
        if args.mode!='baseline':head.load_state_dict(torch.load(args.root/'heads'/args.mode/'best.pt',weights_only=True))
        policy=CartesianPolicy(model,head,kin,args.servo,args.execute,args.mode=='baseline',args.orientation).cuda().eval()
    catalog=episode_catalog(Path(cfg['benchmark']['root']))
    ids=splits[args.partition]
    if args.screen:
        ids=[ep for route,n in [('direct',4),('over',8),('side',8)] for ep in [x for x in ids if catalog[x]==route][:n]]
    name=f'{args.mode}_e{args.execute}_s{args.servo:g}'+('_obaseline' if args.orientation=='baseline' else '')+('_screen' if args.screen else '')
    out=(args.output_root or args.root)/'rollouts'/args.partition/name
    if (out/'complete.json').exists():return
    options=read_json(Path('/workspace/configs/eval/pi3_small.json'))
    options['execute_horizon']=args.execute
    result=evaluate_rollouts(ids,catalog,Path(cfg['benchmark']['root']),out,policy,make_maps(cfg['model']).cuda(),torch.device('cuda'),options)
    assert not result['privileged_action_map']
    end_hashes=source_hashes()
    write_json(out/'complete.json',dict(episodes=len(ids),mode=args.mode,servo_radius=args.servo,execute=args.execute,orientation=args.orientation,source_sha256_at_start=hashes,source_sha256_at_end=end_hashes,sources_unchanged=hashes==end_hashes))


def audit(args):
    kin=PandaKinematics().cuda()
    protocol=read_json(PILOT/'protocol.json')
    errors=[]
    for ep in protocol['splits']['validation'][::10]:
        with h5py.File(Path('/run/user/1016/tsn-1k')/ep/'episode.h5') as f:
            q=torch.tensor(f['qpos'][:,:7],device='cuda')
            xyz=kin(q)[:,:3,3].cpu().numpy()
            errors.append(float(np.max(np.linalg.norm(xyz-f['ee_pose'][:,:3],axis=1))))
    q=torch.tensor([[0,.4,0,-1.96,0,2.35,.78]],device='cuda')
    pose,jac=kin(q,True)
    numerical=[]
    for j in range(7):
        dq=torch.zeros_like(q);dq[:,j]=.001
        numerical.append((kin(q+dq)[:,:3,3]-kin(q-dq)[:,:3,3])/.002)
    jac_error=float((torch.stack(numerical,-1)-jac[:,:3]).abs().max())
    target=pose[:,:3,3]+torch.tensor([[.02,-.03,.015]],device='cuda')
    ik=kin.inverse(q,target,pose[:,:3,:3])
    ikerror=float((kin(ik)[:,:3,3]-target).norm())
    assert max(errors)<1e-5 and jac_error<.001 and ikerror<.001,(errors,jac_error,ikerror)
    write_json(args.root/'kinematics_audit.json',dict(max_fk_error_m=max(errors),jacobian_error=jac_error,ik_error_m=ikerror))
    print('kinematics audit passed',max(errors),jac_error,ikerror,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','rollout','audit']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--mode',default='visual',choices=['state','visual','uncertainty','memory','baseline','pilot_state_correction','pilot_visual','pilot_uncertainty','pilot_memory']);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--servo',type=float,default=.08);p.add_argument('--execute',type=int,default=15);p.add_argument('--screen',action='store_true');p.add_argument('--partition',default='validation',choices=['validation','test'])
    p.add_argument('--orientation',choices=['current','baseline'],default='current')
    p.add_argument('--output-root',type=Path,help='Fresh destination for reproducing rollouts with existing checkpoints')
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);seed_everything(SEED)
    globals()[args.stage](args)

if __name__=='__main__':main()
