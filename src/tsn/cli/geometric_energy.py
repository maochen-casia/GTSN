"""Fit explicit geometric candidate energies on expert + perturbation data."""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import create_output, write_json, read_json
from tsn.common.seed import seed_everything, require_device
from tsn.models.compact_policy import CompactRouteHead, compact_state, goal_xyz
from tsn.models.clearance_policy import ClearancePolicy
from tsn.models.geometric_energy import RouteTrust, surface_risk, trust_features


def prepare(cache, parent, device):
    metadata = read_json(cache/'metadata.json')
    if metadata.get('test_data_cached', False):
        raise ValueError('Test data cannot enter training')
    keys = ('tokens', 'geometry', 'pose', 'state', 'tcp', 'waypoint', 'points', 'source', 'route', 'frame', 'episode')
    a = {k: torch.from_numpy(np.load(cache/f'{k}.npy', allow_pickle=False)).to(device) for k in keys}
    if not torch.isin(a['source'], a['source'].new_tensor([0, 1])).all():
        raise ValueError('Only expert and perturbation-only recovery are permitted')
    # Causal 0/15/30/45 frame context, matching the cache's sequence cadence.
    rows = torch.arange(len(a['source']), device=device)
    history = rows[:, None].expand(-1, 4).clone()
    mask = torch.zeros_like(history, dtype=torch.bool)
    mask[:, -1] = True
    ep, source, frame = (a[k].cpu().numpy() for k in ('episode', 'source', 'frame'))
    for e in np.unique(ep):
        indices = np.flatnonzero((ep == e) & (source == 0))
        for n, i in enumerate(indices):
            context = indices[max(0, n-3):n+1]
            if len(context)>1 and not np.all(np.diff(frame[context]) > 0):
                raise ValueError('Expert observation timestamps must strictly increase')
            history[i, -len(context):] = torch.as_tensor(context, device=device)
            mask[i, -len(context):] = True
    outputs = {k: [] for k in ('features', 'risk', 'norm', 'target', 'regret')}
    coordinates = a['tcp'].new_tensor(ClearancePolicy.correction_coordinates)
    for ids in rows.split(64):
        h = history[ids]
        ages = a['frame'][ids, None]-a['frame'][h]
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            prediction, _ = parent(a['tokens'][h], a['geometry'][h], a['pose'][h], ages, mask[ids], a['state'][ids], a['tcp'][ids])
        tcp, goal = a['tcp'][ids], goal_xyz(a['state'][ids])
        knots = torch.cat((torch.zeros_like(prediction[:, :1]), prediction), 1)
        t = torch.arange(1, 31, device=device)/5
        lo = t.long().clamp(max=5)
        alpha = (t-lo)[None, :, None]
        waypoints = tcp[:, None, :3, 3]+knots[:, lo]*(1-alpha)+knots[:, lo+1]*alpha
        near = (goal-tcp[:, :3, 3]).norm(dim=-1)<.08
        direct = tcp[:, None, :3, 3]+(goal-tcp[:, :3, 3])[:, None]*torch.linspace(1/30, 1, 30, device=device)[None, :, None]
        waypoints = torch.where(near[:, None, None], direct, waypoints)
        direction = goal-tcp[:, :3, 3]
        direction[:, 2] = 0
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(.01)
        side = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])), -1)
        up = torch.zeros_like(side); up[:, 2] = 1
        offsets = coordinates[None, :, :1]*side[:, None]+coordinates[None, :, 1:]*up[:, None]
        fade = (((goal-tcp[:, :3, 3]).norm(dim=-1)-.025)/.055).clamp(0, 1)
        ramp = torch.linspace(1/30, 1, 30, device=device).sin()*1.1883951
        candidates = waypoints[:, None]+offsets[:, :, None]*ramp[None, None, :, None]*fade[:, None, None, None]
        points = a['points'][h].float().flatten(1, 2)
        valid = mask[ids, :, None].expand(-1, -1, 400).flatten(1)
        valid = valid & (points[..., 0]>.10) & (points[..., 0]<1.05) & (points[..., 1].abs()<.60) & (points[..., 2]>.04) & (points[..., 2]<.65)
        rotation = tcp[:, None, :3, :3].expand(-1, 30, -1, -1)
        with torch.inference_mode():
            risk = surface_risk(candidates, rotation, tcp, points, valid)
            regret = (candidates[:, :, 4::5]-tcp[:, None, None, :3, 3]-a['waypoint'][ids, None]).square().mean((-1, -2))
            target = (-regret/(.015**2)).softmax(-1)
            outputs['features'].append(trust_features(waypoints, tcp, goal, risk))
            outputs['risk'].append(risk)
            outputs['norm'].append((offsets.norm(dim=-1)/.05).square())
            outputs['regret'].append(regret)
            outputs['target'].append(target)
    return {k: torch.cat(v) for k, v in outputs.items()}, a, metadata


def objective(model, data, ids):
    penalty = model(data['features'][ids])
    cost = data['risk'][ids]+penalty[:, None]*data['norm'][ids]
    logp = (-cost/.03).log_softmax(-1)
    loss = -(data['target'][ids]*logp).sum(-1)
    return loss.mean()+.1*((penalty-.08)/.04).square().mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    create_output(args.output_dir)
    device = require_device(args.device)
    torch.set_num_threads(1); seed_everything(20261007)
    checkpoint = load_checkpoint(args.checkpoint)
    for partition in ('train', 'validation'):
        metadata = read_json(args.cache/partition/'metadata.json')
        if metadata['episode_ids'] != checkpoint['splits'][partition]:
            raise ValueError('Cache split mismatch')
    parent = CompactRouteHead().to(device).eval().requires_grad_(False)
    parent.load_state_dict(compact_state(checkpoint)[1], strict=True)
    train, arrays, metadata = prepare(args.cache/'train', parent, device)
    validation, _, vm = prepare(args.cache/'validation', parent, device)
    if metadata['episode_ids'] != checkpoint['splits']['train'] or vm['episode_ids'] != checkpoint['splits']['validation']:
        raise ValueError('Cache split mismatch')
    weights = torch.zeros(len(arrays['source']), dtype=torch.float64)
    for s, mix in enumerate((.65, .35)):
        for r, ratio in enumerate((.2, .4, .4)):
            group = ((arrays['source']==s)&(arrays['route']==r)).cpu()
            weights[group] = mix*ratio/int(group.sum())
    protocol = dict(seed=20261007, epochs=args.epochs, draws_per_epoch=32768, batch_size=256,
        source_mix=[.65, .35], route_mix=[.2, .4, .4], additional_data=False,
        sources='Expert routes plus original independent perturbation-only recovery; no corrective/on-policy data',
        cache=str(args.cache), observations=len(weights), recovery_observations=int((arrays['source']==1).sum()),
        frozen_parent=str(args.checkpoint), training_rotation='Current TCP rotation repeated across the 30-step chunk',
        context='Last four cached observations at nominal 15-step cadence rounded to original cache rows; actual timestamps retained; independent recovery resets',
        deployment_rotation='Frozen backbone proposed wrist rotation, unchanged shared controller',
        objective='Expert candidate-distribution cross entropy at temperature .03 plus .1 trust regularization',
        target_temperature_m=.015, learning_rate=.001, architecture='8→32→1, positive trust coefficient [.02,.12]',
        checkpoint_rule='Epoch 20 fixed in advance; epoch zero is a fixed-trust control', test_used=False)
    write_json(args.output_dir/'protocol.json', protocol)
    model = RouteTrust().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(20261007)
    records = []; order_hash = hashlib.sha256(); start = time.monotonic()
    val_ids = torch.arange(len(validation['risk']), device=device)
    for epoch in range(args.epochs+1):
        if epoch:
            order = torch.multinomial(weights, 32768, replacement=True, generator=generator)
            order_hash.update(order.numpy().tobytes())
            model.train()
            for ids in order.to(device).split(256):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model, train, ids)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            coefficient = model(validation['features'])
            cost = validation['risk']+coefficient[:, None]*validation['norm']
            choice = cost.argmin(-1)
            regret = validation['regret'][val_ids, choice]
            record = dict(epoch=epoch, validation_loss=float(objective(model, validation, val_ids)),
                validation_candidate_rmse_m=float(regret.mean().sqrt()),
                trust_mean=float(coefficient.mean()), trust_min=float(coefficient.min()), trust_max=float(coefficient.max()))
        records.append(record); print(record, flush=True)
        write_json(args.output_dir/'epochs.json', records)
    exported = dict(format_version=1, architecture='geometric_energy', parent_checkpoint=str(args.checkpoint),
        parent_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        trust={k: v.cpu() for k, v in model.state_dict().items()}, training=protocol)
    save_checkpoint(args.output_dir/'energy.pt', exported)
    write_json(args.output_dir/'complete.json', dict(seconds=time.monotonic()-start, test_used=False,
        sampling_sha256=order_hash.hexdigest(), selected_epoch=args.epochs, initial=records[0], final=records[-1]))


if __name__ == '__main__':
    main()
