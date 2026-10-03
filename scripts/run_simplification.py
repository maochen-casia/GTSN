"""Frozen-Pi3 simplification experiment. Execute in the project Docker image."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset

from run_fresh_consensus import SEED, sha256, source_hashes
from run_gtsn import build_history, load_data, inputs
from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import read_json, write_json
from tsn.common.seed import seed_everything
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.loaders import make_loader
from tsn.data.recovery_dataset import RecoveryDataset
from tsn.data.splits import episode_catalog, validate_splits
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.evaluation.open_loop import device_batch
from tsn.features.state import policy_state
from tsn.models.cartesian_policy import PandaKinematics
from tsn.models.consensus_policy import ConsensusPolicy
from tsn.models.fresh_consensus import FreshConsensus
from tsn.models.gtsn_policy import geometry_nll
from tsn.models.simplified_consensus import VARIANTS, compact_head


def reference(root, device='cuda'):
    checkpoint = load_checkpoint(root/'baseline.pt')
    assert checkpoint['epoch'] == 15
    model = FreshConsensus(checkpoint['config']['model'], initialize_backbone=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint['config'], checkpoint['splits']


@torch.inference_mode()
def cache(root):
    model, cfg, splits = reference(root)
    model.requires_grad_(False)
    data_root = Path(cfg['benchmark']['root'])
    catalog = episode_catalog(data_root)
    validate_splits(splits, catalog)
    for partition in ('train', 'validation'):
        directory = root/'cache'/partition
        directory.mkdir(parents=True, exist_ok=False)
        ids = splits[partition]
        expert = FrameDataset(data_root, ids, catalog, 30,
                              frame_stride=2 if partition == 'train' else 1, include_rgb=True)
        recovery = None
        if partition == 'train':
            recovery = RecoveryDataset(Path(cfg['train']['recovery_sources'][0]['path']), 30, include_rgb=True)
            assert set(recovery.route_by_episode) <= set(ids)
        dataset = ConcatDataset([expert, recovery]) if recovery is not None else expert
        specs = {'tokens': ((16, 768), np.float16), 'geometry': ((16, 6), np.float32),
                 'teacher': ((16, 3), np.float32), 'geometry_valid': ((16,), bool),
                 'baseline': ((30, 7), np.float32), 'state': ((16,), np.float32),
                 'pose': ((4, 4), np.float32), 'waypoint': ((6, 3), np.float32),
                 'tcp': ((4, 4), np.float32), 'route': ((), np.int64),
                 'episode': ((), np.int64), 'frame': ((), np.int64), 'source': ((), np.int64)}
        arrays = {k: np.lib.format.open_memmap(directory/f'{k}.npy', mode='w+', dtype=d,
                                              shape=(len(dataset), *s)) for k, (s, d) in specs.items()}
        lookup = {ep: i for i, ep in enumerate(ids)}
        batches = make_loader(dataset, dict(device='cuda', batch_size=128, num_workers=8), False, SEED)
        offset, started = 0, time.monotonic()
        for raw in batches:
            b = device_batch(raw, torch.device('cuda'))
            state = policy_state(b['qpos'], b['goal_pose'], model.maps.settings)
            base, tokens, geometry = model.backbone(b['rgb'], state, b['K'], b['T_B_C'], return_features=True)
            teacher = model.maps(b['depth'], b['K'], b['T_B_C'], b['goal_pose'][:, :3], b['future_ee'], b['valid_future'])
            depth = F.interpolate(b['depth'][:, None], (80, 80), mode='nearest-exact')
            valid = torch.isfinite(depth) & (depth >= model.maps.settings.near_m) & (depth <= model.maps.settings.far_m)
            tcp = model.kinematics(b['qpos'][:, :7])
            waypoint = model.kinematics(b['qpos'][:, None, :7]+b['target'][:, 4::5])[..., :3, 3]-tcp[:, None, :3, 3]
            values = dict(tokens=tokens, geometry=geometry, baseline=base, state=state,
                          pose=b['T_B_C'], waypoint=waypoint, tcp=tcp, route=b['route'], frame=b['frame_index'],
                          geometry_valid=F.adaptive_avg_pool2d(valid.float(), (4, 4)).flatten(1) >= .999,
                          teacher=F.adaptive_avg_pool2d(teacher[:, :3], (4, 4)).flatten(2).transpose(1, 2))
            size = len(state)
            for key, value in values.items():
                arrays[key][offset:offset+size] = value.float().cpu().numpy() if value.is_floating_point() else value.cpu().numpy()
            arrays['episode'][offset:offset+size] = [lookup[x.removesuffix('_recovery')] for x in raw['episode_id']]
            arrays['source'][offset:offset+size] = np.arange(offset, offset+size) >= len(expert)
            offset += size
            if offset % 2560 == 0 or offset == len(dataset):
                write_json(root/'status.json', dict(state='running', stage='cache', partition=partition, frames=offset, total=len(dataset)))
                print(f'cache {partition} {offset}/{len(dataset)} {time.monotonic()-started:.1f}s', flush=True)
        for value in arrays.values():
            value.flush()
        for key, value in zip(('history', 'ages', 'mask'), build_history(arrays['episode'], arrays['frame'], arrays['source'])):
            np.save(directory/f'{key}.npy', value)
        write_json(directory/'metadata.json', dict(samples=len(dataset), expert_samples=len(expert),
                   episode_ids=ids, backbone_sha256=sha256(root/'baseline.pt'), test_data_cached=False))
        expert.close()
        if recovery is not None:
            recovery.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def head_inputs(d, idx):
    return (*inputs(d, idx)[:6], d['tcp'][idx])


@torch.inference_mode()
def rmse(head, data):
    head.eval()
    total, count = 0., 0
    for idx in torch.arange(len(data['state']), device='cuda').split(256):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction, _ = head(*head_inputs(data, idx))
        total += float((prediction-data['waypoint'][idx]).square().sum())
        count += prediction.numel()
    return (total/count)**.5


def train(root, epochs):
    # No backbone is present during optimization; cached features are immutable.
    model, _, _ = reference(root, 'cpu')
    heads = {variant: compact_head(model, variant).cuda() for variant in VARIANTS}
    for head in heads.values():
        head.requires_grad_(True)
    del model
    gc.collect()
    tr, val = load_data(root/'cache/train'), load_data(root/'cache/validation')
    routes, sources = tr['route'].cpu().numpy(), tr['source'].cpu().numpy()
    weights = np.zeros(len(routes), dtype=np.float64)
    for s, fraction in enumerate((.65, .35)):
        for r, probability in enumerate((.2, .4, .4)):
            mask = (sources == s) & (routes == r)
            assert mask.any()
            weights[mask] = fraction*probability/mask.sum()
    records, best, selected, optimizers, schedulers = {}, {}, {}, {}, {}
    for variant, head in heads.items():
        out = root/'heads'/variant
        out.mkdir(parents=True)
        initial = rmse(head, val)
        records[variant] = [dict(epoch=0, validation_rmse_m=initial)]
        best[variant], selected[variant] = initial, 0
        torch.save(head.state_dict(), out/'initial.pt')
        torch.save(head.state_dict(), out/'best.pt')
        optimizers[variant] = torch.optim.AdamW(head.parameters(), lr=3e-5, weight_decay=1e-4)
        schedulers[variant] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[variant], epochs)
    generator = torch.Generator().manual_seed(SEED)
    n = read_json(root/'cache/train/metadata.json')['expert_samples']
    for epoch in range(1, epochs+1):
        order = torch.multinomial(torch.from_numpy(weights), n, replacement=True, generator=generator)
        order_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
        for variant, head in heads.items():
            started = time.monotonic()
            head.train()
            losses = []
            for idx in order.cuda().split(256):
                optimizers[variant].zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    # Match the baseline's separate per-branch supervision.
                    outputs = [branch(*head_inputs(tr, idx)) for branch in head.heads]
                    loss = sum(F.huber_loss(pred, tr['waypoint'][idx], delta=.02) for pred, _ in outputs)
                    for _, aux in outputs:
                        if 'logvar' in aux:
                            loss = loss+.001*geometry_nll(aux, tr['teacher'][idx], tr['geometry_valid'][idx])
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                optimizers[variant].step()
                losses.append(float(loss.detach()))
            score = rmse(head, val)
            if score < best[variant]:
                best[variant], selected[variant] = score, epoch
                torch.save(head.state_dict(), root/'heads'/variant/'best.pt')
            schedulers[variant].step()
            record = dict(epoch=epoch, validation_rmse_m=score, loss=float(np.mean(losses)),
                          sampling_sha256=order_hash, seconds=time.monotonic()-started)
            records[variant].append(record)
            write_json(root/'heads'/variant/'epochs.json', records[variant])
            write_json(root/'heads'/variant/'summary.json', dict(completed_epochs=epoch, selected_epoch=selected[variant],
                       best_validation_rmse_m=best[variant], parameters=sum(p.numel() for p in head.parameters()), frozen_backbone=True))
            write_json(root/'status.json', dict(state='running', stage='training', variant=variant, **record))
            print(variant, json.dumps(record), flush=True)


def rollout(args):
    if args.partition == 'test':
        selection = read_json(args.root/'selection.json')
        assert args.variant in (selection['selected'], 'control')
    model, config, splits = reference(args.root)
    head = compact_head(model, args.variant)
    head.load_state_dict(torch.load(args.root/'heads'/args.variant/'best.pt', map_location='cuda', weights_only=True))
    policy = ConsensusPolicy(model.backbone, head, model.kinematics,
                             servo_radius=.08, execute=15, orientation='baseline').cuda().eval()
    root = Path(config['benchmark']['root'])
    output = args.root/'rollouts'/args.partition/args.variant
    output.mkdir(parents=True, exist_ok=False)
    result = evaluate_rollouts(splits[args.partition], episode_catalog(root), root, output,
                              policy, model.maps, torch.device('cuda'), config['eval'])
    assert not result['privileged_action_map']
    assert len(result['results']) == 100
    write_json(output/'complete.json', dict(episodes=100, variant=args.variant, partition=args.partition,
               head_sha256=sha256(args.root/'heads'/args.variant/'best.pt'), backbone_sha256=sha256(args.root/'baseline.pt')))


def jobs(root, partition, variants):
    # Independent simulator processes share only the explicitly assigned idle GPU.
    logs = root/'logs'
    logs.mkdir(exist_ok=True)
    for start in range(0, len(variants), 3):
        running = []
        try:
            for variant in variants[start:start+3]:
                log = (logs/f'{partition}_{variant}.log').open('w')
                process = subprocess.Popen([sys.executable, __file__, 'rollout', '--root', str(root),
                    '--partition', partition, '--variant', variant], stdout=log, stderr=subprocess.STDOUT)
                running.append((variant, process, log))
            while any(p.poll() is None for _, p, _ in running):
                if any(p.poll() not in (None, 0) for _, p, _ in running):
                    raise RuntimeError('Rollout failed; inspect logs')
                write_json(root/'status.json', dict(state='running', stage='rollouts', partition=partition,
                    progress={v: len(list((root/'rollouts'/partition/v/'episodes').glob('*/metrics.json'))) for v, _, _ in running}))
                time.sleep(10)
            assert all(p.returncode == 0 for _, p, _ in running)
        finally:
            for _, process, log in running:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                log.close()


def finish(root):
    validation = {v: read_json(root/'rollouts/validation'/v/'closed_loop.json')['overall'] for v in VARIANTS}
    # Require less than five percentage points loss against both the original
    # baseline validation score and the equally fine-tuned control.
    eligible = [v for v in VARIANTS if v != 'control' and
                validation[v]['success_rate'] > max(.85, validation['control']['success_rate'])-.05+1e-8]
    summaries = {v: read_json(root/'heads'/v/'summary.json') for v in VARIANTS}
    selected = min(eligible, key=lambda v: (summaries[v]['parameters'], -validation[v]['success_rate'], v)) if eligible else 'control'
    write_json(root/'selection.json', dict(selected=selected, eligible=eligible, validation=validation,
               criterion='smallest head with validation success drop strictly below 5 percentage points vs original and matched control',
               test_used_for_selection=False, original_validation_success=.85, original_test_success=.78))
    jobs(root, 'test', list(dict.fromkeys(['control', selected])))
    test = {v: read_json(root/'rollouts/test'/v/'closed_loop.json')['overall'] for v in dict.fromkeys(['control', selected])}
    accepted = selected != 'control' and test[selected]['success_rate'] > max(.78, test['control']['success_rate'])-.05+1e-8
    model, config, splits = reference(root, 'cpu')
    final_variant = selected if accepted else 'control'
    head = compact_head(model, final_variant)
    head.load_state_dict(torch.load(root/'heads'/final_variant/'best.pt', map_location='cpu', weights_only=True))
    save_checkpoint(root/'simplified.pt', dict(format_version=1, architecture='compact_consensus', variant=final_variant,
        config=config, splits=splits, backbone=model.backbone.state_dict(), head=head.state_dict(),
        baseline_sha256=sha256(root/'baseline.pt'), selected_epoch=summaries[final_variant]['selected_epoch']))
    write_json(root/'results.json', dict(validation=validation, test=test, summaries=summaries,
               selected=selected, accepted=accepted, deployed_variant=final_variant,
               original_validation_success=.85, original_test_success=.78))
    assert sha256(root/'baseline.pt') == read_json(root/'protocol.json')['baseline_sha256']
    assert source_hashes() == read_json(root/'protocol.json')['source_sha256']
    orders = [[r['sampling_sha256'] for r in read_json(root/'heads'/v/'epochs.json')[1:]] for v in VARIANTS]
    assert all(order == orders[0] for order in orders)
    write_json(root/'audit.json', dict(passed=True, matched_sampling=True, frozen_backbone=True,
               baseline_unchanged=True, source_snapshot_unchanged=True, test_used_for_selection=False,
               validation_episodes_per_variant=100, test_episodes_per_variant=100))
    write_json(root/'status.json', dict(state='complete', selected=selected, accepted=accepted))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'rollout', 'finish'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--partition', choices=['validation', 'test'], default='validation')
    parser.add_argument('--variant', choices=VARIANTS, default='control')
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_everything(SEED)
    if args.stage == 'rollout':
        rollout(args)
        return
    try:
        if args.stage == 'all':
            assert 1 <= args.epochs <= 10
            digest = sha256(args.root/'baseline.pt')
            assert digest == '5f5e5c2b920a98f4d464a673e76eea372a15cd93e6ac748a235cddf2d5b696af'
            write_json(args.root/'protocol.json', dict(baseline_epoch=15, baseline_sha256=digest,
                source_sha256=source_hashes(), variants=VARIANTS, epochs=args.epochs, seed=SEED,
                lr=3e-5, batch_size=256, frozen_backbone=True, test_used_for_selection=False,
                docker_image_id=os.environ.get('GTSN_DOCKER_IMAGE_ID')))
            cache(args.root)
            train(args.root, args.epochs)
            gc.collect()
            torch.cuda.empty_cache()
            jobs(args.root, 'validation', list(VARIANTS))
        finish(args.root)
    except Exception:
        write_json(args.root/'status.json', dict(state='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
