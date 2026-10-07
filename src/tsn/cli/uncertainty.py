"""Fit a metric error-quantile head using existing expert/perturbation targets."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import create_output, read_json, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.models.compact_policy import CompactRouteHead, compact_state
from tsn.models.uncertain_clearance import PointUncertainty, uncertainty_features, padding_factor


def prepare(path, head, split, device):
    metadata = read_json(path/'metadata.json')
    if metadata['episode_ids'] != split or metadata.get('test_data_cached', False):
        raise ValueError('Unverified split or test data in uncertainty calibration')
    keys = ('tokens', 'points', 'point_teacher', 'point_valid', 'pose', 'tcp', 'source', 'route')
    arrays = {key: torch.from_numpy(np.load(path/(key+'.npy'), allow_pickle=False)).to(device) for key in keys}
    if not torch.isin(arrays['source'], arrays['source'].new_tensor([0, 1])).all():
        raise ValueError('Only original expert and independent perturbation observations are allowed')
    valid = arrays['point_valid'].bool() & torch.isfinite(arrays['points']).all(-1) & torch.isfinite(arrays['point_teacher']).all(-1)
    p = arrays['points'].float()
    valid &= (p[..., 0]>.10)&(p[..., 0]<1.05)&(p[..., 1].abs()<.6)&(p[..., 2]>.04)&(p[..., 2]<.65)
    valid &= (p-arrays['tcp'][:, None, :3, 3]).norm(dim=-1)>.07
    error = (arrays['points'].float()-arrays['point_teacher'].float()).norm(dim=-1)
    error = torch.where(valid, error, torch.zeros_like(error))
    features = []
    with torch.inference_mode():
        for ids in torch.arange(len(p), device=device).split(128):
            visual = head.visual(arrays['tokens'][ids].float())
            features.append(uncertainty_features(visual, p[ids], arrays['pose'][ids], arrays['tcp'][ids]).half())
    return dict(features=torch.cat(features), target=error, valid=valid,
                source=arrays['source'], route=arrays['route']), metadata


def pinball(prediction, target, valid):
    residual = target-prediction
    loss = torch.maximum(.9*residual, -.1*residual)
    per_row = (loss*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return per_row.mean()


@torch.no_grad()
def calibration(model, data):
    predicted = torch.cat([model(x.float()) for x in data['features'].split(128)])
    valid = data['valid']
    p, target = predicted[valid], data['target'][valid]
    if not len(p):
        raise ValueError('No valid reconstruction targets')
    record = dict(valid_points=len(p), pinball_loss_m=float(pinball(predicted, data['target'], valid)),
        empirical_coverage=float((target<=p).float().mean()), mean_radius_m=float(p.mean()),
        mean_error_m=float(target.mean()), predicted_radius_quantiles_m=torch.quantile(p, p.new_tensor([.1,.5,.9])).tolist(),
        observed_error_quantiles_m=torch.quantile(target, target.new_tensor([.1,.5,.9])).tolist(),
        mean_padding_factor=float(padding_factor(p).mean()))
    bounds = torch.quantile(p, p.new_tensor([0,.2,.4,.6,.8,1]))
    record['prediction_bins'] = []
    for i in range(5):
        mask = (p>=bounds[i]) & (p<=bounds[i+1] if i==4 else p<bounds[i+1])
        if mask.any():
            record['prediction_bins'].append(dict(points=int(mask.sum()), mean_prediction_m=float(p[mask].mean()),
                mean_error_m=float(target[mask].mean()), coverage=float((target[mask]<=p[mask]).float().mean())))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy-checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Use at least one fixed training epoch')
    create_output(args.output_dir)
    device = require_device(args.device)
    torch.set_num_threads(1)
    seed_everything(20261008)
    energy = load_checkpoint(args.energy_checkpoint)
    if energy['architecture'] != 'geometric_energy':
        raise ValueError('Expected the existing expert/perturbation-only energy checkpoint')
    parent = load_checkpoint(Path(energy['parent_checkpoint']))
    head = CompactRouteHead().to(device).eval().requires_grad_(False)
    head.load_state_dict(compact_state(parent)[1], strict=True)
    train, metadata = prepare(args.cache/'train', head, parent['splits']['train'], device)
    validation, _ = prepare(args.cache/'validation', head, parent['splits']['validation'], device)
    weights = torch.zeros(len(train['source']), dtype=torch.float64)
    for source, mix in enumerate((.65,.35)):
        for route, ratio in enumerate((.2,.4,.4)):
            group = ((train['source']==source)&(train['route']==route)).cpu()
            weights[group] = mix*ratio/int(group.sum())
    protocol = dict(architecture='74→64→1 metric point-error quantile', parameters=4865,
        source='Existing expert and independent perturbation-only RGB point targets',
        cache=str(args.cache), observations=len(weights), expert_observations=int((train['source']==0).sum()),
        perturbation_observations=int((train['source']==1).sum()), source_mix=[.65,.35], route_mix=[.2,.4,.4],
        quantile=.9, radius_range_m=[.005,.2], objective='Per-observation mean pinball loss over valid point targets',
        epochs=args.epochs, learning_rate=.001, draws_per_epoch=8192, batch_rows=64, seed=20261008,
        geometry_targets='Existing recorded-depth teacher points; training supervision only',
        deployment_inputs='Current RGB-derived visual cells, metric points/roughness, measured camera and TCP',
        additional_data=False, test_used=False, checkpoint_selection='Fixed final epoch before rollouts')
    write_json(args.output_dir/'protocol.json', protocol)
    model = PointUncertainty().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(20261008)
    orders = hashlib.sha256()
    records = []
    started = time.monotonic()
    for epoch in range(args.epochs+1):
        if epoch:
            model.train()
            order = torch.multinomial(weights, 8192, replacement=True, generator=generator)
            orders.update(order.numpy().tobytes())
            for ids in order.to(device).split(64):
                optimizer.zero_grad(set_to_none=True)
                predicted = model(train['features'][ids].float())
                loss = pinball(predicted, train['target'][ids], train['valid'][ids])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
        model.eval()
        record = dict(epoch=epoch, validation=calibration(model, validation))
        records.append(record)
        write_json(args.output_dir/'epochs.json', records)
        print(record, flush=True)
    train_calibration = calibration(model, train)
    write_json(args.output_dir/'train_calibration.json', train_calibration)
    saved = dict(format_version=1, architecture='uncertainty_clearance', energy_checkpoint=str(args.energy_checkpoint.resolve()),
        energy_sha256=hashlib.sha256(args.energy_checkpoint.read_bytes()).hexdigest(),
        uncertainty={k:v.cpu() for k,v in model.state_dict().items()}, uniform_factor=train_calibration['mean_padding_factor'],
        training=protocol)
    save_checkpoint(args.output_dir/'uncertainty.pt', saved)
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for partition in ('train','validation') for path in sorted((args.cache/partition).glob('*.npy'))
              if path.stem in ('tokens','points','point_teacher','point_valid','pose','tcp','source','route')}
    write_json(args.output_dir/'input_sha256.json', inputs)
    write_json(args.output_dir/'complete.json', dict(seconds=time.monotonic()-started, selected_epoch=args.epochs,
        optimizer_test_access=False, sampling_sha256=orders.hexdigest(), final_validation=records[-1]['validation'],
        train_uniform_factor=saved['uniform_factor']))


if __name__ == '__main__':
    main()
