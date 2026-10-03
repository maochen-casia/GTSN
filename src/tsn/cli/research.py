"""Reproducible frozen-backbone route-evidence experiments (run in Docker)."""
import argparse
import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import create_output, read_json, write_json
from tsn.common.seed import seed_everything
from tsn.models.compact_policy import CompactRouteHead, compact_state
from tsn.models.evidence_policy import EvidenceRouteHead
from tsn.training.runner import sampling_weights


class ResidentCache:
    """Small frozen feature cache, loaded on one project GPU for repeated fits."""
    def __init__(self, root, device):
        names = ('tokens', 'geometry', 'pose', 'ages', 'mask', 'state', 'tcp',
                 'history', 'waypoint', 'teacher', 'geometry_valid', 'route', 'source')
        self.data = {k: torch.from_numpy(np.load(root/f'{k}.npy')).to(device) for k in names}
        self.device = device
        self.metadata = read_json(root/'metadata.json')

    def __len__(self):
        return len(self.data['state'])

    def inputs(self, indices):
        history = self.data['history'][indices]
        return tuple(self.data[k][history] for k in ('tokens', 'geometry', 'pose')) + tuple(
            self.data[k][indices] for k in ('ages', 'mask', 'state', 'tcp'))


@torch.inference_mode()
def score(head, data, batch_size=256):
    head.eval()
    squared, count = 0., 0
    for ids in torch.arange(len(data), device=data.device).split(batch_size):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction, _ = head(*data.inputs(ids))
        squared += float((prediction-data.data['waypoint'][ids]).square().sum())
        count += prediction.numel()
    return (squared/count)**.5


@torch.inference_mode()
def calibrate(head, data):
    total = torch.zeros(3, device=data.device)
    count = 0
    for ids in torch.arange(len(data), device=data.device).split(512):
        raw = head.base.distribution(head.base.visual(data.data['tokens'][ids].float())).float()
        mean = data.data['geometry'][ids, ..., :3]+.25*raw[..., :3].tanh()
        variance = (-5+4*raw[..., 3:].tanh()).exp()
        valid = data.data['geometry_valid'][ids]
        total += (((mean-data.data['teacher'][ids]).square()/variance)*valid[..., None]).sum((0, 1))
        count += int(valid.sum())
    if count == 0:
        raise ValueError('No valid training geometry')
    head.variance_scale.copy_((total/count).clamp(.1, 20))
    return head.variance_scale.tolist()


def train(args):
    output = args.output_dir
    create_output(output)
    device = torch.device('cuda')
    seed_everything(args.seed)
    checkpoint = load_checkpoint(args.checkpoint)
    _, weights = compact_state(checkpoint)
    base = CompactRouteHead().to(device)
    base.load_state_dict(weights)
    head_config = dict(variant=args.variant, radius=args.radius, correction=args.correction)
    head = EvidenceRouteHead(base, **head_config).to(device)
    training = ResidentCache(args.cache/'train', device)
    validation = ResidentCache(args.cache/'validation', device)
    for split, data in [('train', training), ('validation', validation)]:
        if data.metadata['episode_ids'] != checkpoint['splits'][split]:
            raise ValueError('Cache and checkpoint splits differ')
    if training.metadata['backbone_sha256'] != read_json(args.cache.parent/'protocol.json')['baseline_sha256']:
        raise ValueError('Cache backbone provenance mismatch')
    variance_scale = calibrate(head, training)
    protocol = dict(architecture='route_evidence', head_config=head_config,
                    base_checkpoint=str(args.checkpoint), cache=str(args.cache), seed=args.seed,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    variance_scale=variance_scale, selection='minimum validation waypoint RMSE, including epoch zero',
                    frozen_backbone=True, frozen_proposal=True, test_used_for_selection=False,
                    splits=checkpoint['splits'])
    write_json(output/'protocol.json', protocol)
    trainable = [p for p in head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    weights = sampling_weights(training.data['route'].cpu().numpy(), training.data['source'].cpu().numpy())
    generator = torch.Generator().manual_seed(args.seed)
    records, best, selected = [], float('inf'), 0
    expert_samples = training.metadata['expert_samples']
    for epoch in range(args.epochs+1):
        started = time.monotonic()
        total, digest = 0., None
        if epoch:
            head.train()
            order = torch.multinomial(weights, expert_samples, replacement=True, generator=generator)
            digest = hashlib.sha256(order.numpy().tobytes()).hexdigest()
            for ids in order.to(device).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction, _ = head(*training.inputs(ids))
                    loss = F.huber_loss(prediction, training.data['waypoint'][ids], delta=.02)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite route loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(ids)
            scheduler.step()
        rmse = score(head, validation, args.batch_size)
        record = dict(epoch=epoch, rmse_m=rmse, loss=total/expert_samples,
                      sampling_sha256=digest, seconds=time.monotonic()-started)
        records.append(record)
        if rmse < best:
            best, selected = rmse, epoch
            torch.save(dict(architecture='route_evidence', base_checkpoint=str(args.checkpoint),
                            head_config=head_config, head=head.state_dict(), epoch=epoch,
                            config=checkpoint['config'], splits=checkpoint['splits']), output/'best.pt')
        write_json(output/'epochs.json', records)
        print(record, flush=True)
    write_json(output/'complete.json', dict(selected_epoch=selected, rmse_m=best,
               new_parameters=sum(p.numel() for p in trainable), variant=args.variant))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=EvidenceRouteHead.variants, default='full')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--radius', type=float, default=.12)
    parser.add_argument('--correction', type=float, default=.06)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(args)


if __name__ == '__main__':
    main()
