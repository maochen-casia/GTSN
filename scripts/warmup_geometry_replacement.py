"""Initialize new attention modules using strictly training-only geometry labels.

    This cache is used only for module warmup. Subsequent policy fine tuning
    reads RGB again and trains every existing head with only Pi3 frozen.
"""
import argparse
import gc
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from tsn.common.checkpoint import load_checkpoint
from tsn.common.config import create_output, read_json, write_json
from tsn.common.seed import seed_everything
from tsn.models.c1_memory import PersistentGeometry
from tsn.models.c2_embodiment import AttentionEmbodimentGeometry, EmbodimentGeometry
from tsn.models.c3_clearance import route_candidates
from tsn.training.runner import retention_targets


def move(value, device):
    if isinstance(value, torch.Tensor):return value.to(device)
    if isinstance(value, dict):return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):return [move(v, device) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.15)
    config = read_json(args.config)
    seed_everything(config['train']['seed'])
    parent = load_checkpoint(config['train']['initialize_from'])
    manifest = read_json(args.cache/'manifest.json')
    if (manifest['partition'] != 'train' or manifest['train_episodes'] != parent['splits']['train'] or
            not Path(manifest['parent_checkpoint']).samefile(config['train']['initialize_from'])):
        raise ValueError('Warmup requires the identical parent and its training partition')
    del parent; gc.collect()
    examples = move(torch.load(args.cache/'examples.pt', weights_only=True), 'cuda')
    settings = config['model']['learned_geometry']
    memory = PersistentGeometry(**config['model']['memory'], replacement=True,
                                hidden_dim=settings['width'], depth=settings['depth']).cuda()
    body = AttentionEmbodimentGeometry(config['model']['robot'], settings['width'], settings['depth'],
                                      settings.get('calibrated_risk', False)).cuda()
    parameters = []
    if settings['c1']:parameters += list(memory.selector.parameters())
    if settings['c2']:parameters += list(body.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)
    create_output(args.output_dir)
    cached = []
    if settings['c2'] and body.calibrated:
        calibrate = torch.optim.AdamW(body.risk_head.parameters(), lr=.003, weight_decay=0)
        for _ in range(2048):
            loss = body.calibration_loss()
            calibrate.zero_grad(set_to_none=True); loss.backward(); calibrate.step()
        print(f'Neural distance calibration MSE {float(loss):.8g}', flush=True)
    if settings['c2']:
        with torch.no_grad():
            for number, example in enumerate(examples, 1):
                original = PersistentGeometry(**config['model']['memory'])
                for frame in example['history']:
                    p, u = frame['points'], frame['radius']
                    original.update(p, torch.ones(len(p), dtype=torch.bool, device=p.device), u, frame['step'],
                                    frame['tcp'][:3, 3], example['goal'])
                p, u = original.query()
                candidates, _ = route_candidates(example['positions'][None], example['tcp'][None], example['goal'][None])
                padding = config['model']['clearance']['max_padding']*((u-.015)/.085).clamp(0, 1)
                context = {k: example[k] for k in ('state', 'goal', 'arm_points')}
                values = body.descriptors(candidates[0], example['rotations'], example['tcp'], p, padding,
                                          example['fingers'], example['mount'], context,
                                          config['model']['clearance']['margin'])
                if values is None:continue
                if values['scene'].shape[-2] < 128:
                    values['scene'] = values['scene'][..., torch.arange(128, device=p.device)%values['scene'].shape[-2], :]
                values['state'] = values['state'][None].expand(14, -1)
                teacher = EmbodimentGeometry.contact_risk(body, candidates[0], example['rotations'], example['tcp'],
                    p, padding, example['fingers'], config['model']['clearance']['margin'], example['mount'])
                cached.append((move(values, 'cpu'), teacher.cpu()))
                if number % 256 == 0:
                    write_json(args.output_dir/'progress.json', dict(phase='descriptors', examples=number))
        print(f'Prepared {len(cached)} training-only risk descriptors', flush=True)
    records = []
    for epoch in range(1, args.epochs+1):
        started = time.monotonic(); total_memory = total_body = 0.; memory_steps = body_steps = 0
        if settings['c1']:
            for index in torch.randperm(len(examples)).tolist():
                example = examples[index]; memory.reset()
                with torch.no_grad():
                    for frame in example['history']:
                        p = frame['points']
                        memory.update(p, torch.ones(len(p), dtype=torch.bool, device=p.device), frame['radius'],
                                      frame['step'], frame['tcp'][:3, 3], example['goal'], frame['state'])
                p = torch.cat([f['points'] for f in example['history']])
                if not len(p):continue
                u = torch.cat([f['radius'] for f in example['history']])
                label_mask = torch.cat([f['label_mask'] for f in example['history']])
                if not label_mask.any():continue
                reliable = torch.cat([f['reliability'] for f in example['history']])
                age = torch.cat([p.new_full((len(f['points']),), memory.step-f['step']) for f in example['history']])
                with torch.no_grad():
                    _, nearest = torch.cdist(p, memory.points).min(-1)
                    target = retention_targets(p, reliable, example['tcp'][:3, 3]+example['target'])
                scores = memory.selector(p, u, memory.support[nearest], memory.scatter[nearest], age,
                                         example['tcp'][:3, 3], example['goal'], example['state'])
                loss = F.binary_cross_entropy_with_logits(scores[label_mask], target[label_mask])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
                total_memory += float(loss); memory_steps += 1
        if settings['c2']:
            for indices in torch.randperm(len(cached)).split(4):
                rows = [cached[i] for i in indices.tolist()]
                values = {k: torch.stack([r[0][k] for r in rows]).cuda() for k in ('body', 'scene', 'state')}
                values['ids'] = rows[0][0]['ids'].cuda()
                target = torch.stack([r[1] for r in rows]).cuda()
                predicted = body.predict_descriptors(values)
                # Calibrate absolute risk and relative candidate differences.
                loss = F.mse_loss(predicted, target)+F.mse_loss(predicted-predicted[:, :1], target-target[:, :1])
                if body.calibrated:loss += body.calibration_loss()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite warmup loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
                total_body += float(loss); body_steps += 1
        record = dict(epoch=epoch, memory_loss=total_memory/max(1, memory_steps),
                      body_loss=total_body/max(1, body_steps), seconds=time.monotonic()-started)
        records.append(record); write_json(args.output_dir/'epochs.json', records); print(record, flush=True)
    state = {}
    if settings['c1']:state.update({'memory.selector.'+k: v.detach().cpu() for k, v in memory.selector.state_dict().items()})
    if settings['c2']:
        physical = set(EmbodimentGeometry(config['model']['robot']).state_dict())
        state.update({'embodiment.'+k: v.detach().cpu() for k, v in body.state_dict().items() if k not in physical})
    torch.save(state, args.output_dir/'modules.pt')
    write_json(args.output_dir/'complete.json', dict(epochs=args.epochs, partition='train', examples=len(examples),
        only_new_modules_initialized=True, subsequent_training='all heads; only Pi3 encoder frozen'))


if __name__ == '__main__':main()
