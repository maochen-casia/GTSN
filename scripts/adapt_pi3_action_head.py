"""Matched action-head adaptation with frozen maps and train-only updates."""
import argparse
import hashlib
import inspect
import time
from pathlib import Path

import numpy as np
import torch

from tsn.common.config import experiment_path, read_json, write_json
from tsn.common.seed import seed_everything
from tsn.evaluation.map_diagnostics import GROUPS, episode_bootstrap
from tsn.evaluation.metrics import PredictionMetrics
from tsn.models.geometry_policy import GeometryPolicy
from tsn.training.losses import imitation_loss

MODES = ['predicted', 'oracle_point_goal', 'oracle_all', 'zero_all']


def load_cache(root, mode):
    def read(key):
        return np.load(root / f'{key}.npy', mmap_mode='r')
    geometry = np.array(read('predicted'), copy=True)
    if mode == 'oracle_all':
        geometry[:] = read('teacher')
    elif mode == 'oracle_point_goal':
        geometry[:, :5] = read('teacher')[:, :5]
    elif mode == 'zero_all':
        geometry.fill(0)
    result = {'maps': torch.from_numpy(geometry).cuda()}
    for key in ('state', 'target', 'valid', 'route', 'episode'):
        result[key] = torch.from_numpy(np.array(read(key), copy=True)).cuda()
    return result


@torch.inference_mode()
def evaluate(model, data, batch_size=256):
    model.eval()
    metrics = PredictionMetrics(30)
    errors = []
    for start in range(0, len(data['state']), batch_size):
        stop = start + batch_size
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(data['maps'][start:stop].float(), data['state'][start:stop])
        metrics.update(prediction, data['target'][start:stop], data['valid'][start:stop], data['route'][start:stop])
        error = (prediction.float() - data['target'][start:stop]).square().sum(-1) * data['valid'][start:stop]
        errors.append(torch.stack([error.sum(1), error[:, :15].sum(1)], 1).cpu().numpy())
    return metrics.result(), np.concatenate(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    root = experiment_path(args.root)
    output = root / 'adapted_heads' / f'seed_{args.seed}' / args.mode
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    seed_everything(args.seed)
    cfg, options = read_json(root / 'head_config.json'), read_json(root / 'training_config.json')
    model = GeometryPolicy(**{key: cfg[key] for key in inspect.signature(GeometryPolicy).parameters}).cuda()
    model.load_state_dict(torch.load(root / 'action_head_initial.pt', weights_only=True, map_location='cuda'))
    train, val = load_cache(root / 'training_cache', args.mode), load_cache(root / 'validation_cache', args.mode)
    metadata = read_json(root / 'training_cache/metadata.json')
    routes = train['route'].cpu().numpy()
    sources = np.load(root / 'training_cache/source.npy')
    weights = np.zeros(len(routes), dtype=np.float64)
    for source, fraction in enumerate([.65, .35]):
        for route, probability in enumerate([.2, .4, .4]):
            mask = (sources == source) & (routes == route)
            assert mask.any()
            weights[mask] = fraction * probability / mask.sum()
    generator = torch.Generator().manual_seed(args.seed)
    sample_weights = torch.from_numpy(weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    initial, initial_error = evaluate(model, val)
    best_rmse, best_epoch, best_result, best_error = initial['rmse_rad'], 0, initial, initial_error
    torch.save(model.state_dict(), output / 'best.pt')
    write_json(output / 'protocol.json', {'mode': args.mode, 'seed': args.seed, 'epochs': args.epochs,
        'initialization': 'identical epoch-7 action head', 'initialization_sha256': hashlib.sha256((root / 'action_head_initial.pt').read_bytes()).hexdigest(),
        'updated_parameters': 'action policy only; Pi3 encoder and map heads frozen', 'training': metadata,
        'batch_size': 128, 'sampling': '65% expert/35% recovery; direct/over/side 20%/40%/40%',
        'learning_rate': options['learning_rate'], 'weight_decay': options['weight_decay'],
        'scheduler': 'cosine, 10 epochs', 'selection': 'lowest validation RMSE, including unadapted epoch 0',
        'teacher_maps_privileged': args.mode.startswith('oracle'), 'test_set_used': False,
        'initial_validation': initial})
    history = []
    seed_everything(args.seed)
    for epoch in range(1, args.epochs+1):
        started = time.monotonic()
        model.train()
        indices = torch.multinomial(sample_weights, metadata['expert_samples'], replacement=True, generator=generator)
        sampling_hash = hashlib.sha256(indices.numpy().tobytes()).hexdigest()
        indices = indices.cuda()
        total_loss = torch.zeros((), device='cuda')
        for start in range(0, len(indices), 128):
            index = indices[start:start+128]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(train['maps'][index].float(), train['state'][index])
                loss = imitation_loss(prediction, train['target'][index], train['valid'][index],
                                      options['huber_beta_rad'], options['horizon_decay_steps'], options['horizon_weight_floor'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss {args.mode}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip_norm'])
            optimizer.step()
            total_loss += loss.detach() * len(index)
        validation, errors = evaluate(model, val)
        if validation['rmse_rad'] < best_rmse:
            best_rmse, best_epoch, best_result, best_error = validation['rmse_rad'], epoch, validation, errors
            torch.save(model.state_dict(), output / 'best.pt')
        record = {'epoch': epoch, 'train_loss': float(total_loss / len(indices)), 'validation': validation,
                  'best_epoch': best_epoch, 'sampling_sha256': sampling_hash, 'elapsed_seconds': time.monotonic()-started}
        history.append(record)
        write_json(output / 'epochs.json', history)
        scheduler.step()
        print(f'{args.mode} seed {args.seed} epoch {epoch}/{args.epochs}: RMSE {validation["rmse_rad"]:.6f}, best {best_rmse:.6f}; {record["elapsed_seconds"]:.1f}s', flush=True)
    episode = val['episode'].cpu().numpy()
    count = len(read_json(root / 'protocol.json')['episode_ids'])
    squared = np.zeros((count, 2), dtype=np.float64)
    counts = np.zeros((count, 2), dtype=np.float64)
    np.add.at(squared, episode, best_error)
    n = val['valid'].sum(1).cpu().numpy()*7
    n15 = val['valid'][:, :15].sum(1).cpu().numpy()*7
    np.add.at(counts, episode, np.stack([n, n15], 1))
    baseline = np.load(root / 'paired_episode_errors.npz')
    result = {'mode': args.mode, 'seed': args.seed, 'completed_epochs': args.epochs, 'selected_epoch': best_epoch,
              'initial_validation': initial, 'selected_validation': best_result, 'final_validation': history[-1]['validation'],
              'executed_first15_rmse_rad': float(np.sqrt(squared[:, 1].sum()/counts[:, 1].sum())),
              'vs_original_policy': episode_bootstrap(squared[:, 0], counts[:, 0], baseline['predicted'][:, 0], baseline['episode_routes'])}
    np.savez(output / 'paired_episode_errors.npz', squared=squared, counts=counts)
    write_json(output / 'results.json', result)
    write_json(output / 'status.json', {'state': 'complete'})


if __name__ == '__main__':
    main()
