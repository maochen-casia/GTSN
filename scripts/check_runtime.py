"""Verify CUDA training and real closed-loop rendering without changing archives."""
import argparse
from pathlib import Path
import tempfile

import torch
from torch.utils.data import default_collate

from tsn.common.config import read_json
from tsn.data.navigation import NavigationDataset
from tsn.data.splits import episode_catalog, make_splits
from tsn.evaluation.closed_loop import rollout
from tsn.features.maps import GeometryMaps
from tsn.models.policy import NavigationPolicy, load_policy
from tsn.training.runner import training_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('/workspace/configs/cam_var.json'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), 'CUDA device unavailable'
    device = torch.device('cuda')
    print('GPU:', torch.cuda.get_device_name(), 'torch:', torch.__version__,
          'CUDA:', torch.version.cuda, flush=True)
    config = read_json(args.config)
    root = Path(config['benchmark']['root'])
    catalog, splits = episode_catalog(root), make_splits(config['benchmark'])
    dataset = NavigationDataset(root, splits['train'][:1], catalog, 15,
                                observation_hw=config['train']['observation_hw'])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in default_collate([dataset[0]]).items()}
    model = NavigationPolicy(config['model']).to(device).train()
    maps = GeometryMaps(config['model']['maps']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['learning_rate'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, components = training_loss(model, maps, batch, config['train']['loss_weights'])
    assert torch.isfinite(loss), 'Non-finite training loss'
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer.step()
    print('Real RGB training forward/backward/AdamW step passed:', float(loss.detach()), flush=True)
    dataset.close()
    del model, maps, optimizer, batch, loss, components, gradients
    torch.cuda.empty_cache()
    policy, maps, saved = load_policy(args.checkpoint, device)
    episode = saved['splits']['validation'][0]
    options = dict(saved['config']['eval'], max_control_steps=15, render_videos=False)
    with tempfile.TemporaryDirectory(prefix='gtsn-runtime-') as directory:
        with torch.inference_mode():
            result = rollout(episode, catalog[episode], root, Path(directory),
                             policy, maps, device, options)
        assert result['control_steps'] > 0
        print('Real closed-loop observation, inference and physics passed:',
              episode, result['control_steps'], 'steps', flush=True)


if __name__ == '__main__':
    main()
