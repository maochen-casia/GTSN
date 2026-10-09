"""Verify real GPU optimization and native-camera closed-loop baseline inputs."""
import argparse
import copy
from pathlib import Path
import torch
from tsn.baselines.policies import BaselinePolicy, METHODS
from tsn.common.config import read_json, write_json
from tsn.data.splits import episode_catalog
from tsn.evaluation.closed_loop import rollout
from tsn.features.maps import GeometryMaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.1)
    device = torch.device('cuda')
    config = read_json(root/'base_config.json')
    splits = read_json(root/'cache/splits.json')
    dataset = Path(config['benchmark']['root'])
    catalog = episode_catalog(dataset)
    output = root/'smoke'
    output.mkdir(exist_ok=False)
    receipts = {}
    for method in METHODS:
        adapted = read_json(root/'configs'/f'{method}.json')
        model = BaselinePolicy(adapted).to(device)
        batch = {'rgb': torch.randint(256, (8, 2, 84, 112, 3), dtype=torch.uint8, device=device),
            'points': torch.randn(8, 2, 512, 3, device=device), 'state': torch.randn(8, 2, 34, device=device),
            'target': torch.randn(8, 30, 7, device=device)*.1}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss = model.loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        options = {**config['eval'], 'max_control_steps': 15}
        directory = output/method
        directory.mkdir()
        episode = splits['validation'][0]
        result = rollout(episode, catalog[episode], dataset, directory, model,
                         GeometryMaps(config['model']['maps']).to(device), device, options)
        assert 0 < result['control_steps'] <= 15
        assert result['baseline_method'] == method
        assert result['depth_input_at_inference'] == model.requires_depth
        receipts[method] = {'gpu_loss': float(loss.detach()), 'control_steps': result['control_steps'],
            'depth_input_at_inference': model.requires_depth, 'baseline_method': result['baseline_method'],
            'smoke_only': True, 'test_partition_used': False}
        print(receipts[method], flush=True)
        del model, optimizer, batch
        torch.cuda.empty_cache()
    write_json(root/'smoke_complete.json', receipts)


if __name__ == '__main__':
    main()
