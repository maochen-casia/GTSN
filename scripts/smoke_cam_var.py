"""Check real Pi3 forward/backward and live control before the four full runs."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from tsn.common.config import read_json, write_json
from tsn.common.seed import seed_everything
from tsn.data.navigation import NavigationDataset
from tsn.data.splits import episode_catalog, make_splits
from tsn.features.maps import GeometryMaps
from tsn.features.state import policy_state
from tsn.models.policy import NavigationPolicy
from tsn.training.runner import training_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    torch.set_num_threads(1)
    results = {}
    for variant in read_json(root/'protocol.json')['variants']:
        config = read_json(root/'configs'/f'{variant}.json')
        seed_everything(config['train']['seed'])
        model = NavigationPolicy(config['model']).cuda()
        maps = GeometryMaps(config['model']['maps']).cuda()
        splits = make_splits(config['benchmark'])
        catalog = episode_catalog(Path(config['benchmark']['root']))
        ids = [next(ep for ep in splits['train'] if catalog[ep] == route) for route in ('direct', 'over', 'side')]
        dataset = NavigationDataset(Path(config['benchmark']['root']), ids, catalog, 15,
                                    observation_hw=config['train']['observation_hw'], use_history=model.use_history)
        # Later expert frames exercise all causal slots and several real mounts.
        indices = [min(dataset.expert.offsets[i]+3, dataset.expert.offsets[i+1]-1) for i in range(3)]
        rows = [dataset[index] for index in indices]
        batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value
                 for key, value in next(iter(DataLoader(rows, batch_size=3))).items()}
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, terms = training_loss(model, maps, batch, config['train']['loss_weights'])
        loss.backward()
        encoder_gradients = [p.grad for p in model.perception.encoder.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in encoder_gradients)
        assert sum(float(g.abs().sum()) for g in encoder_gradients) > 0
        assert torch.isfinite(model.perception.point_head.weight.grad).all()
        if not model.use_clearance:
            assert all(p.grad is None for p in model.clearance.parameters())
        model.eval()
        state = policy_state(batch['qpos'][:1], batch['goal_pose'][:1], maps.settings)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(batch['history_rgb'][:1, -1], state, batch['K'][:1], batch['T_B_C'][:1])
        assert prediction.shape == (1, 30, 7) and torch.isfinite(prediction).all()
        results[variant] = dict(loss=float(loss), history_slots=batch['history_mask'].shape[1],
            encoder_gradient_tensors=len(encoder_gradients), finite_control=True,
            loss_components={name:float(value) for name, value in terms.items()})
        print(variant, results[variant], flush=True)
        dataset.close()
        del model, maps, batch, rows, loss, terms, prediction, encoder_gradients
        torch.cuda.empty_cache()
    write_json(root/'smoke_complete.json', results)


if __name__ == '__main__':
    main()
