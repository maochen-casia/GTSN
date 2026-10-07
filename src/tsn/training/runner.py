"""Fresh main-model training on expert routes and independent perturbations."""
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler

from tsn.common.checkpoint import save_checkpoint
from tsn.common.config import create_output, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.data.loaders import make_loader
from tsn.data.navigation import NavigationDataset, sampling_weights
from tsn.data.splits import make_splits, episode_catalog
from tsn.features.maps import GeometryMaps
from tsn.features.state import policy_state
from tsn.models.c1_memory import PersistentGeometry, workspace_mask
from tsn.models.policy import NavigationPolicy, metric_points
from tsn.models.route import cartesian_proposal, goal_position
from tsn.training.losses import geometry_nll, map_loss, point_quantile_loss


def encode_observations(model, maps, batch):
    """Past features are detached; the current RGB observation trains perception."""
    count = len(batch['qpos'])
    states = policy_state(batch['history_qpos'].flatten(0, 1),
                          batch['history_goal_pose'].flatten(0, 1), maps.settings).reshape(count, 4, 16)
    current = model.perception(batch['history_rgb'][:, -1], states[:, -1],
                               batch['K'], batch['T_B_C'])
    joint, tokens, geometry, dense = current
    previous_tokens = tokens.new_zeros(count*3, 16, 768)
    previous_geometry = geometry.new_zeros(count*3, 16, 6)
    previous_points = torch.zeros(count*3, 400, 3, device=tokens.device)
    valid = batch['history_mask'][:, :3].flatten()
    if valid.any():
        with torch.no_grad():
            _, old_tokens, old_geometry, old_dense = model.perception(
                batch['history_rgb'][:, :3].flatten(0, 1)[valid], states[:, :3].flatten(0, 1)[valid],
                batch['history_K'][:, :3].flatten(0, 1)[valid], batch['history_T_B_C'][:, :3].flatten(0, 1)[valid])
            previous_tokens[valid], previous_geometry[valid] = old_tokens, old_geometry
            previous_points[valid] = metric_points(old_dense)
    return dict(joint=joint, dense=dense, state=states[:, -1],
        tokens=torch.cat((previous_tokens.reshape(count, 3, 16, 768), tokens[:, None]), 1),
        geometry=torch.cat((previous_geometry.reshape(count, 3, 16, 6), geometry[:, None]), 1),
        points=torch.cat((previous_points.reshape(count, 3, 400, 3), metric_points(dense)[:, None]), 1))


def predict_route(model, maps, batch):
    features = encode_observations(model, maps, batch)
    with torch.autocast(device_type=batch['qpos'].device.type, enabled=False):
        tcp = model.kinematics(batch['qpos'][:, :7])
        target = model.kinematics(batch['qpos'][:, None, :7]+batch['target'][:, 4::5])[..., :3, 3]-tcp[:, None, :3, 3]
    prediction, auxiliary = model.route(features['tokens'], features['geometry'], batch['history_T_B_C'],
        batch['history_ages'], batch['history_mask'], features['state'], tcp)
    return features, tcp, prediction, target, auxiliary


def training_loss(model, maps, batch, weights):
    features, tcp, prediction, target, auxiliary = predict_route(model, maps, batch)
    with torch.autocast(device_type=tcp.device.type, enabled=False):
        teacher = maps(batch['depth'], batch['K'], batch['T_B_C'], batch['goal_pose'][:, :3],
                       batch['future_ee'], batch['valid_future'])
        depth = F.interpolate(batch['depth'][:, None], teacher.shape[-2:], mode='nearest-exact')[:, 0]
        observed = torch.isfinite(depth) & (depth >= maps.settings.near_m) & (depth <= maps.settings.far_m)
        teacher_cells = F.adaptive_avg_pool2d(teacher[:, :3], (4, 4)).flatten(2).transpose(1, 2)
        cells_valid = F.adaptive_avg_pool2d(observed[:, None].float(), (4, 4)).flatten(1) >= .999
        points = features['points'][:, -1].detach()
        radii = model.clearance.predict_error(model.route.visual(features['tokens'][:, -1].float()).detach(),
                                              points, batch['T_B_C'], tcp)
        teacher_points = metric_points(teacher)
        point_valid = F.interpolate(observed[:, None].float(), (20, 20), mode='nearest-exact').flatten(1).bool()
        point_valid &= workspace_mask(points) & ((points-tcp[:, None, :3, 3]).norm(dim=-1) > .07)
        errors = (points-teacher_points).norm(dim=-1)
        losses = dict(route=F.huber_loss(prediction, target, delta=.02),
            joints=F.huber_loss(features['joint'].float(), batch['target'], delta=.02),
            maps=map_loss(features['dense'], teacher, observed),
            geometry=geometry_nll(auxiliary, teacher_cells, cells_valid),
            uncertainty=point_quantile_loss(radii, errors, point_valid))
        with torch.no_grad():
            positions, rotations, _ = cartesian_proposal(prediction.detach(), features['joint'].detach(),
                batch['qpos'][:, :7], tcp, goal_position(features['state']), model.kinematics)
        trust_losses = []
        for row in range(len(tcp)):
            memory = PersistentGeometry(model.memory.capacity, model.memory.merge_radius, model.memory.voxel_size)
            with torch.no_grad():
                for slot in torch.where(batch['history_mask'][row])[0].tolist():
                    past_tcp = model.kinematics(batch['history_qpos'][row, slot, :7])
                    fingers = batch['history_qpos'][row, slot, 7:9]
                    p = features['points'][row, slot].detach()
                    u = model.clearance.predict_error(model.route.visual(features['tokens'][row:row+1, slot].float()),
                        p[None], batch['history_T_B_C'][row:row+1, slot], past_tcp[None])[0]
                    valid = workspace_mask(p) & ~model.embodiment.self_mask(p, past_tcp, fingers)
                    step = int(batch['frame_index'][row]-batch['history_ages'][row, slot])
                    memory.update(p, valid, u, step, past_tcp[:3, 3], goal_position(features['state'])[row])
                surfaces, error = memory.query()
            candidates, costs = model.clearance.costs(positions[row], rotations[row], tcp[row],
                goal_position(features['state'])[row], surfaces, error,
                batch['qpos'][row, 7:9], model.embodiment)
            with torch.no_grad():
                regret = (candidates[:, 4::5]-tcp[row, :3, 3]-target[row]).square().mean((-1, -2))
                labels = (-regret/.015**2).softmax(-1)
            trust_losses.append(-(labels*(-costs/.03).log_softmax(-1)).sum())
        losses['trust'] = torch.stack(trust_losses).mean()
    return sum(weights[name]*value for name, value in losses.items()), losses


@torch.inference_mode()
def validation_rmse(model, maps, loader, device):
    model.eval()
    squared, count = 0., 0
    for raw in loader:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            _, _, prediction, target, _ = predict_route(model, maps, batch)
        squared += float((prediction-target).square().sum()); count += prediction.numel()
    if not count:
        raise ValueError('Empty validation data')
    return (squared/count)**.5


def train(config, output):
    options = config['train']
    device = require_device(options['device'])
    seed_everything(options['seed'])
    create_output(output)
    splits = make_splits(config['benchmark'])
    catalog = episode_catalog(Path(config['benchmark']['root']))
    root = Path(config['benchmark']['root'])
    training = NavigationDataset(root, splits['train'], catalog, options['frame_stride'], options['recovery_root'])
    validation = NavigationDataset(root, splits['validation'], catalog, options['validation_frame_stride'])
    model = NavigationPolicy(config['model']).to(device)
    maps = GeometryMaps(config['model']['maps']).to(device)
    generator = torch.Generator().manual_seed(options['seed'])
    sampler = WeightedRandomSampler(sampling_weights(training.route, training.source),
                                   options['draws_per_epoch'], replacement=True, generator=generator)
    train_loader = make_loader(training, options, True, options['seed'], sampler=sampler)
    val_loader = make_loader(validation, options, False, options['seed'])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=options['learning_rate'], weight_decay=options['weight_decay'])
    write_json(output/'config.json', config); write_json(output/'splits.json', splits)
    best, records = float('inf'), []
    try:
        for epoch in range(1, options['epochs']+1):
            model.train(); started = time.monotonic(); total = 0.; rows = 0
            for raw in train_loader:
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    loss, _ = training_loss(model, maps, batch, options['loss_weights'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, options['gradient_clip_norm'])
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
                optimizer.step()
                size = len(batch['qpos']); total += float(loss.detach())*size; rows += size
            score = validation_rmse(model, maps, val_loader, device)
            if score < best:
                best = score
                save_checkpoint(output/'best.pt', dict(format_version=1, architecture='gtsn_main',
                    config=config, splits=splits, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    selected_epoch=epoch, validation_rmse_m=score))
            record = dict(epoch=epoch, loss=total/rows, validation_rmse_m=score, seconds=time.monotonic()-started)
            records.append(record); write_json(output/'epochs.json', records); print(record, flush=True)
        write_json(output/'complete.json', dict(epochs=options['epochs'], best_validation_rmse_m=best,
            navigation_initialized_from_scratch=True, test_used_for_selection=False))
    finally:
        training.close(); validation.close()
