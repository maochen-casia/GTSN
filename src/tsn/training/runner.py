"""Fresh main-model training on expert routes and independent perturbations."""
import time
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler
from torch import distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel

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


class ShardedWeightedSampler(WeightedRandomSampler):
    """Draw one reproducible global sample stream, then partition it by rank."""
    def __init__(self, weights, draws, generator, rank=0, world=1):
        if draws % world:
            raise ValueError('Global draws must be divisible by the process count')
        super().__init__(weights, draws, replacement=True, generator=generator)
        self.rank, self.world = rank, world

    def __iter__(self):
        return iter(list(super().__iter__())[self.rank::self.world])

    def __len__(self):
        return self.num_samples//self.world


class TrainingObjective(nn.Module):
    """Put all differentiable training paths inside the distributed forward."""
    def __init__(self, model, maps, weights):
        super().__init__()
        self.policy, self.maps, self.weights = model, maps, weights

    def forward(self, batch):
        return training_loss(self.policy, self.maps, batch, self.weights)


def combined_totals(total, rows, components, device):
    values = torch.tensor([total, rows, *components.values()], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(values)
    values = values.tolist()
    return values[0], int(values[1]), dict(zip(components, values[2:]))


def encode_observations(model, maps, batch):
    """Past features are detached; the current RGB observation trains perception."""
    count = len(batch['qpos'])
    slots = batch['history_qpos'].shape[1]
    past = slots-1
    states = policy_state(batch['history_qpos'].flatten(0, 1),
                          batch['history_goal_pose'].flatten(0, 1), maps.settings).reshape(count, slots, 16)
    current = model.perception(batch['history_rgb'][:, -1], states[:, -1],
                               batch['K'], batch['T_B_C'])
    joint, tokens, geometry, dense = current
    previous_tokens = tokens.new_zeros(count*past, 16, 768)
    previous_geometry = geometry.new_zeros(count*past, 16, 6)
    previous_points = torch.zeros(count*past, 400, 3, device=tokens.device)
    valid = batch['history_mask'][:, :past].flatten()
    if valid.any():
        with torch.no_grad():
            _, old_tokens, old_geometry, old_dense = model.perception(
                batch['history_rgb'][:, :past].flatten(0, 1)[valid], states[:, :past].flatten(0, 1)[valid],
                batch['history_K'][:, :past].flatten(0, 1)[valid], batch['history_T_B_C'][:, :past].flatten(0, 1)[valid])
            previous_tokens[valid], previous_geometry[valid] = old_tokens, old_geometry
            previous_points[valid] = metric_points(old_dense)
    return dict(joint=joint, dense=dense, state=states[:, -1],
        tokens=torch.cat((previous_tokens.reshape(count, past, 16, 768), tokens[:, None]), 1),
        geometry=torch.cat((previous_geometry.reshape(count, past, 16, 6), geometry[:, None]), 1),
        points=torch.cat((previous_points.reshape(count, past, 400, 3), metric_points(dense)[:, None]), 1))


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
        radii = (model.clearance.predict_error(model.route.visual(features['tokens'][:, -1].float()).detach(),
                                              points, batch['T_B_C'], tcp)
                 if model.use_clearance else points.new_zeros(points.shape[:2]))
        teacher_points = metric_points(teacher)
        point_valid = F.interpolate(observed[:, None].float(), (20, 20), mode='nearest-exact').flatten(1).bool()
        point_valid &= workspace_mask(points) & ((points-tcp[:, None, :3, 3]).norm(dim=-1) > .07)
        errors = (points-teacher_points).norm(dim=-1)
        losses = dict(route=F.huber_loss(prediction, target, delta=.02),
            joints=F.huber_loss(features['joint'].float(), batch['target'], delta=.02),
            maps=map_loss(features['dense'], teacher, observed),
            geometry=geometry_nll(auxiliary, teacher_cells, cells_valid),
            uncertainty=point_quantile_loss(radii, errors, point_valid) if model.use_clearance else prediction.sum()*0)
        if not model.use_clearance:
            losses['trust'] = prediction.sum()*0
            return sum(weights[name]*value for name, value in losses.items()), losses
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
                    mount = torch.linalg.inv(past_tcp)@batch['history_T_B_C'][row, slot]
                    valid = workspace_mask(p) & ~model.embodiment.self_mask(p, past_tcp, fingers, mount)
                    step = int(batch['frame_index'][row]-batch['history_ages'][row, slot])
                    memory.update(p, valid, u, step, past_tcp[:3, 3], goal_position(features['state'])[row])
                surfaces, error = memory.query()
            candidates, costs = model.clearance.costs(positions[row], rotations[row], tcp[row],
                goal_position(features['state'])[row], surfaces, error,
                batch['qpos'][row, 7:9], model.embodiment, torch.linalg.inv(tcp[row])@batch['T_B_C'][row])
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
    if dist.is_initialized():
        values = torch.tensor([squared, count], dtype=torch.float64, device=device)
        dist.all_reduce(values)
        squared, count = values.tolist()
    return (squared/count)**.5


def train(config, output):
    options = config['train']
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = require_device(options['device'])
    if world > 1:
        if options['batch_size'] % world:
            raise ValueError('Global batch size must be divisible by the process count')
        if device.type == 'cuda':
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    seed_everything(options['seed'])
    if rank == 0:
        create_output(output)
    if world > 1:
        dist.barrier()
    splits = make_splits(config['benchmark'])
    catalog = episode_catalog(Path(config['benchmark']['root']))
    root = Path(config['benchmark']['root'])
    training = NavigationDataset(root, splits['train'], catalog, options['frame_stride'], options['recovery_root'],
                                 options.get('observation_hw'), config['model'].get('contributions', {}).get('c1', True))
    validation = NavigationDataset(root, splits['validation'], catalog, options['validation_frame_stride'],
                                   observation_hw=options.get('observation_hw'),
                                   use_history=config['model'].get('contributions', {}).get('c1', True))
    model = NavigationPolicy(config['model']).to(device)
    maps = GeometryMaps(config['model']['maps']).to(device)
    generator = torch.Generator().manual_seed(options['seed'])
    sampler = ShardedWeightedSampler(sampling_weights(training.route, training.source),
                                     options['draws_per_epoch'], generator, rank, world)
    loader_options = {**options, 'batch_size': options['batch_size']//world}
    train_loader = make_loader(training, loader_options, True, options['seed']+rank, sampler=sampler)
    val_sampler = list(range(rank, len(validation), world)) if world > 1 else None
    val_loader = make_loader(validation, loader_options, False, options['seed']+rank, sampler=val_sampler)
    objective = TrainingObjective(model, maps, options['loss_weights'])
    if world > 1:
        objective = DistributedDataParallel(objective, device_ids=[device.index] if device.type == 'cuda' else None)
        seed_everything(options['seed']+rank)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=options['learning_rate'], weight_decay=options['weight_decay'])
    if rank == 0:
        write_json(output/'config.json', config); write_json(output/'splits.json', splits)
    initialization = dict(navigation_initialized_from_scratch=True,
        pretrained_weights=config['model']['perception']['pretrained_weights'],
        pretrained_scope='perception.encoder', encoder_frozen=config['model']['perception']['freeze_encoder'],
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters),
        training_frames=len(training), validation_frames=len(validation),
        draws_per_epoch=options['draws_per_epoch'], test_used_for_selection=False,
        distributed_processes=world, global_batch_size=options['batch_size'],
        contributions=dict(c1=model.use_history, c2=model.use_embodiment, c3=model.use_clearance),
        history_slots=training.history.shape[1])
    if rank == 0:
        write_json(output/'initialization.json', initialization); print(initialization, flush=True)
    best, records = float('inf'), []
    try:
        for epoch in range(1, options['epochs']+1):
            model.train(); started = time.monotonic(); total = 0.; rows = 0
            components = {name: 0. for name in options['loss_weights']}
            for batch_index, raw in enumerate(train_loader, 1):
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    loss, losses = objective(batch)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, options['gradient_clip_norm'])
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
                optimizer.step()
                size = len(batch['qpos']); total += float(loss.detach())*size; rows += size
                for name, value in losses.items():components[name] += float(value.detach())*size
                if batch_index % 64 == 0 or batch_index == len(train_loader):
                    global_total, global_rows, global_components = combined_totals(total, rows, components, device)
                    progress = dict(epoch=epoch, batch=batch_index, batches=len(train_loader),
                        samples=global_rows, loss=global_total/global_rows, seconds=time.monotonic()-started,
                        loss_components={name: value/global_rows for name, value in global_components.items()})
                    if rank == 0:
                        write_json(output/'progress.json', progress); print(progress, flush=True)
            total, rows, components = combined_totals(total, rows, components, device)
            score = validation_rmse(model, maps, val_loader, device)
            if score < best:
                best = score
                if rank == 0:
                    save_checkpoint(output/'best.pt', dict(format_version=1, architecture='gtsn_main',
                        config=config, splits=splits, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        selected_epoch=epoch, validation_rmse_m=score))
            record = dict(epoch=epoch, loss=total/rows, validation_rmse_m=score,
                loss_components={name: value/rows for name, value in components.items()},
                seconds=time.monotonic()-started)
            records.append(record)
            if rank == 0:
                write_json(output/'epochs.json', records); print(record, flush=True)
            if world > 1:
                dist.barrier()
        if rank == 0:
            write_json(output/'complete.json', dict(epochs=options['epochs'], best_validation_rmse_m=best,
                navigation_initialized_from_scratch=True, test_used_for_selection=False))
    finally:
        training.close(); validation.close()
        if world > 1:
            dist.destroy_process_group()
