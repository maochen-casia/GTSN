"""Fresh main-model training on expert routes and independent perturbations."""
import time
import os
import copy
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler
from torch import distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import create_output, write_json
from tsn.common.seed import require_device, seed_everything
from tsn.data.loaders import make_loader
from tsn.data.navigation import NavigationDataset, sampling_weights
from tsn.data.splits import make_splits, episode_catalog
from tsn.features.maps import GeometryMaps
from tsn.features.state import policy_state
from tsn.models.c1_memory import PersistentGeometry, workspace_mask
from tsn.models.c2_embodiment import EmbodimentGeometry
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


def validate_fresh_joint(config):
    """Make the official-Pi3-only training contract explicit and enforceable."""
    options = config['train']
    if not options.get('fresh_joint', False):
        return
    model = config['model']
    forbidden = ('initialize_from', 'module_initialization', 'geometry_only')
    if any(options.get(key) for key in forbidden):
        raise ValueError('Fresh joint training cannot load experiment or warmup weights')
    if model['perception']['freeze_encoder'] or not model['perception']['pretrained_weights']:
        raise ValueError('Fresh joint training requires trainable official Pi3 weights')
    geometry = model.get('learned_geometry', {})
    if (geometry.get('mode') != 'replacement' or not geometry.get('c1') or not geometry.get('c2')
            or not all(model.get('contributions', {}).get(key, False) for key in ('c1', 'c2', 'c3'))):
        raise ValueError('Fresh joint training requires the complete learned C1/C2/C3 policy')
    if geometry.get('robot_representation') == 'surface':
        if options['loss_weights'].get('nodes', 0) <= 0 or options['loss_weights'].get('embodiment', 0) <= 0:
            raise ValueError('Surface-node training needs positive selection and embodiment supervision')
        if geometry.get('query_capacity', model['memory']['capacity']) > geometry.get('scene_attention_capacity', 1024):
            raise ValueError('Surface attention must accommodate the complete queried scene')


def parameter_hashes(model):
    """Record initialization without saving a second, usable model checkpoint."""
    return {name: hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, parameter in model.named_parameters()}


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
    point_count = model.point_grid_hw[0]*model.point_grid_hw[1]
    previous_points = torch.zeros(count*past, point_count, 3, device=tokens.device)
    valid = batch['history_mask'][:, :past].flatten()
    if valid.any():
        with torch.no_grad():
            _, old_tokens, old_geometry, old_dense = model.perception(
                batch['history_rgb'][:, :past].flatten(0, 1)[valid], states[:, :past].flatten(0, 1)[valid],
                batch['history_K'][:, :past].flatten(0, 1)[valid], batch['history_T_B_C'][:, :past].flatten(0, 1)[valid])
            previous_tokens[valid], previous_geometry[valid] = old_tokens, old_geometry
            previous_points[valid] = metric_points(old_dense, model.point_grid_hw)
    return dict(joint=joint, dense=dense, state=states[:, -1],
        tokens=torch.cat((previous_tokens.reshape(count, past, 16, 768), tokens[:, None]), 1),
        geometry=torch.cat((previous_geometry.reshape(count, past, 16, 6), geometry[:, None]), 1),
        points=torch.cat((previous_points.reshape(count, past, point_count, 3), metric_points(dense, model.point_grid_hw)[:, None]), 1))


def predict_route(model, maps, batch):
    features = encode_observations(model, maps, batch)
    with torch.autocast(device_type=batch['qpos'].device.type, enabled=False):
        tcp = model.kinematics(batch['qpos'][:, :7])
        target = model.kinematics(batch['qpos'][:, None, :7]+batch['target'][:, 4::5])[..., :3, 3]-tcp[:, None, :3, 3]
    prediction, auxiliary = model.route(features['tokens'], features['geometry'], batch['history_T_B_C'],
        batch['history_ages'], batch['history_mask'], features['state'], tcp)
    return features, tcp, prediction, target, auxiliary


def retention_targets(points, reliable, future_positions):
    """Depth correctness and proximity to the expert's future occupied path.

    These are training labels, not a deployment priority formula. Reliable
    background remains useful; surfaces near the future path receive more
    supervision so state/goal conditioning has a task-dependent target.
    """
    distance = torch.cdist(points.float(), future_positions.float()).amin(-1)
    return reliable.float()*(.25+.75*torch.exp(-.5*(distance/.12).square()))


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
        teacher_points = metric_points(teacher, model.point_grid_hw)
        point_valid = F.interpolate(observed[:, None].float(), model.point_grid_hw, mode='nearest-exact').flatten(1).bool()
        point_valid &= workspace_mask(points) & ((points-tcp[:, None, :3, 3]).norm(dim=-1) > .07)
        errors = (points-teacher_points).norm(dim=-1)
        losses = dict(route=F.huber_loss(prediction, target, delta=.02),
            joints=F.huber_loss(features['joint'].float(), batch['target'], delta=.02),
            maps=map_loss(features['dense'], teacher, observed),
            geometry=geometry_nll(auxiliary, teacher_cells, cells_valid),
            uncertainty=point_quantile_loss(radii, errors, point_valid) if model.use_clearance else prediction.sum()*0)
        if 'memory' in weights:
            if model.learned_c1:
                selector = model.memory.selector
                point_losses = []
                for row in range(len(tcp)):
                    valid = point_valid[row]
                    if valid.any():
                        p, u = points[row, valid], radii[row, valid].detach()
                        logits = selector(p, u, p.new_ones(len(p)), p.new_zeros(len(p)),
                            p.new_zeros(len(p)), tcp[row, :3, 3], goal_position(features['state'])[row],
                            features['state'][row])
                        reliability = (errors[row, valid].detach() <= .04).float()
                        if model.memory.replacement:
                            reliability = retention_targets(p, reliability, tcp[row, :3, 3]+target[row])
                        point_losses.append(F.binary_cross_entropy_with_logits(logits, reliability))
                losses['memory'] = (torch.stack(point_losses).mean() if point_losses
                                    else sum(p.sum()*0 for p in selector.parameters()))
            else:
                losses['memory'] = prediction.sum()*0
        if not model.use_clearance:
            losses['trust'] = prediction.sum()*0
            if 'embodiment' in weights:
                losses['embodiment'] = prediction.sum()*0
            return sum(weights[name]*value for name, value in losses.items()), losses
        with torch.no_grad():
            positions, rotations, joint_seeds = cartesian_proposal(prediction.detach(), features['joint'].detach(),
                batch['qpos'][:, :7], tcp, goal_position(features['state']), model.kinematics)
        trust_losses, body_losses, node_losses = [], [], []
        for row in range(len(tcp)):
            memory = PersistentGeometry(model.memory.capacity, model.memory.merge_radius, model.memory.voxel_size)
            memory.selector = model.memory.selector
            memory.strength = model.memory.strength
            memory.replacement = model.memory.replacement
            memory.query_source, memory.query_capacity = model.memory.query_source, model.memory.query_capacity
            with torch.no_grad():
                for slot in torch.where(batch['history_mask'][row])[0].tolist():
                    past_tcp = model.kinematics(batch['history_qpos'][row, slot, :7])
                    fingers = batch['history_qpos'][row, slot, 7:9]
                    p = features['points'][row, slot].detach()
                    u = model.clearance.predict_error(model.route.visual(features['tokens'][row:row+1, slot].float()),
                        p[None], batch['history_T_B_C'][row:row+1, slot], past_tcp[None])[0]
                    mount = torch.linalg.inv(past_tcp)@batch['history_T_B_C'][row, slot]
                    valid = workspace_mask(p) & ~model.self_surface_mask(p, past_tcp, fingers, mount, batch['history_qpos'][row, slot])
                    step = int(batch['frame_index'][row]-batch['history_ages'][row, slot])
                    memory.update(p, valid, u, step, past_tcp[:3, 3], goal_position(features['state'])[row],
                                  policy_state(batch['history_qpos'][row:row+1, slot],
                                               batch['history_goal_pose'][row:row+1, slot], maps.settings)[0])
            surfaces, error, point_weights = memory.query(return_weights=True)
            context = model.geometry_context(batch['qpos'][row], features['state'][row],
                                             goal_position(features['state'])[row], joint_seeds[row])
            candidates, costs = model.clearance.costs(positions[row], rotations[row], tcp[row],
                goal_position(features['state'])[row], surfaces, error,
                batch['qpos'][row, 7:9], model.embodiment, torch.linalg.inv(tcp[row])@batch['T_B_C'][row],
                point_weights, context)
            with torch.no_grad():
                regret = (candidates[:, 4::5]-tcp[row, :3, 3]-target[row]).square().mean((-1, -2))
                labels = (-regret/.015**2).softmax(-1)
            trust_losses.append(-(labels*(-costs/.03).log_softmax(-1)).sum())
            if model.surface_robot:
                surface_aux = model.embodiment.pop_training_aux()
                body_losses.append(F.mse_loss(surface_aux['risk'], surface_aux['teacher'])+model.embodiment.calibration_loss())
                node_losses.append(surface_aux['selection_loss'])
                continue
            if 'embodiment' in weights and model.learned_c2:
                # Training-only calibration target; deployment uses solely the
                # network output. This stabilizes risk scale alongside ranking.
                padding = model.clearance.padding(error).detach()
                mount = torch.linalg.inv(tcp[row])@batch['T_B_C'][row]
                with torch.no_grad():
                    teacher_risk = EmbodimentGeometry.contact_risk(model.embodiment, candidates, rotations[row], tcp[row],
                        surfaces, padding, batch['qpos'][row, 7:9], model.clearance.margin, mount)
                neural_risk = model.embodiment.contact_risk(candidates, rotations[row], tcp[row], surfaces, padding,
                    batch['qpos'][row, 7:9], model.clearance.margin, mount, context=context)
                body_losses.append(F.mse_loss(neural_risk, teacher_risk))
                if getattr(model.embodiment, 'calibrated', False):
                    body_losses[-1] += model.embodiment.calibration_loss()
        losses['trust'] = torch.stack(trust_losses).mean()
        if 'embodiment' in weights:
            losses['embodiment'] = torch.stack(body_losses).mean() if body_losses else prediction.sum()*0
            if model.learned_c2:
                # Keep a valid graph for fully filtered clouds, including DDP.
                losses['embodiment'] += sum(p.sum()*0 for p in model.embodiment.parameters())
        if 'nodes' in weights:
            losses['nodes'] = torch.stack(node_losses).mean() if node_losses else prediction.sum()*0
    return sum(weights[name]*value for name, value in losses.items()), losses


def initialize_geometry_update(model, checkpoint, config, splits):
    """Load parent heads exactly; add new geometry tensors and freeze its encoder.

    All other parameters retain their trainability unless the earlier adapter
    experiment explicitly requests geometry_only.
    """
    saved = load_checkpoint(checkpoint)
    if saved.get('architecture') != 'gtsn_main' or saved['splits'] != splits:
        raise ValueError('Geometry updates require a main-policy checkpoint and identical splits')
    previous, requested = copy.deepcopy(saved['config']['model']), copy.deepcopy(config['model'])
    for settings in (previous, requested):
        settings.pop('learned_geometry', None)
        settings['perception'].pop('freeze_encoder', None)
    if previous != requested or not config['model']['perception']['freeze_encoder']:
        raise ValueError('Freeze the parent encoder and preserve the parent policy settings')
    missing, unexpected = model.load_state_dict(saved['model'], strict=False)
    allowed = ('memory.selector.', 'embodiment.point_embedding.', 'embodiment.scene_embedding.',
               'embodiment.region_embedding.', 'embodiment.context_embedding.',
               'embodiment.self_attention.', 'embodiment.scene_attention.', 'embodiment.risk_head.',
               'embodiment.palm_probes', 'embodiment.wrist_probes', 'embodiment.body_embedding.',
               'embodiment.scene_encoder.', 'embodiment.region_encoder.', 'embodiment.state_encoder.',
               'embodiment.blocks.', 'embodiment.pool.', 'embodiment.readout', 'embodiment.priority_head.')
    if unexpected or any(not name.startswith(allowed) for name in missing):
        raise ValueError(f'Parent checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    if config['train'].get('geometry_only', False):
        model.requires_grad_(False)
        if model.memory.selector is not None:
            model.memory.selector.requires_grad_(True)
        if model.learned_c2:
            for name, parameter in model.embodiment.named_parameters():
                parameter.requires_grad_(True)
    digest = hashlib.sha256()
    with Path(checkpoint).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return dict(parent_checkpoint=str(checkpoint), parent_sha256=digest.hexdigest(),
                geometry_only=config['train'].get('geometry_only', False), new_tensors=missing)


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
    validate_fresh_joint(config)
    if options.get('geometry_only', False):
        raise ValueError('Use scripts/geometry_update.py for geometry-only training; '
                         'unchanged route RMSE cannot select a geometry adapter')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = require_device(options['device'])
    if world > 1:
        if options['batch_size'] % world:
            raise ValueError('Global batch size must be divisible by the process count')
        if device.type == 'cuda':
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    if device.type == 'cuda' and options.get('gpu_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(options['gpu_memory_fraction'], device=device)
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
    parent = options.get('initialize_from')
    model = NavigationPolicy(config['model'], initialize_encoder=not bool(parent)).to(device)
    provenance = initialize_geometry_update(model, parent, config, splits) if parent else None
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
    if options.get('module_initialization'):
        if provenance is None:
            raise ValueError('Module warmup initialization requires a parent checkpoint')
        warm = torch.load(options['module_initialization'], map_location='cpu', weights_only=True)
        new_names = set(provenance['new_tensors'])
        if set(warm)-new_names:
            raise ValueError('Module warmup may initialize only new tensors')
        model.load_state_dict(warm, strict=False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if options.get('fresh_joint') and len(parameters) != len(list(model.parameters())):
        raise ValueError('Every parameter must be trainable in fresh joint training')
    new_names = set(provenance['new_tensors']) if provenance else set()
    old_parameters = [p for name, p in model.named_parameters() if p.requires_grad and name not in new_names]
    new_parameters = [p for name, p in model.named_parameters() if p.requires_grad and name in new_names]
    optimizer = torch.optim.AdamW([{'params': old_parameters, 'lr': options['learning_rate']},
        {'params': new_parameters, 'lr': options.get('module_learning_rate', options['learning_rate'])}],
        weight_decay=options['weight_decay'])
    if rank == 0:
        write_json(output/'config.json', config); write_json(output/'splits.json', splits)
    initialization = dict(navigation_initialized_from_scratch=not bool(parent), geometry_update=provenance,
        pretrained_weights=config['model']['perception']['pretrained_weights'],
        pretrained_scope='perception.encoder', encoder_frozen=config['model']['perception']['freeze_encoder'],
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters),
        frozen_parameter_names=[name for name, p in model.named_parameters() if not p.requires_grad],
        training_frames=len(training), validation_frames=len(validation),
        draws_per_epoch=options['draws_per_epoch'], test_used_for_selection=False,
        distributed_processes=world, global_batch_size=options['batch_size'],
        contributions=dict(c1=model.use_history, c2=model.use_embodiment, c3=model.use_clearance),
        history_slots=training.history.shape[1])
    initialization.update(scene_points_per_observation=model.point_grid_hw[0]*model.point_grid_hw[1],
        scene_query_capacity=model.memory.query_capacity, robot_representation='surface' if model.surface_robot else 'regions',
        robot_nodes_per_pose=model.embodiment.node_count if model.surface_robot else
                             13 if model.learned_c2 and model.memory.replacement else None,
        robot_surface_pool=len(model.embodiment.surface.local_points) if model.surface_robot else None)
    if rank == 0:
        write_json(output/'initialization.json', initialization)
        if options.get('fresh_joint'):
            write_json(output/'initial_parameter_sha256.json', parameter_hashes(model))
        print({k: v for k, v in initialization.items() if k != 'frozen_parameter_names'}, flush=True)
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
                if batch_index == 1 and rank == 0:
                    groups = ('perception.encoder', 'perception.decoder', 'perception.joint_head',
                              'perception.point_head', 'route', 'clearance.error_head', 'clearance.trust_head',
                              'memory.selector', 'embodiment')
                    if model.surface_robot:
                        groups += ('embodiment.selection_head', 'embodiment.surface_embedding', 'embodiment.blocks')
                    gradients = {prefix: dict(
                        trainable_parameters=sum(p.numel() for name, p in model.named_parameters()
                                                 if name.startswith(prefix+'.') and p.requires_grad),
                        gradient_l1=sum(float(p.grad.detach().abs().sum()) for name, p in model.named_parameters()
                                        if name.startswith(prefix+'.') and p.grad is not None)) for prefix in groups}
                    write_json(output/f'gradient_audits/epoch_{epoch:03d}.json', gradients)
                    if epoch == 1:
                        write_json(output/'gradient_audit.json', gradients)
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
                navigation_initialized_from_scratch=not bool(parent), test_used_for_selection=False))
    finally:
        training.close(); validation.close()
        if world > 1:
            dist.destroy_process_group()
