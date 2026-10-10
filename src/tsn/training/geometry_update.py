"""Independent frozen-parent C1/C2 adapter training on causal cached examples."""
import copy
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from tsn.common.config import create_output, read_json, write_json
from tsn.common.seed import seed_everything
from tsn.data.navigation import NavigationDataset, sampling_weights
from tsn.data.splits import episode_catalog, validate_splits
from tsn.features.state import policy_state
from tsn.models.c1_memory import PersistentGeometry, workspace_mask
from tsn.models.policy import load_policy, metric_points
from tsn.models.route import goal_position
from tsn.training.runner import predict_route


@torch.no_grad()
def cache_examples(checkpoint, output, draws=2048, seed=20261009, device='cuda'):
    """Cache only training examples; no held-out RGB/depth/future labels are read."""
    create_output(output)
    seed_everything(seed)
    model, maps, saved = load_policy(checkpoint, device)
    model.requires_grad_(False)
    options = saved['config']['train']
    root = Path(saved['config']['benchmark']['root'])
    catalog = episode_catalog(root)
    validate_splits(saved['splits'], catalog)
    data = NavigationDataset(root, saved['splits']['train'], catalog, options['frame_stride'],
        options['recovery_root'], options.get('observation_hw'), include_depth_history=True)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.multinomial(sampling_weights(data.route, data.source), draws, replacement=True,
                               generator=generator).tolist()
    examples = []
    loader = DataLoader(data, batch_size=4, sampler=indices, num_workers=2)
    from tsn.models.route import cartesian_proposal
    try:
        for raw in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                features, tcp, prediction, target, _ = predict_route(model, maps, batch)
            positions, rotations, _ = cartesian_proposal(prediction, features['joint'], batch['qpos'][:, :7],
                tcp, goal_position(features['state']), model.kinematics)
            observations = []
            for slot in range(batch['history_mask'].shape[1]):
                pose = batch['history_T_B_C'][:, slot]
                past_tcp = model.kinematics(batch['history_qpos'][:, slot, :7])
                points = features['points'][:, slot].float()
                error = model.clearance.predict_error(model.route.visual(features['tokens'][:, slot].float()),
                                                      points, pose, past_tcp)
                depth = batch['history_depth'][:, slot]
                teacher = maps(depth, batch['history_K'][:, slot], pose, batch['goal_pose'][:, :3],
                               batch['future_ee'], batch['valid_future'])
                teacher_points = metric_points(teacher)
                depth_small = F.interpolate(depth[:, None], (20, 20), mode='nearest-exact').flatten(1)
                depth_valid = torch.isfinite(depth_small) & (depth_small >= maps.settings.near_m) & (depth_small <= maps.settings.far_m)
                observations.append((points, error, teacher_points, depth_valid, past_tcp,
                                     torch.linalg.inv(past_tcp)@pose))
            for row in range(len(tcp)):
                history = []
                for slot in torch.where(batch['history_mask'][row])[0].tolist():
                    p, u, teacher, teacher_valid, past_tcp, mount = (value[row] for value in observations[slot])
                    fingers = batch['history_qpos'][row, slot, 7:9]
                    valid = workspace_mask(p) & ~model.embodiment.self_mask(p, past_tcp, fingers, mount)
                    valid &= (p-past_tcp[:3, 3]).norm(dim=-1) > .07
                    history.append(dict(points=p[valid].cpu(), radius=u[valid].cpu(),
                        reliability=((p-teacher).norm(dim=-1) <= .04)[valid].float().cpu(),
                        label_mask=teacher_valid[valid].cpu(), tcp=past_tcp.cpu(),
                        state=policy_state(batch['history_qpos'][row:row+1, slot],
                            batch['history_goal_pose'][row:row+1, slot], maps.settings)[0].cpu(),
                        step=int(batch['frame_index'][row]-batch['history_ages'][row, slot])))
                examples.append(dict(history=history, state=features['state'][row].float().cpu(),
                    tcp=tcp[row].cpu(), goal=goal_position(features['state'])[row].cpu(),
                    fingers=batch['qpos'][row, 7:9].cpu(), mount=(torch.linalg.inv(tcp[row])@batch['T_B_C'][row]).cpu(),
                    arm_points=model.kinematics.arm_points(batch['qpos'][row, :7]).cpu(),
                    positions=positions[row].cpu(), rotations=rotations[row].cpu(), target=target[row].cpu()))
            if len(examples) % 128 == 0:
                write_json(output/'progress.json', dict(examples=len(examples), total=draws))
                print(f'Cached {len(examples)}/{draws} training examples', flush=True)
    finally:
        data.close()
    torch.save(examples, output/'examples.pt')
    write_json(output/'manifest.json', dict(examples=len(examples), seed=seed, indices=indices,
        parent_checkpoint=str(checkpoint), partition='train', train_episodes=saved['splits']['train'],
        source_counts={str(i): int(sum(data.source[index] == i for index in indices)) for i in (0, 1)},
        route_counts={str(i): int(sum(data.route[index] == i for index in indices)) for i in (0, 1, 2)},
        source_mixture=[.65, .35], route_mixture=[.2, .4, .4], test_used_for_training=False))
    write_json(output/'complete.json', dict(examples=len(examples)))


def adapter_objective(model, example, component):
    """Shared deployment memory and clearance paths, with depth reliability labels."""
    memory = PersistentGeometry(model.memory.capacity, model.memory.merge_radius, model.memory.voxel_size)
    memory.selector, memory.strength = model.memory.selector, model.memory.strength
    reliability_losses = []
    for frame in example['history']:
        p, u = frame['points'], frame['radius']
        memory.update(p, torch.ones(len(p), dtype=torch.bool, device=p.device), u, frame['step'],
                      frame['tcp'][:3, 3], example['goal'], frame['state'])
        valid = frame['label_mask']
        if component == 'c1' and valid.any():
            logits = memory.scores(p, u)
            reliability_losses.append(F.binary_cross_entropy_with_logits(logits[valid], frame['reliability'][valid]))
    surfaces, error, point_weights = memory.query(return_weights=True)
    context = {key: example[key] for key in ('state', 'goal', 'arm_points')}
    candidates, costs = model.clearance.costs(example['positions'], example['rotations'], example['tcp'],
        example['goal'], surfaces, error, example['fingers'], model.embodiment, example['mount'], point_weights, context)
    regret = (candidates[:, 4::5]-example['tcp'][:3, 3]-example['target']).square().mean((-1, -2))
    labels = (-regret/.015**2).softmax(-1)
    navigation = -(labels*(-costs/.03).log_softmax(-1)).sum()
    reliability = torch.stack(reliability_losses).mean() if reliability_losses else navigation*0
    # The point-weight prior regularizes C1 when candidate labels are alike.
    # C2 uses the bounded residual strength and optimizer weight decay.
    penalty = (point_weights-1).square().mean() if len(point_weights) else navigation*0
    anchor = sum(parameter.square().sum()*0 for parameter in model.parameters() if parameter.requires_grad)
    return navigation+.1*reliability+.01*penalty+anchor, dict(navigation=navigation.detach(), reliability=reliability.detach())


def train_adapter(checkpoint, cache, output, component, epochs=4, seed=20261009, device='cuda'):
    """Train exactly one new module; all parent tensors stay bitwise unchanged."""
    from tsn.models.policy import NavigationPolicy
    from tsn.training.runner import initialize_geometry_update
    create_output(output)
    seed_everything(seed)
    saved = load_checkpoint(checkpoint)
    manifest = read_json(cache/'manifest.json')
    if (manifest['partition'] != 'train' or manifest['train_episodes'] != saved['splits']['train'] or
            not Path(manifest['parent_checkpoint']).samefile(checkpoint)):
        raise ValueError('Adapter cache must belong to this parent and its training partition')
    config = copy.deepcopy(saved['config'])
    config['model']['perception']['freeze_encoder'] = True
    config['model']['learned_geometry'] = dict(c1=component == 'c1', c2=component == 'c2',
                                             c1_strength=1., c2_strength=1.)
    config['train'].update(initialize_from=str(checkpoint), geometry_only=True)
    model = NavigationPolicy(config['model'], initialize_encoder=False).to(device).eval()
    provenance = initialize_geometry_update(model, checkpoint, config, saved['splits'])
    parent_keys = set(saved['model'])
    del saved['model']
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    raw = torch.load(cache/'examples.pt', map_location='cpu', weights_only=True)
    # Hold all compact geometry on GPU; the frozen image model is never called.
    def move(value):
        if isinstance(value, torch.Tensor):return value.to(device)
        if isinstance(value, dict):return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):return [move(item) for item in value]
        return value
    examples = move(raw)
    records = []
    for epoch in range(1, epochs+1):
        started, total = time.monotonic(), 0.
        for index in torch.randperm(len(examples)).tolist():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = adapter_objective(model, examples[index], component)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite adapter loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite adapter gradients')
            optimizer.step()
            total += float(loss.detach())
        record = dict(epoch=epoch, loss=total/len(examples), seconds=time.monotonic()-started)
        records.append(record)
        write_json(output/'epochs.json', records)
        print(record, flush=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    parent = load_checkpoint(checkpoint)['model']
    unchanged = all(torch.equal(state[key], parent[key]) for key in parent_keys)
    if not unchanged:raise RuntimeError('A frozen parent tensor changed')
    head = model.memory.selector.head[-1] if component == 'c1' else model.embodiment.risk_head[-1]
    if not float(head.weight.detach().abs().sum()):
        raise RuntimeError('The adapter output was not trained')
    save_checkpoint(output/'trained.pt', dict(format_version=1, architecture='gtsn_main', config=config,
        splits=saved['splits'], model=state, geometry_update=provenance, component=component,
        trained_epochs=epochs, test_used_for_selection=False))
    write_json(output/'initialization.json', {**provenance, 'component': component, 'seed': seed,
        'trainable_parameters': sum(p.numel() for p in parameters), 'parent_tensors_unchanged': unchanged,
        'cached_training_examples': len(examples), 'held_out_examples': 0})
    write_json(output/'complete.json', dict(epochs=epochs, parent_tensors_unchanged=unchanged))
