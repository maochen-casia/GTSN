"""Collect causal on-policy histories with simulator-verified training targets.

Only train episodes are admitted. Teacher trials use the real controller's
wrist/IK and restore the live physics state; no teacher is used at evaluation.
"""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from tsn.common.config import create_output, read_json, write_json
from tsn.common.checkpoint import load_checkpoint
from tsn.data.splits import ROUTES, episode_catalog, validate_splits
from tsn.features.state import policy_state
from tsn.models.compact_policy import goal_xyz, load_compact_policy
from tsn.models.current_view_policy import CurrentViewHead, metric_points
from tsn.simulation.episode import EpisodeSimulation


def training_subset(splits, catalog, count, shard, shards, seed):
    if count < 5 or count % 5 or not 0 <= shard < shards:
        raise ValueError('Use a multiple of five episodes and a valid shard index')
    rng = np.random.default_rng(seed)
    chosen = []
    for route, fraction in zip(ROUTES, (.2, .4, .4)):
        ids = sorted(ep for ep in splits['train'] if catalog[ep] == route)
        size = int(round(count*fraction))
        if size > len(ids):
            raise ValueError('Requested more training scenes than exist')
        selected = sorted(rng.choice(ids, size=size, replace=False).tolist())
        chosen.extend(selected[shard::shards])
    return sorted(chosen)


def verify_targets(simulation, candidates):
    """Choose the first contact-free 30-step candidate and restore all trials.

    Save velocities and drive targets as well as joint poses. Labels are not
    accepted on clipping, nonfinite targets, or any forbidden physics contact.
    """
    if candidates.ndim != 3 or candidates.shape[1:] != (30, 7):
        raise ValueError('Teacher candidates must contain exactly 30 seven-joint targets')
    physics = simulation.scene.physx_system
    packed = physics.pack()
    drives = [(j.get_drive_target(), j.get_drive_velocity_target()) for j in simulation.joints]
    before = simulation.snapshot()['qpos'].copy()
    velocity = simulation.robot.get_qvel().copy()
    def restore():
        physics.unpack(packed)
        for joint, (target, speed) in zip(simulation.joints, drives):
            joint.set_drive_target(target)
            joint.set_drive_velocity_target(speed)
    chosen, tried = None, 0
    try:
        for index, targets in enumerate(candidates):
            restore()
            if not np.isfinite(targets).all():
                continue
            tried += 1
            safe = True
            for target in targets:
                _, contacts, clipped = simulation.step(target)
                if contacts or clipped:
                    safe = False
                    break
            if safe:
                chosen = index
                break
    finally:
        restore()
    np.testing.assert_allclose(simulation.snapshot()['qpos'], before, rtol=0, atol=1e-7)
    np.testing.assert_allclose(simulation.robot.get_qvel(), velocity, rtol=0, atol=1e-7)
    return chosen, tried


def corrective_candidates(route, goal_direction):
    """Small representable offsets, ordered by size; training teacher only."""
    direction = goal_direction.clone()
    direction[2] = 0
    direction = direction/direction.norm().clamp_min(.01)
    side = torch.stack((-direction[1], direction[0], direction.new_zeros(())))
    up = direction.new_tensor([0., 0., 1.])
    directions = torch.stack((up, side, -side, -direction, direction,
                               (up+side)/2**.5, (up-side)/2**.5, -up))
    offsets = torch.cat([directions*size for size in (.015, .025, .035)])
    ramp = route.new_tensor([.5, 1., 1., 1., 1., 1.])
    return route[None] + offsets[:, None]*ramp[None, :, None]


@torch.inference_mode()
def corrective_label(policy, simulation, route, action, state, pose, tcp, base):
    """Preserve a verified proposal or find the first safe bounded correction."""
    q = state[:, :7].float()*torch.pi
    selected, attempts = verify_targets(simulation, (q[:, None]+action.float()).cpu().numpy())
    if selected is not None:
        return route, True, False, attempts
    goal = goal_xyz(state)
    # The terminal servo ignores route knots, so no learned route correction
    # can change this trial. Keep its history but do not invent a target.
    if float((goal[0]-tcp[0, :3, 3]).norm()) < policy.servo_radius:
        return route, False, True, attempts
    routes = corrective_candidates(route, goal[0]-tcp[0, :3, 3])
    count = len(routes)
    actions = policy.route_actions(routes, base.expand(count, -1, -1), q.expand(count, -1),
        tcp.expand(count, -1, -1), goal.expand(count, -1), pose.expand(count, -1, -1))
    selected, extra = verify_targets(simulation, (q[:, None]+actions).cpu().numpy())
    return (route if selected is None else routes[selected]), selected is not None, True, attempts+extra


@torch.inference_mode()
def collect(checkpoint_path, output, device, count=200, shard=0, shards=1, seed=20261005, initial_head=None):
    create_output(output)
    policy, maps, checkpoint = load_compact_policy(checkpoint_path, device)
    if initial_head is not None:
        parent = load_checkpoint(initial_head)
        if parent['splits'] != checkpoint['splits']:
            raise ValueError('Collector and parent splits differ')
        policy.head = CurrentViewHead.from_checkpoint(parent).to(device).eval()
    if not getattr(policy.head, 'use_points', False):
        raise ValueError('Collector requires current-view geometry to cache dense RGB predictions')
    root = Path(checkpoint['config']['benchmark']['root'])
    catalog = episode_catalog(root)
    validate_splits(checkpoint['splits'], catalog)
    ids = training_subset(checkpoint['splits'], catalog, count, shard, shards, seed)
    lookup = {ep: i for i, ep in enumerate(checkpoint['splits']['train'])}
    options = dict(checkpoint['config']['eval'], render_videos=False, stop_on_collision=True)
    write_json(output/'protocol.json', dict(checkpoint=str(checkpoint_path),
        initial_head=str(initial_head) if initial_head else None, partition='train',
        episodes=ids, requested_episodes=count, shard=shard, shards=shards, seed=seed,
        controller='compact', clearance='disabled', teacher_mode='corrective',
        teacher='30-step physics verification with deployed wrist/IK',
        correction_bound_m=.035,
        invalid_label='retain observation in history; mask action loss', test_used=False))
    capture = {}
    hook = policy.backbone.register_forward_hook(lambda module, inputs, result: capture.update(result=result))
    route_actions = policy.route_actions
    def capture_route(route, *args):
        capture['route'] = route.detach().clone()
        return route_actions(route, *args)
    policy.route_actions = capture_route
    records, sequence_indices, episodes = [], [], []
    started = time.monotonic()
    try:
        for ep in ids:
            with h5py.File(root/ep/'episode.h5', 'r') as data:
                reference_q = np.asarray(data['qpos'][:], dtype=np.float32)
                first_ee = np.asarray(data['ee_pose'][0], dtype=np.float32)
                calibration, extrinsic = data['intrinsics'][:], data['T_ee_camera_cv'][:]
                goal = np.asarray(data['goal_pose_xyz_wxyz'][:], dtype=np.float32)
                names = tuple(x.decode() if isinstance(x, bytes) else str(x) for x in data['joint_names'][:])
                shape = data['depth_m'].shape[1:]
            policy.reset_episode()
            K = torch.as_tensor(calibration, device=device, dtype=torch.float32)[None]
            goal_tensor = torch.as_tensor(goal, device=device)[None]
            indices, labels, trials, unsafe_count, corrected_count = [], 0, 0, 0, 0
            termination = 'timeout'
            with EpisodeSimulation(read_json(root/ep/'scene.json'), calibration, extrinsic, shape,
                                   reference_q[0], first_ee, names, options) as simulation:
                measured = simulation.snapshot()
                for step in range(0, options['max_control_steps'], 15):
                    if np.linalg.norm(measured['T_B_E'][:3, 3]-goal[:3]) <= options['goal_position_tolerance_m']:
                        termination = 'success'
                        break
                    rgb, depth = simulation.render()
                    state = policy_state(torch.as_tensor(measured['qpos'], device=device)[None], goal_tensor, maps.settings)
                    pose = torch.as_tensor(measured['T_B_C'], device=device, dtype=torch.float32)[None]
                    tcp = policy.kinematics(state[:, :7]*torch.pi)
                    policy.observe_step(step)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                        action = policy(torch.from_numpy(rgb).to(device)[None], state, K, pose)
                        base, tokens, geometry, dense = capture['result']
                    waypoint, supervised, unsafe, attempts = corrective_label(policy, simulation,
                        capture['route'][0], action, state, pose, tcp, base)
                    unsafe_count += int(unsafe)
                    corrected_count += int(unsafe and supervised)
                    trials += attempts
                    depth_tensor = torch.from_numpy(depth).to(device)[None]
                    teacher_map = maps(depth_tensor, K, pose, torch.zeros(1, 3, device=device),
                        torch.zeros(1, 1, 3, device=device), torch.zeros(1, 1, dtype=torch.bool, device=device))
                    small_depth = F.interpolate(depth_tensor[:, None],
                        (maps.settings.height, maps.settings.width), mode='nearest-exact')
                    valid = torch.isfinite(small_depth) & (small_depth >= maps.settings.near_m) & (small_depth <= maps.settings.far_m)
                    values = dict(tokens=tokens.half(), geometry=geometry.float(), state=state, pose=pose, tcp=tcp,
                        points=metric_points(dense), point_teacher=metric_points(teacher_map),
                        point_valid=F.interpolate(valid.float(), (20, 20), mode='nearest-exact').flatten(1).bool(),
                        teacher=F.adaptive_avg_pool2d(teacher_map[:, :3], (4, 4)).flatten(2).transpose(1, 2),
                        geometry_valid=F.adaptive_avg_pool2d(valid.float(), (4, 4)).flatten(1) >= .999)
                    record = {k: v[0].float().cpu().numpy() if v.dtype == torch.bfloat16 else v[0].cpu().numpy() for k, v in values.items()}
                    record.update(waypoint=waypoint.float().cpu().numpy(), route=np.int64(ROUTES.index(catalog[ep])),
                        episode=np.int64(lookup[ep]), frame=np.int64(step), source=np.int64(2), supervision_valid=np.bool_(supervised),
                        policy_unsafe=np.bool_(unsafe), action_weight=np.float32(8. if unsafe else 1.))
                    indices.append(len(records))
                    records.append(record)
                    labels += int(supervised)
                    anchor = measured['qpos'][:7].copy()
                    contacts = []
                    for target in anchor+action[0, :min(15, options['max_control_steps']-step)].float().cpu().numpy():
                        measured, contacts, _ = simulation.step(target)
                        if contacts:
                            termination = 'collision'
                            break
                        if np.linalg.norm(measured['T_B_E'][:3, 3]-goal[:3]) <= options['goal_position_tolerance_m']:
                            termination = 'success'
                            break
                    if termination != 'timeout':
                        break
            if labels:
                sequence_indices.append(np.array(indices))
            row = dict(episode=ep, route=catalog[ep], observations=len(indices), verified_labels=labels,
                       teacher_trials=trials, termination=termination, unsafe_proposals=unsafe_count,
                       corrected_proposals=corrected_count)
            episodes.append(row)
            # Save each episode immediately so collection progress is auditable.
            directory = output/'episodes'
            directory.mkdir(exist_ok=True)
            if indices:
                np.savez_compressed(directory/f'{ep}.npz', **{k: np.stack([records[i][k] for i in indices]) for k in records[indices[0]]})
            write_json(output/'status.json', dict(completed=len(episodes), total=len(ids), samples=len(records),
                verified_labels=sum(r['verified_labels'] for r in episodes), seconds=time.monotonic()-started, last=row))
            print('collect', json.dumps(row), flush=True)
    finally:
        hook.remove()
        policy.route_actions = route_actions
    if not sequence_indices:
        raise RuntimeError('No physics-verified recovery labels collected')
    for key in records[0]:
        np.save(output/f'{key}.npy', np.stack([r[key] for r in records]))
    np.save(output/'sequence_offsets.npy', np.cumsum([0]+[len(s) for s in sequence_indices]))
    np.save(output/'sequence_indices.npy', np.concatenate(sequence_indices))
    write_json(output/'metadata.json', dict(samples=len(records), sequences=len(sequence_indices),
        episode_ids=checkpoint['splits']['train'], collected_episode_ids=ids, partition='train',
        frozen_checkpoint=str(checkpoint_path), observation_period=15, teacher_verified_steps=30,
        teacher_mode='corrective'))
    write_json(output/'summary.json', dict(episodes=episodes, seconds=time.monotonic()-started))
    write_json(output/'complete.json', dict(episodes=len(ids), samples=len(records),
        verified_labels=sum(r['verified_labels'] for r in episodes),
        unsafe_proposals=sum(r['unsafe_proposals'] for r in episodes),
        corrected_proposals=sum(r['corrected_proposals'] for r in episodes), test_used=False))
