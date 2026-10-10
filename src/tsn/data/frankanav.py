"""Front-only FrankaNav conversion with explicit robot and timestamp provenance.

The real Robotiq gripper has one independent joint. This pilot uses a separate
schema instead of silently pretending it has the benchmark's two Panda fingers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import re

import cv2
import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F

SCHEMA = 'frankanav-front-1.0'


class MetadataUnpickler(pickle.Unpickler):
    """Admit plain containers and NumPy arrays, without arbitrary pickle globals."""
    def find_class(self, module, name):
        allowed = {
            ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype,
            ('numpy.core.numeric', '_frombuffer'): np.core.numeric._frombuffer,
            ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
        }
        key = (module.replace('numpy._core.', 'numpy.core.'), name)
        if key not in allowed:
            raise ValueError(f'Unsupported metadata pickle global: {module}.{name}')
        return allowed[key]


def read_metadata(path):
    with Path(path).open('rb') as stream:
        rows = MetadataUnpickler(stream).load()
    if not isinstance(rows, list) or len(rows) < 2 or not all(isinstance(row, dict) for row in rows):
        raise ValueError('Metadata must contain at least two per-frame dictionaries')
    ids = np.asarray([row['frame_id'] for row in rows], dtype=np.int64)
    if not np.all(np.diff(ids) > 0):
        raise ValueError('Frame IDs must be unique and strictly increasing')
    return rows


def timestamp_ns(stamp):
    sec, nano = int(stamp['sec']), int(stamp['nanosec'])
    if sec <= 0 or not 0 <= nano < 1_000_000_000:
        raise ValueError('Missing or invalid acquisition timestamp')
    return sec * 1_000_000_000 + nano


def payload_path(root, episode, value, view, suffix):
    """Only read the selected episode/view, including when metadata is malformed."""
    relative = Path(value)
    if (relative.is_absolute() or len(relative.parts) != 3 or relative.parts[:2] != (episode, view)
            or not re.fullmatch(r'\d{4,}' + re.escape(suffix), relative.name)):
        raise ValueError(f'Unexpected {view} payload path: {value}')
    path = (Path(root) / relative).resolve()
    if path.parent != (Path(root) / episode / view).resolve() or not path.is_file():
        raise FileNotFoundError(path)
    return path


def interpolate_joints(times, values, queries):
    """Interpolate measured joint states on their own clock; collapse repeated stamps."""
    times, values, queries = np.asarray(times), np.asarray(values), np.asarray(queries)
    if (times.ndim != 1 or values.shape != (len(times), 7) or not np.isfinite(values).all()
            or not np.isfinite(times).all() or not np.isfinite(queries).all()
            or np.any(np.diff(times) < 0)):
        raise ValueError('Joint measurements require finite values and ordered timestamps')
    unique, reverse = np.unique(times[::-1], return_index=True)
    # A repeated timestamp denotes a repeated latest message; keep its last snapshot.
    last = len(times) - 1 - reverse
    if len(unique) < 2:
        raise ValueError('At least two distinct joint timestamps are required')
    return np.stack([np.interp(queries, unique, values[last, joint]) for joint in range(7)], -1)


def causal_indices(times, queries):
    times, queries = np.asarray(times), np.asarray(queries)
    if not len(times) or np.any(np.diff(times) <= 0) or np.any(queries < times[0]):
        raise ValueError('Causal observations require increasing times and no pre-episode query')
    return np.searchsorted(times, queries, side='right') - 1


def nearest_indices(times, queries):
    """Return original indices; depth messages can be repeated or arrive out of order."""
    times, queries = np.asarray(times), np.asarray(queries)
    if not len(times):
        raise ValueError('No depth timestamps')
    order = np.argsort(times, kind='stable')
    ordered = times[order]
    right = np.searchsorted(ordered, queries).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    take_right = np.abs(ordered[right] - queries) < np.abs(ordered[left] - queries)
    return order[np.where(take_right, right, left)]


def link7_fk(q, calibration):
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 7 or not np.isfinite(q).all():
        raise ValueError('Panda FK requires seven finite joint angles')
    result = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for index, joint in enumerate(calibration['joint_origins']):
        if not np.allclose(joint['axis'], [0, 0, 1]):
            raise ValueError('Unsupported Panda joint axis')
        origin = np.eye(4)
        origin[:3, :3] = Rotation.from_euler('xyz', joint['rpy']).as_matrix()
        origin[:3, 3] = joint['xyz']
        rotation = np.broadcast_to(np.eye(4), result.shape).copy()
        c, s = np.cos(q[..., index]), np.sin(q[..., index])
        rotation[..., 0, 0] = rotation[..., 1, 1] = c
        rotation[..., 0, 1], rotation[..., 1, 0] = -s, s
        result = result @ origin @ rotation
    return result


def xyz_wxyz(transforms):
    quaternion = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    for index in range(1, len(quaternion)):
        if quaternion[index] @ quaternion[index - 1] < 0:
            quaternion[index] *= -1
    return np.concatenate((transforms[:, :3, 3], quaternion), -1)


def scaled_intrinsics(K, source_hw, target_hw):
    result = np.asarray(K, dtype=np.float64).copy()
    sy, sx = target_hw[0] / source_hw[0], target_hw[1] / source_hw[1]
    result[0, :] *= sx
    result[1, :] *= sy
    result[0, 2] += (sx - 1) / 2
    result[1, 2] += (sy - 1) / 2
    return result


def rectify_resize(rgb, depth, maps, target_hw):
    """Remove Brown distortion identically in RGB/depth before pinhole resizing."""
    rgb = cv2.remap(rgb, *maps, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
    depth = cv2.remap(depth, *maps, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float()
    rgb = F.interpolate(tensor, target_hw, mode='bilinear', align_corners=False, antialias=True)
    depth = F.interpolate(torch.from_numpy(depth)[None, None], target_hw, mode='nearest-exact')
    return rgb[0].round().clamp(0, 255).byte().permute(1, 2, 0).numpy(), depth[0, 0].numpy()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_record(path, root):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=digest.hexdigest())


def validate_processed(path):
    """Audit the real schema without accepting it as a simulated benchmark episode."""
    with h5py.File(path, 'r') as f:
        count = len(f['qpos'])
        height, width = f['depth_m'].shape[1:]
        shapes = dict(qpos=(count, 8), ee_pose=(count, 7), T_base_camera_cv=(count, 4, 4),
                      T_base_depth_camera_cv=(count, 4, 4), T_ee_camera_cv=(4, 4), intrinsics=(3, 3),
                      goal_pose_xyz_wxyz=(7,), time_seconds=(count,), rgb=(count, height, width, 3))
        if f.attrs['schema_version'] != SCHEMA or count < 2:
            raise ValueError('Unexpected real-data schema or trajectory length')
        for key, shape in shapes.items():
            if f[key].shape != shape or not np.isfinite(f[key][:]).all():
                raise ValueError(f'Invalid processed {key}')
        if not np.allclose(np.diff(f['time_seconds'][:]), .05, atol=1e-9):
            raise ValueError('Processed trajectory must be at 20 Hz')
        if not np.allclose(f['qpos'][:, 7], 0) or not f['gripper_open'][:].all():
            raise ValueError('User-specified always-open Robotiq state was not preserved')
        for key in ('T_base_camera_cv', 'T_base_depth_camera_cv'):
            matrices = f[key][:]
            if (not np.allclose(matrices[:, 3], [0, 0, 0, 1])
                    or not np.allclose(np.linalg.det(matrices[:, :3, :3]), 1, atol=1e-5)):
                raise ValueError(f'Invalid rigid transform {key}')
        if not np.allclose(np.linalg.norm(f['ee_pose'][:, 3:], axis=1), 1, atol=1e-6):
            raise ValueError('Invalid wxyz quaternions')
        if np.any(f['rgb_time_seconds'][:] > f['time_seconds'][:] + 1e-9):
            raise ValueError('An exported RGB observation looks into the future')
        valid = f['depth_pair_valid'][:]
        if np.any(f['depth_m'][:][~valid] != 0) or np.any(f['depth_m'][:] < 0):
            raise ValueError('Invalid or unsynchronized depth was not masked')
        return dict(passed=True, frames=count, image_hw=[height, width],
                    rgb_dtype=str(f['rgb'].dtype), depth_dtype=str(f['depth_m'].dtype))


def process_episode(root, episode, output, calibration, target_hw=(192, 256), max_skew_ms=20.):
    root, output = Path(root).resolve(), Path(output)
    rows = read_metadata(root / episode / 'data.pkl')
    frame_ids = np.asarray([row['frame_id'] for row in rows], dtype=np.int64)
    stamps = {name: np.asarray([timestamp_ns(getter(row)) for row in rows], dtype=np.int64)
              for name, getter in {
                  'rgb': lambda r: r['timestamps']['images']['front'],
                  'depth': lambda r: r['timestamps']['depths']['front_depth'],
                  'joints': lambda r: r['timestamps']['states']['/franka/joint_states'],
                  'trigger': lambda r: r['timestamps']['trigger'],
              }.items()}
    origin = stamps['rgb'][0]
    times = {key: (value - origin) / 1e9 for key, value in stamps.items()}
    if np.any(np.diff(times['rgb']) <= 0):
        raise ValueError(f'{episode}: RGB timestamps are not strictly increasing')
    q = np.stack([row['/franka/joint_states']['position'] for row in rows])
    qvel = np.stack([row['/franka/joint_states']['velocity'] for row in rows])
    effort = np.stack([row['/franka/joint_states']['effort'] for row in rows])
    commands = np.stack([row['/factr_teleop/right/cmd_franka_pos']['position'] for row in rows])
    measured_ee = np.stack([row['/franka/end_effector_pose'] for row in rows])
    for values, shape in ((q, (len(rows), 7)), (qvel, q.shape), (effort, q.shape),
                          (commands, q.shape), (measured_ee, (len(rows), 4, 4))):
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f'{episode}: invalid recorded robot array')
    if (not np.allclose(measured_ee[:, 3], [0, 0, 0, 1]) or
            not np.allclose(np.linalg.det(measured_ee[:, :3, :3]), 1, atol=1e-4)):
        raise ValueError(f'{episode}: invalid recorded EE transforms')
    rgb_paths = [payload_path(root, episode, row['images']['front'], 'front', '.jpg') for row in rows]
    depth_paths = [payload_path(root, episode, row['depths']['front_depth'], 'front_depth', '.npy') for row in rows]
    unused_rgb = sorted(set((root / episode / 'front').glob('*.jpg')) - set(rgb_paths))
    unused_depth = sorted(set((root / episode / 'front_depth').glob('*.npy')) - set(depth_paths))
    source_files = [file_record(path, root) for path in [root / episode / 'data.pkl', *rgb_paths, *depth_paths,
                                                       *unused_rgb, *unused_depth]]
    native_hw = tuple(calibration['image_hw'])
    K = np.asarray(calibration['K']).reshape(3, 3)
    mount = np.asarray(calibration['T_link7_camera_cv']).reshape(4, 4)
    if (len(calibration['joint_origins']) != 7 or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1])
            or not np.isfinite(mount).all() or not np.allclose(mount[3], [0, 0, 0, 1])
            or not np.allclose(mount[:3, :3].T @ mount[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(mount[:3, :3]), 1, atol=1e-6)
            or calibration['gripper']['state'] != 'always_open'
            or calibration['gripper']['open_joint_position_rad'] != 0):
        raise ValueError('Invalid pinhole calibration, camera transform, or open Robotiq configuration')
    native_valid_fraction, native_invalid_count, native_max_depth = [], 0, 0.
    # Check every downloaded payload, including frames not selected on the 20 Hz grid.
    for rgb_path in rgb_paths + unused_rgb:
        with Image.open(rgb_path) as image:
            if image.size != native_hw[::-1]:
                raise ValueError(f'RGB size disagrees with calibration: {rgb_path}')
            image.load()
    for depth_path in depth_paths + unused_depth:
        depth = np.load(depth_path, allow_pickle=False, mmap_mode='r')
        if depth.shape != native_hw or depth.dtype.kind != 'f':
            raise ValueError(f'Depth shape/dtype disagrees with calibration: {depth_path}')
        finite = np.isfinite(depth)
        native_invalid_count += int((~finite | (depth < 0)).sum())
        native_valid_fraction.append(float((finite & (depth >= .02) & (depth <= 2)).mean()))
        if finite.any():
            native_max_depth = max(native_max_depth, float(depth[finite].max()))
    maps = cv2.initUndistortRectifyMap(K, np.asarray(calibration['distortion']), np.eye(3), K,
                                     native_hw[::-1], cv2.CV_32FC1)
    flange = np.eye(4)
    flange[2, 3] = calibration['link7_to_flange_z_m']
    # Include a single terminal hold on the next 20 Hz tick; never speed up a demonstration.
    grid = np.arange(int(np.ceil(times['rgb'][-1] * 20)) + 1, dtype=np.float64) / 20
    current_q = interpolate_joints(times['joints'], q, grid)
    current_q[grid > times['rgb'][-1]] = q[-1]
    rgb_indices = causal_indices(times['rgb'], grid)
    depth_indices = nearest_indices(times['depth'], times['rgb'][rgb_indices])
    delta_ms = (times['depth'][depth_indices] - times['rgb'][rgb_indices]) * 1000
    pair_valid = np.abs(delta_ms) <= max_skew_ms
    rgb_q = interpolate_joints(times['joints'], q, times['rgb'][rgb_indices])
    depth_q = interpolate_joints(times['joints'], q, times['depth'][depth_indices])
    T_camera = link7_fk(rgb_q, calibration) @ mount
    T_depth = link7_fk(depth_q, calibration) @ mount
    poses = xyz_wxyz(link7_fk(current_q, calibration) @ flange)
    # Goals are hindsight demonstration endpoints; the recorder has no requested-goal field.
    goal = xyz_wxyz((link7_fk(q[-1:], calibration) @ flange))[0]
    folder = output / episode
    folder.mkdir(parents=True, exist_ok=False)
    h5_path = folder / 'episode.h5'
    invalid_count, valid_fractions = 0, []
    with h5py.File(h5_path, 'x') as f:
        f.attrs.update(schema_version=SCHEMA, source_domain='real', source_episode=episode,
                       robot_model='Panda arm with Robotiq 2F-85', route_type='unlabelled',
                       ee_frame='panda_link8 flange; not Panda hand TCP or Robotiq fingertip',
                       camera_convention='OpenCV: x right, y down, z forward; integer pixel centers',
                       depth_unit='metres', rgb_fps=20, gripper_state='always_open; user assumption',
                       goal_source='hindsight final measured joint state FK; not a recorded requested goal',
                       training_ready=False, calibration_provenance=json.dumps(calibration['provenance']),
                       processing='undistort then half-pixel resize; RGB causal hold; measured joints interpolate',
                       max_depth_rgb_skew_ms=max_skew_ms)
        arrays = dict(qpos=np.column_stack((current_q, np.zeros(len(grid)))).astype(np.float32),
                      qvel=interpolate_joints(times['joints'], qvel, grid).astype(np.float32),
                      joint_effort=interpolate_joints(times['joints'], effort, grid).astype(np.float32),
                      ee_pose=poses.astype(np.float32), T_base_camera_cv=T_camera,
                      T_base_depth_camera_cv=T_depth, T_ee_camera_cv=np.linalg.inv(flange) @ mount,
                      intrinsics=scaled_intrinsics(K, native_hw, target_hw), image_size_wh=target_hw[::-1],
                      goal_pose_xyz_wxyz=goal.astype(np.float32), time_seconds=grid,
                      gripper_open=np.ones(len(grid), dtype=bool), source_rgb_frame_ids=frame_ids[rgb_indices],
                      source_depth_frame_ids=frame_ids[depth_indices], rgb_time_seconds=times['rgb'][rgb_indices],
                      depth_time_seconds=times['depth'][depth_indices], depth_rgb_skew_ms=delta_ms,
                      depth_pair_valid=pair_valid, terminal_hold=grid > times['rgb'][-1],
                      joint_names=np.asarray([f'panda_joint{i}' for i in range(1, 8)] + ['finger_joint'], dtype='S'))
        for key, value in arrays.items():
            f.create_dataset(key, data=value)
        f['T_ee_camera_cv'].attrs['parent_frame'] = 'panda_link8 flange'
        f['T_ee_camera_cv'].attrs['note'] = 'Converted from the upstream panda_link7 calibration using +0.107m flange offset'
        native = f.create_group('native')
        for key, value in dict(arm_qpos=q, arm_qvel=qvel, joint_effort=effort,
                               arm_command=commands, T_base_recorded_ee=measured_ee,
                               recorded_ee_pose=xyz_wxyz(measured_ee), frame_ids=frame_ids,
                               time_seconds=times['rgb']).items():
            native.create_dataset(key, data=value)
        for key, value in stamps.items():
            native.create_dataset(key + '_timestamp_ns', data=value)
        native.create_dataset('language_instruction', data=np.asarray([row['language_instruction'] for row in rows],
                                                                       dtype=h5py.string_dtype('utf-8')))
        f.create_dataset('rgb', shape=(len(grid), *target_hw, 3), dtype='u1',
                         chunks=(1, *target_hw, 3), compression='gzip', compression_opts=4, shuffle=True)
        f.create_dataset('depth_m', shape=(len(grid), *target_hw), dtype='f4',
                         chunks=(1, *target_hw), compression='gzip', compression_opts=4, shuffle=True)
        for index, (r, d) in enumerate(zip(rgb_indices, depth_indices)):
            with Image.open(rgb_paths[r]) as image:
                rgb = np.array(image.convert('RGB'))
            depth = np.load(depth_paths[d], allow_pickle=False)
            if rgb.shape != (*native_hw, 3) or depth.shape != native_hw or depth.dtype.kind != 'f':
                raise ValueError(f'{episode}: payload dimensions or depth dtype disagree with calibration')
            invalid_count += int((~np.isfinite(depth) | (depth < 0)).sum())
            rgb, depth = rectify_resize(rgb, depth, maps, target_hw)
            if not pair_valid[index]:
                depth.fill(0)
            f['rgb'][index], f['depth_m'][index] = rgb, depth
            valid_fractions.append(float(((depth >= .02) & (depth <= 2)).mean()))
    audit = validate_processed(h5_path)
    raw_link7 = link7_fk(q, calibration)
    predicted_ee = raw_link7 @ flange
    error_m = np.linalg.norm(predicted_ee[:, :3, 3] - measured_ee[:, :3, 3], axis=1)
    angle_error = Rotation.from_matrix(predicted_ee[:, :3, :3].swapaxes(-1, -2) @ measured_ee[:, :3, :3]).magnitude()
    intervals = np.diff(times['rgb'])
    source_bytes = sum(item['bytes'] for item in source_files)
    report = dict(episode=episode, native_frames=len(rows), frames_20hz=len(grid),
                  native_duration_seconds=float(times['rgb'][-1]), exported_duration_seconds=float(grid[-1]),
                  native_average_hz=float((len(rows) - 1) / times['rgb'][-1]),
                  native_interval_ms=dict(min=float(intervals.min() * 1000), median=float(np.median(intervals) * 1000),
                                          max=float(intervals.max() * 1000)),
                  native_image_hw=list(native_hw), image_hw=list(target_hw),
                  all_native_payloads_checked=True,
                  unreferenced_rgb_files=[p.name for p in unused_rgb],
                  unreferenced_depth_files=[p.name for p in unused_depth],
                  unreferenced_file_policy='Validated and hashed; excluded because no corresponding robot state was recorded',
                  native_depth_invalid_values=native_invalid_count,
                  native_depth_max_m=native_max_depth,
                  native_depth_valid_fraction_002_to_2m_mean=float(np.mean(native_valid_fraction)),
                  source_front_bytes=source_bytes, processed_h5_bytes=h5_path.stat().st_size,
                  compression_ratio=source_bytes / h5_path.stat().st_size,
                  unique_source_rgb_files=len(set(item['sha256'] for item in source_files[1:1 + len(rows)])),
                  original_depth_rgb_skew_max_ms=float(np.max(np.abs(times['depth'] - times['rgb'])) * 1000),
                  matched_depth_rgb_skew_max_ms=float(np.max(np.abs(delta_ms))),
                  depth_pairs_masked=int((~pair_valid).sum()),
                  depth_pairs_usable=int(pair_valid.sum()),
                  matched_usable_depth_rgb_skew_max_ms=float(np.max(np.abs(delta_ms[pair_valid]))) if pair_valid.any() else None,
                  depth_valid_fraction_002_to_2m_mean=float(np.mean(valid_fractions)),
                  invalid_source_depth_values_in_selected_frames=invalid_count,
                  recorded_ee_fk_position_error_mm=dict(median=float(np.median(error_m) * 1000),
                                                        max=float(error_m.max() * 1000)),
                  recorded_ee_fk_rotation_error_deg_max=float(np.rad2deg(angle_error.max())),
                  recorded_ee_timestamp_available=False, command_timestamp_available=False,
                  terminal_measured_joint_speed_max_rad_s=float(np.abs(qvel[-1]).max()),
                  goal_source='hindsight endpoint; success and requested goal were not recorded',
                  training_ready=False, audit=audit)
    write_json(folder / 'source_files.json', source_files)
    write_json(folder / 'report.json', report)
    return report
