"""Convert the front-only real-data pilot and build an offline review gallery."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from tsn.data.frankanav import process_episode, write_json


def review_frame(rgb, depth, label):
    color = (plt.get_cmap('turbo')((depth / 1.0).clip(0, 1))[..., :3] * 255).astype(np.uint8)
    color[depth <= 0] = 0
    height, width = depth.shape
    panel = Image.new('RGB', (width * 2, height + 28), '#121c2b')
    panel.paste(Image.fromarray(rgb), (0, 28))
    panel.paste(Image.fromarray(color), (width, 28))
    ImageDraw.Draw(panel).text((8, 7), label + ' | RGB / depth (0-1m; black=invalid)', fill='white')
    return np.array(panel)


def save_ply(path, points, colors):
    with path.open('w') as stream:
        stream.write('ply\nformat ascii 1.0\nelement vertex ' + str(len(points)) + '\n')
        stream.write('property float x\nproperty float y\nproperty float z\n')
        stream.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for point, color in zip(points, colors):
            stream.write(' '.join(f'{v:.6f}' for v in point) + ' ' + ' '.join(str(int(v)) for v in color) + '\n')


def previews(folder, report):
    with h5py.File(folder / 'episode.h5', 'r') as f:
        count = len(f['qpos'])
        selected = [0, count // 2, count - 1]
        panels = []
        with imageio.get_writer(folder / 'review.mp4', fps=20, codec='libx264', quality=7,
                                macro_block_size=1, ffmpeg_log_level='error', ffmpeg_params=['-threads', '1']) as writer:
            for index in range(count):
                label = f"{report['episode']} t={f['time_seconds'][index]:.2f}s source={f['source_rgb_frame_ids'][index]}"
                panel = review_frame(f['rgb'][index], f['depth_m'][index], label)
                writer.append_data(panel)
                if index in selected:
                    panels.append(Image.fromarray(panel))
        storyboard = Image.new('RGB', (panels[0].width * 3, panels[0].height), 'white')
        for index, panel in enumerate(panels):
            storyboard.paste(panel, (index * panel.width, 0))
        storyboard.save(folder / 'storyboard.jpg', quality=92)
        usable = np.flatnonzero(f['depth_pair_valid'][:])
        cloud_index = int(usable[0]) if len(usable) else 0
        rgb, depth, K = f['rgb'][cloud_index], f['depth_m'][cloud_index], f['intrinsics'][:]
        transform = f['T_base_depth_camera_cv'][cloud_index]
        v, u = np.mgrid[:depth.shape[0], :depth.shape[1]]
        rays = np.stack((u, v, np.ones_like(u)), -1) @ np.linalg.inv(K).T
        valid = (depth >= .02) & (depth <= 2)
        points = (rays * depth[..., None])[valid] @ transform[:3, :3].T + transform[:3, 3]
        colors = rgb[valid]
        save_ply(folder / 'front_first_valid_base.ply', points, colors)
        positions, times = f['ee_pose'][:, :3], f['time_seconds'][:]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
        for ax, pair in zip(axes[:2], [(0, 1), (0, 2)]):
            ax.scatter(points[::4, pair[0]], points[::4, pair[1]], c=colors[::4] / 255, s=1, alpha=.5)
            ax.plot(positions[:, pair[0]], positions[:, pair[1]], color='#e85d04', lw=2, label='20 Hz flange path')
            ax.scatter(*positions[0, list(pair)], color='#2a9d8f', s=45, label='start')
            ax.scatter(*positions[-1, list(pair)], color='#9d0208', marker='x', s=65, label='hindsight endpoint')
            ax.set(xlabel='XYZ'[pair[0]] + ' (m)', ylabel='XYZ'[pair[1]] + ' (m)',
                   title=f'Front depth at t={times[cloud_index]:.2f}s in panda_link0')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-.2, 1.1)
            ax.set_ylim(-.65, .65) if pair == (0, 1) else ax.set_ylim(-.15, .9)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        for dim, name in enumerate('XYZ'):
            axes[2].plot(times, positions[:, dim], label=name)
        axes[2].set(xlabel='Recorded elapsed time (s)', ylabel='Flange position (m)', title='Navigation trajectory')
        axes[2].grid(alpha=.2)
        axes[2].legend()
        fig.savefig(folder / 'geometry.png', dpi=140)
        plt.close(fig)


def gallery(output, reports):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>FrankaNav pilot review</title>',
             '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 16px;color:#172638}',
             'img{max-width:100%;height:auto}video{max-width:100%;width:800px}section{border-top:1px solid #ddd;padding:24px 0}',
             'table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd;text-align:left}</style>',
             '<h1>FrankaNav front-camera pilot</h1><p>Three real demonstrations, compressed to 192×256 and normalized to 20 Hz.',
             ' Lens distortion is removed; all point clouds use the calibrated panda_link7 optical transform.',
             ' Joint, RGB and depth source timestamps remain available in the HDF5 files.</p>',
             '<p>Depth with an RGB timestamp mismatch exceeding 20 ms is masked to zero, so some video frames have a black depth panel.',
             ' Geometry plots use the first usable RGB-D pair and show the navigation workspace; the PLY retains all valid points.',
             ' Unreferenced trailing camera files are audited and excluded from trajectories because their robot states are missing.</p>',
             '<p><b>For review before combined training.</b> Gripper state is always open by user instruction.',
             ' The real Robotiq gripper uses one independent joint. The current benchmark loader expects two Panda finger joints;',
             ' mixed training still needs a real-data loader and matching embodiment handling. Goals are hindsight endpoints.',
             ' Real route labels and episode success were not recorded. The tsn-1k-var validation/test splits remain the evaluation set.</p>',
             '<p><a href="report.json">Aggregate audit</a> · <a href="manifest.json">Processed manifest</a> · ',
             '<a href="calibration.json">Pinned calibration and robot definitions</a></p>']
    for report in reports:
        episode = html.escape(report['episode'])
        parts += [f'<section><h2>{episode}</h2><p>{report["native_frames"]} native frames → {report["frames_20hz"]} at 20 Hz;',
                  f' native duration {report["native_duration_seconds"]:.3f}s; HDF5 {report["processed_h5_bytes"]/1e6:.2f} MB;',
                  f' front payload compression {report["compression_ratio"]:.1f}×; {report["depth_pairs_masked"]} unmatched depth frames masked.</p>',
                  f'<video controls preload="metadata" src="{episode}/review.mp4"></video>',
                  f'<p><img src="{episode}/storyboard.jpg" alt="Start, middle and endpoint RGB-D"></p>',
                  f'<p><img src="{episode}/geometry.png" alt="Calibrated depth points and flange path"></p>',
                  f'<p><a href="{episode}/episode.h5">HDF5</a> · <a href="{episode}/report.json">Episode audit</a> · ',
                  f'<a href="{episode}/source_files.json">Source hashes</a> · <a href="{episode}/front_first_valid_base.ply">Base-frame point cloud</a></p></section>']
    (output / 'index.html').write_text('\n'.join(parts) + '</html>\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/datasets_v2/chenmao/frankanav'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, default=Path('/workspace/configs/frankanav_calibration.json'))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--max-depth-rgb-skew-ms', type=float, default=20.)
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.start + args.count > 100:
        parser.error('Episode selection must be within 00000..00099')
    if min(args.height, args.width) <= 0 or not np.isfinite(args.max_depth_rgb_skew_ms) or args.max_depth_rgb_skew_ms < 0:
        parser.error('Positive image dimensions and a finite, nonnegative timestamp tolerance are required')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory; existing artifacts are never replaced')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(args.calibration.read_text())
    write_json(args.output_dir / 'calibration.json', calibration)
    reports = []
    for index in range(args.start, args.start + args.count):
        episode = f'ep_{index:05d}'
        print(f'{episode}: processing and auditing front RGB-D', flush=True)
        report = process_episode(args.root, episode, args.output_dir, calibration,
                                 (args.height, args.width), args.max_depth_rgb_skew_ms)
        previews(args.output_dir / episode, report)
        reports.append(report)
        print(f'{episode}: {report["native_frames"]} → {report["frames_20hz"]} frames; audit passed', flush=True)
    source_root = Path(__file__).resolve().parents[1]
    sources = [source_root / 'src/tsn/data/frankanav.py', Path(__file__).resolve(), args.calibration.resolve()]
    manifest = dict(schema_version='frankanav-front-1.0', domain='real', camera='front',
                    total_episodes=len(reports), episodes=[dict(episode=r['episode'], frames=r['frames_20hz'],
                    h5_sha256=hashlib.sha256((args.output_dir / r['episode'] / 'episode.h5').read_bytes()).hexdigest())
                    for r in reports], training_ready=False, benchmark_evaluation='tsn-1k-var only')
    write_json(args.output_dir / 'manifest.json', manifest)
    write_json(args.output_dir / 'report.json', dict(status='pilot_prepared_for_review', episodes=reports,
        native_frames=sum(r['native_frames'] for r in reports), frames_20hz=sum(r['frames_20hz'] for r in reports),
        source_front_bytes=sum(r['source_front_bytes'] for r in reports),
        processed_h5_bytes=sum(r['processed_h5_bytes'] for r in reports),
        depth_pairs_masked=sum(r['depth_pairs_masked'] for r in reports),
        all_episode_audits_passed=all(r['audit']['passed'] for r in reports),
        gripper_assumption=calibration['gripper'], source_root=str(args.root.resolve()),
        code_and_config_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        next_stage='After pilot review: real embodiment/loader integration, combined training, benchmark-only closed-loop evaluation'))
    gallery(args.output_dir, reports)
    print(f'Review gallery: {args.output_dir / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
