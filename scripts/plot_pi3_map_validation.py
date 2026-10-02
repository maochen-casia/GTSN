"""Export map-intervention results and fixed validation examples in Docker."""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tsn.common.config import experiment_path, read_json
from tsn.data.splits import episode_catalog


def save(fig, directory, name):
    for extension in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = experiment_path(args.root)
    result = read_json(root / 'results.json')
    directory = root / 'figures'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    names = ['predicted', 'oracle_point', 'oracle_goal', 'oracle_future_action',
             'oracle_point_goal', 'oracle_goal_future_action', 'oracle_point_goal_future_action']
    labels = ['Predicted', 'GT point', 'GT goal', 'GT future action', 'GT point + goal', 'GT goal + action', 'All GT']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    values = np.array([result['interventions'][name]['rmse_rad'] for name in names])
    intervals = np.array([result['interventions'][name]['rmse_95ci_rad'] for name in names])
    axes[0].barh(labels[::-1], values[::-1], color=['#397a73']*6+['#657eaa'])
    # Show CIs as absolute endpoints: asymmetric bootstraps need not bracket estimates.
    for y, interval in enumerate(intervals[::-1]):
        axes[0].plot(interval, [y, y], color='black', linewidth=1.1)
    axes[0].set(xlabel='Validation joint RMSE (rad)', title='Frozen action head: map replacement', xlim=(0, .082))
    for name, label in [('predicted', 'Predicted maps'), ('oracle_point_goal', 'GT point + goal'),
                        ('oracle_point_goal_future_action', 'All ground truth')]:
        axes[1].plot(range(1, 31), result['interventions'][name]['rmse_by_horizon_rad'], label=label)
    axes[1].axvline(15, color='grey', linestyle=':', label='Execution horizon')
    axes[1].set(xlabel='Predicted horizon', ylabel='RMSE (rad)', title='Error by action horizon')
    axes[1].legend(fontsize=8)
    save(fig, directory, 'map_interventions')
    protocol = read_json(root / 'protocol.json')
    ids = protocol['episode_ids']
    catalog = episode_catalog(Path('/run/user/1016/tsn-1k'))
    cache = root / 'validation_cache'
    episode = np.load(cache / 'episode.npy', mmap_mode='r')
    predicted = np.load(cache / 'predicted.npy', mmap_mode='r')
    teacher = np.load(cache / 'teacher.npy', mmap_mode='r')
    examples = []
    for route in ('direct', 'over', 'side'):
        selected = next(x for x in ids if catalog[x] == route)
        indices = np.flatnonzero(episode == ids.index(selected))
        for fraction in (.25, .75):
            examples.append((route, selected, int(indices[int((len(indices)-1)*fraction)]), int((len(indices)-1)*fraction)))
    fig, axes = plt.subplots(6, 7, figsize=(16, 13), constrained_layout=True)
    titles = ['Sensor RGB', 'Point GT (XYZ color)', 'Point predicted', 'Visible goal GT', 'Visible goal predicted', 'Future action GT', 'Future action predicted']
    for row, (route, episode_id, index, frame) in enumerate(examples):
        with h5py.File(Path('/run/user/1016/tsn-1k') / episode_id / 'episode.h5', 'r') as handle:
            rgb = handle['rgb'][frame]
        p, t = predicted[index].astype(np.float32), teacher[index].astype(np.float32)
        panels = [rgb, np.moveaxis((t[:3]+2)/4, 0, -1).clip(0, 1),
                  np.moveaxis((p[:3]+2)/4, 0, -1).clip(0, 1), t[4], p[4], t[5], p[5]]
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel, **({'cmap': 'magma', 'vmin': 0, 'vmax': 1} if panel.ndim == 2 else {}))
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=9)
        axes[row, 0].set_ylabel(f'{route}\n{episode_id}\nframe {frame}', fontsize=9)
    fig.suptitle('Fixed examples: 25% and 75% through the first validation episode of each route', fontsize=12)
    save(fig, directory, 'map_examples')
    adapted = root / 'adapted_heads'
    if adapted.is_dir():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for seed in sorted(adapted.glob('seed_*')):
            for mode in ('predicted', 'oracle_point_goal', 'oracle_all', 'zero_all'):
                path = seed / mode / 'results.json'
                if not path.is_file():
                    continue
                item = read_json(path)
                history = read_json(seed / mode / 'epochs.json')
                label = mode.replace('_', ' ') + ' / ' + seed.name.replace('seed_', '')
                axes[0].plot([0]+[x['epoch'] for x in history], [item['initial_validation']['rmse_rad']]+[x['validation']['rmse_rad'] for x in history], label=label)
                axes[1].plot(range(1, 31), item['selected_validation']['rmse_by_horizon_rad'], label=label)
        axes[0].set(xlabel='Action-head adaptation epoch', ylabel='Validation RMSE (rad)', title='Matched training with frozen map heads')
        axes[1].set(xlabel='Predicted horizon', ylabel='Validation RMSE (rad)', title='Selected adapted heads')
        axes[1].axvline(15, color='grey', linestyle=':')
        axes[0].legend(fontsize=7)
        save(fig, directory, 'adapted_heads')
    print(f'Figures exported to {directory}', flush=True)


if __name__ == '__main__':
    main()
