"""Export experiment learning curves and route outcomes using Docker's matplotlib."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    epochs = [json.loads(line) for line in (root / 'train/epochs.jsonl').read_text().splitlines()]
    results = json.loads((root / 'results.json').read_text())
    directory = root / 'figures'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    x = [epoch['epoch'] for epoch in epochs]
    axes[0].plot(x, [e['train']['rmse_rad'] for e in epochs], marker='o', label='Training mixture')
    axes[0].plot(x, [e['validation']['rmse_rad'] for e in epochs], marker='o', label='Validation expert')
    axes[0].axvline(results['checkpoint_epoch'], color='grey', linestyle=':', label='Selected epoch')
    axes[0].set(xlabel='Epoch', ylabel='Joint RMSE (rad)', title='Action prediction')
    axes[0].legend(fontsize=8)
    for channel in ['point', 'heatmap']:
        axes[1].plot(x, [e['map_loss'][channel] for e in epochs], marker='o', label=channel.title())
    axes[1].set(xlabel='Epoch', ylabel='Unweighted component loss', title='Learned map supervision')
    axes[1].legend(fontsize=8)
    horizon = np.arange(1, 31)
    axes[2].plot(horizon, results['open_loop']['rmse_by_horizon_rad'])
    axes[2].axvline(15, color='grey', linestyle=':', label='Execution horizon')
    axes[2].set(xlabel='Predicted horizon', ylabel='Joint RMSE (rad)', title='Held-out test predictions')
    axes[2].legend(fontsize=8)
    for extension in ['png', 'pdf']:
        fig.savefig(directory / f'learning_curves.{extension}', dpi=180)
    plt.close(fig)
    routes = ['direct', 'over', 'side']
    closed = json.loads((root / 'test/closed_loop.json').read_text())['results']
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    bottom = np.zeros(3)
    for termination, color in [('success', '#287c6f'), ('collision', '#ba4b44'), ('timeout', '#aaa38d')]:
        rates = np.array([sum(r['route'] == route and r['termination'] == termination for r in closed)
                          / sum(r['route'] == route for r in closed) for route in routes])
        ax.bar(routes, rates * 100, bottom=bottom, label=termination.title(), color=color)
        bottom += rates * 100
    ax.set(ylabel='Test episodes (%)', ylim=(0, 100), title='Closed-loop outcomes by route')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.18), ncol=3, frameon=False)
    for extension in ['png', 'pdf']:
        fig.savefig(directory / f'route_outcomes.{extension}', dpi=180)
    plt.close(fig)
    print(f'Exported experiment figures: {directory}', flush=True)


if __name__ == '__main__':
    main()
