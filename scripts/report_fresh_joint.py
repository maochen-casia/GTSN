"""Export training curves and route results from a completed fresh joint run."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tsn.common.config import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    records = read_json(root/'training/epochs.json')
    summary = read_json(root/'summary.json')
    x = [row['epoch'] for row in records]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    axes[0].plot(x, [row['validation_rmse_m']*1000 for row in records], color='#2364aa')
    axes[0].axvline(summary['selected_epoch'], color='#666666', linestyle='--', label='Selected epoch')
    axes[0].set(xlabel='Epoch', ylabel='Validation waypoint RMSE (mm)')
    axes[0].legend(frameon=False)
    for key in ('route', 'joints', 'maps', 'uncertainty', 'memory', 'embodiment'):
        axes[1].plot(x, [row['loss_components'][key] for row in records], label=key)
    axes[1].set(xlabel='Epoch', ylabel='Unweighted training loss', yscale='log')
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    routes = ('direct', 'over', 'side')
    for shift, partition, color in ((-.18, 'validation', '#2364aa'), (.18, 'test', '#2a9d8f')):
        axes[2].bar([i+shift for i in range(3)],
                    [100*summary[partition+'_by_route'][route]['success_rate'] for route in routes],
                    width=.36, label=partition, color=color)
    axes[2].set(xticks=range(3), xticklabels=routes, ylabel='Collision-free XYZ success (%)', ylim=(0, 105))
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    fig.savefig(root/'training_curves.png', dpi=180)
    fig.savefig(root/'training_curves.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
