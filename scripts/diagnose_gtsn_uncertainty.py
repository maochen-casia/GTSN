"""Held-out uncertainty ranking and calibration; does not update model parameters."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch

from run_gtsn import inputs, load_data
from tsn.common.config import write_json
from tsn.models.gtsn_policy import GTSNHead


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = load_data(args.root / 'cache/validation')
    results = {}
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for mode in ('uncertainty', 'memory'):
        model = GTSNHead(mode).cuda().eval()
        model.load_state_dict(torch.load(args.root / 'heads' / mode / 'best.pt', weights_only=True))
        errors, variances = [], []
        for index in torch.arange(len(data['state']), device='cuda').split(256):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, aux = model(*inputs(data, index))
            valid = data['geometry_valid'][index]
            errors.append((aux['mean'].float()-data['teacher'][index]).square().mean(-1)[valid].cpu().numpy())
            variances.append(aux['logvar'].exp().mean(-1)[valid].cpu().numpy())
        error, variance = np.concatenate(errors), np.concatenate(variances)
        bins = []
        for indices in np.array_split(np.argsort(variance, kind='stable'), 10):
            bins.append({'tokens': len(indices), 'predicted_variance': float(variance[indices].mean()),
                         'observed_mse': float(error[indices].mean())})
        results[mode] = {'partition': 'validation', 'tokens': len(error), 'bins': bins,
            'spearman_variance_vs_squared_error': float(spearmanr(variance, error).statistic),
            'high_decile_mse_over_low_decile': bins[-1]['observed_mse']/bins[0]['observed_mse'],
            'units': 'squared normalized pooled XYZ, averaged over three coordinates',
            'model_updated': False, 'tokens_are_correlated_within_episodes': True}
        ax.plot([b['predicted_variance'] for b in bins], [b['observed_mse'] for b in bins],
                marker='o', label=mode)
    lower, upper = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([max(0, lower), upper], [max(0, lower), upper], '--', color='gray', label='Ideal mean calibration')
    ax.set(xlabel='Mean predicted variance', ylabel='Observed mean squared error',
           title='Validation geometry: equal-count uncertainty bins')
    ax.legend(frameon=False)
    fig.tight_layout()
    figures = args.root / 'figures'
    figures.mkdir(exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(figures / f'uncertainty_calibration.{extension}', dpi=180)
    write_json(args.root / 'uncertainty_diagnostics.json', results)
    print({m: r['spearman_variance_vs_squared_error'] for m, r in results.items()})


if __name__ == '__main__':
    main()
