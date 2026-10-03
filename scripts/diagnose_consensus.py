"""Compare action disagreement with actual waypoint error using cached validation."""
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from run_cartesian import data, forward, ROOT as BASE_ROOT
from run_consensus import ROOT
from tsn.models.cartesian_policy import CartesianHead, PandaKinematics
from tsn.common.config import write_json


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    kin = PandaKinematics().cuda()
    d = data('validation', kin)
    predictions = {}
    for mode in ('state', 'visual', 'uncertainty', 'memory'):
        head = CartesianHead(mode).cuda().eval()
        head.load_state_dict(torch.load(BASE_ROOT/'heads'/mode/'best.pt', weights_only=True))
        out = []
        for idx in torch.arange(len(d['state']), device='cuda').split(256):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(forward(head, d, idx)[0].cpu().numpy())
        predictions[mode] = np.concatenate(out)
    target = d['waypoint'].cpu().numpy()
    results = {'individual_rmse_m': {k: float(np.sqrt(np.mean((v-target)**2))) for k,v in predictions.items()}}
    results['error_correlation'] = {}
    for first in predictions:
        for second in predictions:
            if first < second:
                x, y = (predictions[first]-target).ravel(), (predictions[second]-target).ravel()
                results['error_correlation'][first+' vs '+second] = float(np.corrcoef(x,y)[0,1])
    for modes in [('state','memory'), ('state','uncertainty'), ('state','uncertainty','memory')]:
        pred = np.stack([predictions[m] for m in modes])
        mean = pred.mean(0)
        disagreement = np.sqrt(np.mean(np.sum((pred[:,:,:3]-mean[None,:,:3])**2,axis=-1),axis=(0,2)))
        error = np.sqrt(np.mean(np.sum((mean[:,:3]-target[:,:3])**2,axis=-1),axis=-1))
        results['_'.join(modes)] = dict(rmse_m=float(np.sqrt(np.mean((mean-target)**2))),
            disagreement_quantiles_m=np.quantile(disagreement,[.25,.5,.75,.9,.95]).tolist(),
            disagreement_error_spearman=float(spearmanr(disagreement,error).statistic),
            high_quartile_error_m=float(error[disagreement>=np.quantile(disagreement,.75)].mean()),
            low_quartile_error_m=float(error[disagreement<=np.quantile(disagreement,.25)].mean()))
    write_json(ROOT/'waypoint_diagnostics.json',results)
    print(results,flush=True)


if __name__ == '__main__':
    main()
