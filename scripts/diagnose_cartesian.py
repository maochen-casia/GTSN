"""Validation-only decoder diagnostics; no selection on test observations."""
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from run_cartesian import ROOT, data, forward, SEED
from tsn.models.cartesian_policy import CartesianHead,PandaKinematics
from tsn.common.config import write_json

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    d=data('validation',PandaKinematics().cuda())
    result={}
    for mode in ('state','visual','uncertainty','memory'):
        model=CartesianHead(mode).cuda().eval()
        model.load_state_dict(torch.load(ROOT/'heads'/mode/'best.pt',weights_only=True))
        errors=[];variances=[];waypoint=[]
        for idx in torch.arange(len(d['state']),device='cuda').split(256):
            with torch.autocast('cuda',dtype=torch.bfloat16):pred,aux=forward(model,d,idx)
            valid=d['geometry_valid'][idx]
            errors.append((aux['mean']-d['teacher'][idx]).square()[valid].cpu().numpy())
            variances.append(aux['logvar'].exp()[valid].cpu().numpy())
            waypoint.append((pred-d['waypoint'][idx]).square().cpu().numpy())
        err,var,wp=np.concatenate(errors),np.concatenate(variances),np.concatenate(waypoint)
        result[mode]=dict(waypoint_rmse_m=float(np.sqrt(wp.mean())),waypoint_rmse_by_horizon_m=np.sqrt(wp.mean((0,2))).tolist(),geometry_variance_supervised=mode in ('uncertainty','memory'))
        if mode in ('uncertainty','memory'):
            result[mode].update(geometry_rmse_normalized=float(np.sqrt(err.mean())),marginal_95_coverage=float((err<=1.96**2*var).mean()),standardized_squared_error=float((err/var).mean()),variance_error_spearman=float(spearmanr(var.mean(1),err.mean(1)).statistic))
    write_json(ROOT/'decoder_diagnostics.json',result)
    print(result)

if __name__=='__main__':main()
