"""Check active embodied controller commands against archived rollout code."""
import argparse
import hashlib
import importlib.util
from pathlib import Path

import torch

from tsn.cli.c1_verify import SyntheticRGBBackbone
from tsn.common.config import create_output,write_json
from tsn.models import embodied_clearance
from tsn.models.embodied_clearance import EmbodiedClearancePolicy
from tsn.models.compact_policy import CompactRouteHead
from tsn.models.geometric_energy import RouteTrust
from tsn.models.kinematics import PandaKinematics
from tsn.models.uncertain_clearance import PointUncertainty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archived-source',type=Path,required=True)
    parser.add_argument('--panel',choices=('signed','parts','tcp'),default='signed')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu',),default='cpu')
    args = parser.parse_args()
    create_output(args.output_dir)
    path = args.archived_source/'src/tsn/models/embodied_clearance.py'
    spec = importlib.util.spec_from_file_location('archived_embodied_controller',path)
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    torch.set_num_threads(1);torch.manual_seed(20261007)
    modules = (SyntheticRGBBackbone(),CompactRouteHead(),PandaKinematics(),RouteTrust(),PointUncertainty())
    state = torch.zeros(1,16)
    state[:,:7] = torch.tensor([[0.,.4,0.,-1.96,0.,2.35,.78]])/torch.pi
    state[:,9:12] = (torch.tensor([[.72,.25,.33]])-torch.tensor([.65,0,.22]))/torch.tensor([.55,.55,.5])
    configs = dict(tcp_signed=dict(mode='tcp',body_weight=0.,field='signed'),
                   hand_signed35=dict(mode='hand',body_weight=.35,mask_self=True,field='signed'),
                   tool_signed35=dict(mode='tool',body_weight=.35,mask_self=True,field='signed'),
                   axial=dict(mode='axial',body_weight=1.))
    if args.panel=='parts':
        configs = dict(tcp=dict(mode='tcp',body_weight=0.),
                       tool_parts100=dict(mode='tool',body_weight=1.,mask_self=True,representation='parts'),
                       axial=dict(mode='axial',body_weight=1.))
    elif args.panel=='tcp':
        configs = dict(tcp=dict(mode='tcp',body_weight=0.))
    results = {}
    with torch.inference_mode():
        for name,config in configs.items():
            old = archived.EmbodiedClearancePolicy(*modules,**config).eval()
            active = EmbodiedClearancePolicy(*modules,**config).eval()
            for step in range(0,400,15):
                rgb = torch.full((1,1,1,3),step//15,dtype=torch.uint8)
                opening = .5+.4*torch.sin(torch.tensor(step*.03))
                state[:,7],state[:,8] = opening,1-opening
                chunks = []
                for policy in (old,active):
                    policy.observe_step(step)
                    chunks.append(policy(rgb,state,torch.eye(3)[None],torch.eye(4)[None]))
                torch.testing.assert_close(chunks[0],chunks[1],rtol=0,atol=0)
                for key,value in old.diagnostics[-1].items():
                    if active.diagnostics[-1][key]!=value:
                        raise ValueError('Executed diagnostic differs: '+name+'/'+key)
                if not torch.isfinite(chunks[1]).all():raise ValueError('Nonfinite command')
            results[name]=dict(observations=27,bitwise_equal_commands=True,executed_diagnostics_equal=True)
    write_json(args.output_dir/'complete.json',dict(panel=args.panel,modes=results,
        active_module_sha256=hashlib.sha256(Path(embodied_clearance.__file__).read_bytes()).hexdigest(),
        archived_module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        torch_version=str(torch.__version__),
        scope='Synthetic RGB features exercise complete history/map/uncertainty/body scoring/IK; no Pi3 reconstruction or simulation replay'))
    print(results,flush=True)


if __name__=='__main__':main()
