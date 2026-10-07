"""Check active C1 controller parity with an immutable rollout source snapshot."""
import argparse
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F

from tsn.common.config import create_output, write_json
from tsn.models.adaptive_geometry import AdaptiveGeometryPolicy
from tsn.models import adaptive_geometry
from tsn.models.compact_policy import CompactRouteHead
from tsn.models.geometric_energy import RouteTrust
from tsn.models.kinematics import PandaKinematics


class SyntheticRGBBackbone(torch.nn.Module):
    """Deterministic changing geometry for controller parity, never training."""
    def forward(self, rgb, state, K, pose, **kwargs):
        index = float(rgb[0, 0, 0, 0])
        x, y = torch.meshgrid(torch.linspace(.42, .75, 20), torch.linspace(-.18, .18, 20), indexing='ij')
        metric = torch.stack((x+.004*index, y, .30+.04*torch.sin(x*30+index/4)))
        dense = torch.zeros(1, 6, 20, 20)
        dense[0, :3] = (metric-torch.tensor([.65, 0, .22])[:, None, None])/torch.tensor([.55, .55, .5])[:, None, None]
        geometry = F.adaptive_avg_pool2d(dense, (4, 4)).flatten(2).transpose(1, 2)
        tokens = torch.sin(torch.arange(16*768).reshape(1, 16, 768).float()*.01+index*.2)
        return torch.zeros(1, 30, 7), tokens, geometry, dense


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archived-source', type=Path, required=True)
    parser.add_argument('--modes', default='current,recent,persistent,unconfirmed,persistent_visual')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    args = parser.parse_args()
    create_output(args.output_dir)
    path = args.archived_source/'src/tsn/models/adaptive_geometry.py'
    spec = importlib.util.spec_from_file_location('frozen_c1_controller', path)
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    torch.set_num_threads(1)
    torch.manual_seed(20261007)
    modules = (SyntheticRGBBackbone(), CompactRouteHead(), PandaKinematics(), RouteTrust())
    q = torch.tensor([[0., .4, 0., -1.96, 0., 2.35, .78]])
    state = torch.zeros(1, 16)
    state[:, :7] = q/torch.pi
    state[:, 9:12] = (torch.tensor([[.72, .25, .33]])-torch.tensor([.65, 0, .22]))/torch.tensor([.55, .55, .5])
    results = {}
    with torch.inference_mode():
        for mode in args.modes.split(','):
            original = archived.AdaptiveGeometryPolicy(*modules, mode).eval()
            current = AdaptiveGeometryPolicy(*modules, mode).eval()
            old_read = 0
            for step in range(0, 400, 15):
                rgb = torch.full((1, 1, 1, 3), step//15, dtype=torch.uint8)
                chunks = []
                for policy in (original, current):
                    policy.observe_step(step)
                    chunks.append(policy(rgb, state, torch.eye(3)[None], torch.eye(4)[None]))
                torch.testing.assert_close(chunks[0], chunks[1], atol=0, rtol=0)
                if original.diagnostics[-1] != current.diagnostics[-1]:
                    raise ValueError('Controller diagnostic mismatch: '+mode)
                old_read += current.diagnostics[-1]['old_only_points'] > 0
                if not torch.isfinite(chunks[1]).all():
                    raise ValueError('Nonfinite command')
            results[mode] = dict(observations=27, bitwise_equal_commands=True,
                                 identical_diagnostics=True, observations_reading_old=int(old_read))
    write_json(args.output_dir/'complete.json', dict(source=str(args.archived_source), modes=results,
        active_module_sha256=hashlib.sha256(Path(adaptive_geometry.__file__).read_bytes()).hexdigest(),
        archived_module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), torch_version=str(torch.__version__),
        scope='Synthetic RGB-derived features exercise complete perception/history/map/energy/IK controller; not Pi3 reconstruction or simulator replay'))
    print(results, flush=True)


if __name__ == '__main__':
    main()
