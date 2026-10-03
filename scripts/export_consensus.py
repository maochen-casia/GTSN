"""Package the validation-selected lightweight heads with exact backbone provenance."""
import hashlib
from pathlib import Path

import torch

from tsn.common.config import read_json, write_json
from run_consensus import ROOT, BASE_ROOT, CHECKPOINT


def main():
    selection = read_json(ROOT/'selection.json')
    name = selection['selected']
    protocol = read_json(ROOT/'rollouts/validation'/name/'protocol.json')
    spec = protocol['specification']
    intervention = 'current_only' if name == 'memory_current_only' else ('no_geometry' if name == 'memory_no_geometry' else 'none')
    package = dict(format='gtsn-consensus-v1', specification=spec, intervention=intervention, heads={},
                   backbone_checkpoint=str(CHECKPOINT),
                   backbone_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())
    for mode in spec['modes']:
        path = BASE_ROOT/'heads'/mode/'best.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol['weight_sha256'][str(path)]
        package['heads'][mode] = torch.load(path, map_location='cpu', weights_only=True)
    dest = ROOT/'selected_heads.pt'
    if dest.exists():
        raise FileExistsError(dest)
    torch.save(package, dest)
    write_json(ROOT/'selected_policy.json',dict(condition=name,
        specification=spec, heads_checkpoint=str(dest),
        intervention=intervention,
        heads_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        backbone_checkpoint=str(CHECKPOINT), backbone_sha256=package['backbone_sha256'],
        docker_image='gtsn-pi3:20261002',
        docker_image_id='sha256:5716cc81a619a9d00e6d08edecedff42a5b2b57fe10c8ff22835920cb480e009',
        orientation='baseline', chunk_size=30, observation_history=1 if intervention=='current_only' else 4,
        input_contract=['RGB uint8','measured joints','goal pose','camera intrinsics','camera-to-base pose'],
        forbidden_inputs=['measured depth','scene objects','route label','expert future or progress'],
        selection_rule=selection['rule'], numerical_source_sha256=protocol['source_sha256_at_start']))
    print(name, dest, flush=True)


if __name__ == '__main__': main()
