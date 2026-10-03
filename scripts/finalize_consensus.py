"""Verify the exported policy and completed study, then write final status."""
import hashlib
from pathlib import Path
import torch
from tsn.common.config import read_json, write_json
from tsn.models.cartesian_policy import CartesianHead
from run_consensus import ROOT, BASE_ROOT


def main():
    audit = read_json(ROOT/'audit.json')
    selection = read_json(ROOT/'selection.json')
    exported = read_json(ROOT/'selected_policy.json')
    results = read_json(ROOT/'results.json')
    runs = read_json(ROOT/'container_runs.json')
    assert audit['passed'] and audit['episodes'] == 1000 and len(audit['conditions']) == 14
    assert set(results['rollouts']['test']) == set(selection['test_conditions'])
    assert all(r['state']['Status']=='exited' and r['state']['ExitCode']==0 for r in runs)
    path = Path(exported['heads_checkpoint'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == exported['heads_sha256']
    package = torch.load(path, map_location='cpu', weights_only=True)
    assert package['specification'] == selection['test_conditions'][selection['selected']]
    for mode, weights in package['heads'].items():
        head = CartesianHead(mode)
        head.load_state_dict(weights, strict=True)
        original = torch.load(BASE_ROOT/'heads'/mode/'best.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(weights[k], original[k]) for k in original)
    assert hashlib.sha256(Path(package['backbone_checkpoint']).read_bytes()).hexdigest() == package['backbone_sha256']
    sources = [Path('/workspace/documents/research_story.md'), *Path('/workspace/scripts').glob('*consensus*.py')]
    write_json(ROOT/'status.json',dict(status='complete', selected=selection['selected'],
        validation_rollouts=600, test_rollouts=400, audit_passed=True, export_verified=True,
        selected_test_success=results['rollouts']['test'][selection['selected']]['success_rate'],
        baseline_test_success=.72, selected_gain_statistically_conclusive=False,
        artifact_sha256={str(p.relative_to('/workspace')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    print('Complete: 1000 audited rollouts; exported heads and backbone checksums verified.',flush=True)


if __name__ == '__main__': main()
