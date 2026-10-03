"""Zero the explicit geometry inputs while preserving RGB tokens and weights."""
import hashlib
from pathlib import Path
import sys

import torch
import run_consensus
from tsn.models.cartesian_policy import CartesianHead


class NoGeometryHead(CartesianHead):
    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        return super().forward(tokens, torch.zeros_like(geometry), poses, ages, mask, state, tcp)


if __name__ == '__main__':
    if '--name' not in sys.argv or sys.argv[sys.argv.index('--name')+1] != 'memory_no_geometry':
        raise ValueError('This runner is only for memory_no_geometry')
    if '--modes' in sys.argv:
        raise ValueError('The intervention fixes state and memory checkpoints')
    original_hashes = run_consensus.hashes
    def hashes():
        return {**original_hashes(), 'scripts/run_geometry_ablation.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    run_consensus.hashes = hashes
    run_consensus.CartesianHead = NoGeometryHead
    run_consensus.main()
