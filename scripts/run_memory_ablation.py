"""Remove historical observations from the SAME frozen memory checkpoint."""
import hashlib
from pathlib import Path
import sys

import run_consensus
from tsn.models.cartesian_policy import CartesianHead


class CurrentOnlyHead(CartesianHead):
    def forward(self, tokens, geometry, poses, ages, mask, state, tcp):
        if self.mode == 'memory':
            tokens, geometry, poses = tokens[:, -1:], geometry[:, -1:], poses[:, -1:]
            ages, mask = ages[:, -1:], mask[:, -1:]
        return super().forward(tokens, geometry, poses, ages, mask, state, tcp)


if __name__ == '__main__':
    # The condition name is fixed so the intervention cannot masquerade as memory.
    if '--name' not in sys.argv or sys.argv[sys.argv.index('--name')+1] != 'memory_current_only':
        raise ValueError('This runner is only for memory_current_only')
    if '--modes' in sys.argv:
        raise ValueError('The matched ablation fixes state and memory checkpoints')
    original_hashes = run_consensus.hashes
    def hashes():
        return {**original_hashes(), 'scripts/run_memory_ablation.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    run_consensus.hashes = hashes
    run_consensus.CartesianHead = CurrentOnlyHead
    run_consensus.main()
