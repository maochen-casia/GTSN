import unittest

import numpy as np
import torch

from tsn.data.consensus_dataset import ConsensusDataset
from tsn.models.cartesian_policy import CartesianHead
from tsn.models.fresh_consensus import StateRouteHead, ZeroDistribution


class FakeExpert:
    offsets = [0, 40, 80]
    ids = ['first', 'second']
    stride = 2

    def __len__(self):
        return 80

    def _handle(self, episode):
        offset = 0 if episode == 'first' else 100
        frames = np.arange(80)+offset
        return {'rgb': np.broadcast_to(frames[:, None, None, None], (80, 2, 2, 3)),
                'qpos': np.broadcast_to(frames[:, None], (80, 9)),
                'T_base_camera_cv': np.broadcast_to(frames[:, None, None], (80, 4, 4))}

    def __getitem__(self, index):
        frame = index % 40*2
        handle = self._handle(self.ids[index//40])
        return {'frame_index': frame, 'rgb': torch.tensor(handle['rgb'][frame]),
                'qpos': torch.tensor(handle['qpos'][frame]), 'T_B_C': torch.tensor(handle['T_base_camera_cv'][frame])}


class FreshConsensusTests(unittest.TestCase):
    def test_history_never_crosses_episode_or_recovery_reset(self):
        expert = FakeExpert()
        recovery = [expert[30]]
        dataset = ConsensusDataset(expert, recovery)
        late = dataset[30]
        self.assertEqual(late['history_rgb'][:, 0, 0, 0].tolist(), [14, 30, 44, 60])
        self.assertEqual(late['history_ages'].tolist(), [46, 30, 16, 0])
        reset = dataset[40]
        self.assertEqual(reset['history_mask'].tolist(), [False, False, False, True])
        self.assertEqual(reset['history_rgb'][-1, 0, 0, 0].item(), 100)
        perturbation = dataset[80]
        self.assertEqual(perturbation['history_mask'].tolist(), [False, False, False, True])
        self.assertEqual(perturbation['history_ages'].sum(), 0)

    def test_state_branch_matches_original_without_unused_modules(self):
        torch.manual_seed(73)
        original = CartesianHead('state')
        torch.manual_seed(73)
        pruned = StateRouteHead()
        args = (torch.randn(2, 4, 16, 768), torch.randn(2, 4, 16, 6),
                torch.eye(4).expand(2, 4, 4, 4), torch.zeros(2, 4),
                torch.ones(2, 4, dtype=torch.bool), torch.randn(2, 16), torch.eye(4).expand(2, 4, 4))
        torch.testing.assert_close(original(*args)[0], pruned(*args)[0])
        pruned(*args)[0].sum().backward()
        self.assertTrue(all(p.requires_grad and p.grad is not None for p in pruned.parameters()))
        self.assertFalse(any(name.startswith('visual') for name, _ in pruned.named_parameters()))

    def test_deterministic_branch_does_not_need_distribution_parameters(self):
        head = CartesianHead('visual')
        args = (torch.randn(2, 4, 16, 768), torch.randn(2, 4, 16, 6),
                torch.eye(4).expand(2, 4, 4, 4), torch.zeros(2, 4),
                torch.ones(2, 4, dtype=torch.bool), torch.randn(2, 16), torch.eye(4).expand(2, 4, 4))
        expected = head(*args)[0]
        head.distribution = ZeroDistribution()
        torch.testing.assert_close(head(*args)[0], expected)
        head(*args)[0].sum().backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters()))


if __name__ == '__main__':
    unittest.main()
