"""Check that seed summaries aggregate correctly and refuse unmatched runs."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_cam_var_seeds import compare
from tsn.common.config import PROJECT_ROOT, read_json, write_json


class SeedComparisonTests(unittest.TestCase):
    def make_runs(self, directory):
        config = read_json(PROJECT_ROOT/'configs/cam_var.json')
        recovery = directory/'shared_recovery'
        write_json(recovery/'source_split.json', dict(train=['train'], validation=['val'], test=['test']))
        config['train']['recovery_root'] = str(recovery)
        roots = []
        for i,(seed,full,ablated) in enumerate(((10, 69, 71), (11, 71, 70), (12, 73, 74))):
            root = directory/f'seed_{seed}'
            root.mkdir()
            (root/'recovery').symlink_to(recovery, target_is_directory=True)
            write_json(root/'benchmark_sha256.json', {'manifest.json':'same benchmark'})
            write_json(root/'source_sha256.json', {'src/model.py':'same model',
                                                 'scripts/run.py':f'orchestration {i}'})
            write_json(root/'recovery_sha256.json', {'sample.npz':'same recovery'})
            summary = dict(protocol=dict(seed=seed), test_used_for_selection=False, models={},
                paired_comparisons={'without_c1':{}}, audit=dict(all_rollouts_complete=True,
                source_hashes_verified=True, benchmark_contents_verified=True,
                complete_disjoint_partitions=True, finite_weights_and_trajectories=True,
                all_pi3_encoders_fully_tuned=True))
            for variant,successes in (('full',full), ('without_c1',ablated)):
                variant_config = copy.deepcopy(config)
                variant_config['train']['seed'] = seed
                variant_config['model']['contributions']['c1'] = variant == 'full'
                write_json(root/'configs'/f'{variant}.json', variant_config)
                summary['models'][variant] = dict(partitions={})
                for partition in ('validation', 'test'):
                    summary['models'][variant]['partitions'][partition] = dict(
                        successes=successes, success_rate=successes/100)
                    write_json(root/variant/partition/'closed_loop.json', {'results':[{}]*100})
            write_json(root/'summary.json', summary)
            roots.append(root)
        return roots

    def test_three_seed_statistics_preserve_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = compare(self.make_runs(Path(temporary)))
            self.assertEqual(result['seeds'], [10, 11, 12])
            self.assertAlmostEqual(result['models']['full']['test']['mean'], 71)
            self.assertAlmostEqual(result['models']['full']['test']['sample_sd'], 2)
            self.assertEqual(result['full_minus_c1_pp']['test']['by_seed'], [-2, 1, -1])
            self.assertAlmostEqual(result['full_minus_c1_pp']['test']['mean'], -2/3)
            self.assertEqual(len(result['records']), 3)

    def test_comparison_refuses_other_training_changes_and_duplicate_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self.make_runs(Path(temporary))
            with self.assertRaises(AssertionError):
                compare([roots[0], roots[0]])
            config = read_json(roots[1]/'configs/full.json')
            config['train']['learning_rate'] *= 2
            write_json(roots[1]/'configs/full.json', config)
            with self.assertRaises(AssertionError):
                compare(roots)
