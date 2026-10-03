"""Checks for the paired episode statistics used in the research report."""
import unittest

from tsn.cli.report import paired


class ResearchStatisticsTests(unittest.TestCase):
    def results(self, success):
        return {'results': [dict(episode_id=f'episode_{i:03}', success=success,
                                route='direct' if i < 20 else ('over' if i < 60 else 'side'))
                            for i in range(100)]}

    def test_identical_outcomes_have_zero_paired_difference(self):
        result = paired(self.results(True), self.results(True))
        self.assertEqual(result['success_difference_pp'], 0.)
        self.assertEqual(result['paired_stratified_95_ci_pp'], [0., 0.])
        self.assertEqual(result['exact_mcnemar_p'], 1.)

    def test_all_gains_have_exact_full_scale_difference(self):
        result = paired(self.results(False), self.results(True))
        self.assertEqual(result['success_difference_pp'], 100.)
        self.assertEqual(result['paired_stratified_95_ci_pp'], [100., 100.])
        self.assertEqual(result['wins'], 100)
        self.assertEqual(result['losses'], 0)
        self.assertLess(result['exact_mcnemar_p'], 1e-20)

    def test_unpaired_episode_sets_are_rejected(self):
        candidate = self.results(True)
        candidate['results'].pop()
        with self.assertRaises(ValueError):
            paired(self.results(False), candidate)
