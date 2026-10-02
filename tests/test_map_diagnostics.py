"""Test counterfactual isolation and background-aware diagnostic metrics."""
import unittest

import numpy as np
import torch

from tsn.evaluation.map_diagnostics import MapQuality, episode_bootstrap, oracle_variants, replace_groups
from tsn.features.maps import MapSettings


class MapDiagnosticTests(unittest.TestCase):
    def test_group_interventions_preserve_other_channels_and_source(self):
        p, t = torch.zeros(2, 6, 2, 2), torch.ones(2, 6, 2, 2)
        self.assertEqual(len(oracle_variants()), 8)
        replaced = replace_groups(p, t, ['goal'])
        self.assertEqual(float(replaced[:, 3:5].sum()), 16)
        self.assertEqual(float(replaced[:, :3].sum() + replaced[:, 5:].sum()), 0)
        self.assertEqual(float(p.sum()), 0)

    def test_foreground_metric_detects_zero_prediction_despite_background(self):
        p, t = torch.zeros(1, 6, 4, 4), torch.zeros(1, 6, 4, 4)
        t[:, 3:, 1, 1] = 1
        quality = MapQuality(MapSettings(height=4, width=4), 'cpu')
        quality.update(p, t, torch.ones(1, 4, 4))
        result = quality.result()
        self.assertEqual(result['pixel_rmse']['future_action'], .25)
        self.assertEqual(result['heatmaps']['future_action']['foreground_rmse'], 1)
        self.assertEqual(result['heatmaps']['future_action']['threshold_scores']['0.1']['recall'], 0)
        self.assertEqual(result['heatmaps']['future_action']['active_frame_detection_recall'], 0)

    def test_paired_bootstrap_identity_and_improvement(self):
        base, counts, routes = np.ones(6), np.ones(6), np.array([0, 0, 1, 1, 2, 2])
        identity = episode_bootstrap(base, counts, base, routes, repetitions=100)
        self.assertEqual(identity['paired_rmse_change_95ci_rad'], [0, 0])
        improved = episode_bootstrap(base/4, counts, base, routes, repetitions=100)
        self.assertEqual(improved['paired_relative_rmse_reduction_95ci'], [.5, .5])


if __name__ == '__main__':
    unittest.main()
