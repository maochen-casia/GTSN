"""Validate reaching/collision semantics for post hoc tolerance reports."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from report_success_thresholds import summarize_records


class SuccessThresholdTests(unittest.TestCase):
    def test_later_and_same_step_collisions_have_distinct_prefix_outcomes(self):
        rows = [
            {'collision_step': 10, 'first_hit_steps': {'1': None, '3': 8, '5': 6, '10': 0}},
            {'collision_step': None, 'first_hit_steps': {'1': 2, '3': 1, '5': 0, '10': 0}},
            {'collision_step': 7, 'first_hit_steps': {'1': None, '3': 7, '5': 6, '10': 5}},
            {'collision_step': None, 'first_hit_steps': dict.fromkeys(('1', '3', '5', '10'))},
        ]
        result = summarize_records(rows)
        self.assertEqual(result['collision_rate'], .5)
        self.assertTrue(all(r['success_rate'] == .25 for r in result['whole_rollout'].values()))
        self.assertEqual(result['first_hit_stopping_estimates']['3'],
            {'successes': 2, 'success_rate': .5, 'collisions': 1, 'collision_rate': .25, 'timeouts': 1})
        self.assertEqual(result['first_hit_stopping_estimates']['5']['successes'], 3)
        self.assertEqual(result['first_hit_stopping_estimates']['5']['collisions'], 0)

    def test_initial_reach_stops_before_later_contact(self):
        result = summarize_records([{'collision_step': 1,
            'first_hit_steps': {'1': None, '3': None, '5': None, '10': 0}}])
        self.assertEqual(result['whole_rollout']['10']['successes'], 0)
        self.assertEqual(result['first_hit_stopping_estimates']['10']['successes'], 1)
        self.assertEqual(result['first_hit_stopping_estimates']['10']['collisions'], 0)

    def test_empty_group_has_no_rate(self):
        result = summarize_records([])
        self.assertIsNone(result['collision_rate'])
        self.assertIsNone(result['whole_rollout']['3']['success_rate'])


if __name__ == '__main__':
    unittest.main()
