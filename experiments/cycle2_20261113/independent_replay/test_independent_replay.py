"""Synthetic-only contract checks. Never discovers or loads real experiment files."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import independent_r1_replay as audit


def fixture():
    specs = audit.graph_specs()
    arrays = {'I_delta': np.zeros((8, 12, 6)), 'C_delta': np.zeros((8, 2, 12, 6)),
              'S_quality': np.zeros((8, 2, 24, 7)),
              'editable_edges': np.array([s['editable'] for s in specs], dtype=np.int64),
              'dirty': np.array([s['dirty'] for s in specs], dtype=np.int64),
              'A_masks': np.array([s['A'] for s in specs], dtype=np.int64),
              'B_masks': np.array([s['B'] for s in specs], dtype=np.int64),
              'I_cost': np.array([[9000 + e, 1200, 252, 84, .012] for e in range(8)], dtype=float),
              'C_base_cost': np.zeros((8, 2, 5)), 'C_increment_cost': np.zeros((8, 2, 6, 5)),
              'S_cost': np.zeros((8, 2, 7, 5))}
    for e in range(8):
        for ci in range(2):
            arrays['C_base_cost'][e, ci] = [4000 + ci, 600, 108, 12, .006]
            for j in range(6):
                arrays['C_increment_cost'][e, ci, j] = [100 + j, 20 + j, 24, 12, .002 + j * .0001]
            for slot in range(7):
                arrays['S_cost'][e, ci, slot] = [9000 + 10 * slot, 1000, 216, 24, .024 + slot * .001]
    return arrays


def good_first_column():
    arrays = fixture()
    arrays['C_delta'][..., 0] = 1
    arrays['S_quality'][..., 1] = 1
    return arrays


def method_row(report, condition='change', k=2, method='candidate', field='methods'):
    return next(row for row in report['summary'][field]
                if (row['condition'], row['k'], row['method']) == (condition, k, method))


class SyntheticChecks(unittest.TestCase):
    def test_graph_endpoint_rotation_and_counts(self):
        specs = audit.graph_specs()
        self.assertEqual(specs[0]['editable'], [0, 3, 4, 6, 8, 9])
        self.assertEqual(specs[0]['changed'], [1, 5, 10])
        self.assertEqual(specs[0]['dirty'], [1, 0, 1, 0, 1, 0])
        for spec in specs:
            self.assertEqual((sum(spec['A']), sum(spec['B']), sum(spec['dirty'])), (9, 6, 3))
            self.assertEqual([spec['A'][j] for j in spec['editable']], [spec['B'][j] for j in spec['editable']])
        for i in range(0, 8, 2):
            for edge in specs[i]['editable']:
                self.assertEqual(specs[i]['A'][edge] + specs[i + 1]['A'][edge], 1)

    def test_zero_credit_stays_and_complete_coverage(self):
        result = audit.replay(fixture(), with_bootstrap=False)
        self.assertEqual((len(result['records']), len(result['summary']['methods']),
                          len(result['summary']['contrasts'])), (592, 36, 34))
        self.assertTrue(all(row['action'] is None for row in result['records']))
        self.assertFalse(result['summary']['point_rule_satisfied'])

    def test_uniform_exact_inclusion_fraction_not_best_pair(self):
        result = audit.replay(good_first_column(), with_bootstrap=False)
        for k in (1, 2):
            row = method_row(result, k=k, method='uniform_pair_expectation')
            self.assertAlmostEqual(row['accuracy'], k / 6, places=14)
            self.assertLess(row['accuracy'], 1.)
            for value in row['episode_accuracy']:
                self.assertAlmostEqual(value, k / 6, places=14)

    def test_uniform_each_subset_once_and_equal_weight(self):
        records = audit.scalar_plans_and_actions(fixture())
        for k, count in [(1, 6), (2, 15)]:
            rows = [r for r in records if r['plan_key'][:4] == [0, 'change', k, 'uniform_pair_expectation']]
            self.assertEqual(len(rows), count)
            self.assertEqual(len({tuple(r['purchased']) for r in rows}), count)
            self.assertTrue(all(r['denominator'] == count for r in rows))

    def test_nochange_candidate_nograph_identical(self):
        arrays = good_first_column()
        arrays['I_delta'][:, :3, 4] = 1
        records = audit.scalar_plans_and_actions(arrays)
        for e in range(8):
            for k in (1, 2):
                rows = {r['plan_key'][3]: r for r in records if r['plan_key'][:3] == [e, 'no_change', k]
                        and r['plan_key'][4] == 0}
                for field in ('purchased', 'updated_gain', 'action'):
                    self.assertEqual(rows['candidate'][field], rows['no_graph_ablation'][field])

    def test_strict_positive_and_physical_edge_tie(self):
        arrays = fixture()
        arrays['I_delta'][:, :, 2] = 1
        arrays['I_delta'][:, :, 5] = 1
        records = audit.scalar_plans_and_actions(arrays)
        rows = [r for r in records if r['plan_key'][3] == 'source_only']
        for row in rows:
            e = row['plan_key'][0]
            self.assertEqual(row['action'], arrays['editable_edges'][e, 2])
        self.assertTrue(all(r['action'] is None for r in records if r['plan_key'][3] == 'current_stay'))

    def test_only_purchased_columns_updated(self):
        arrays = fixture()
        arrays['C_delta'][:] = 1
        for record in audit.scalar_plans_and_actions(arrays):
            for j, value in enumerate(record['updated_gain']):
                self.assertEqual(value, float(j in record['purchased_columns']))

    def test_future_service_or_cost_does_not_change_actions(self):
        a = good_first_column()
        b = copy.deepcopy(a)
        b['S_quality'] = 1 - b['S_quality']
        b['S_cost'][..., 0] += 100
        self.assertEqual(audit.scalar_plans_and_actions(a), audit.scalar_plans_and_actions(b))

    def test_add_delete_action_categories_with_complements(self):
        result = audit.replay(good_first_column(), with_bootstrap=False)
        row = method_row(result, method='full_revalidation_high_cost')
        self.assertEqual(row['expected_episode_action_counts'], {'stay': 0., 'add': 4., 'delete': 4.})

    def test_unqueried_nan_slots_accepted(self):
        arrays = fixture()
        arrays['S_quality'][..., 1:] = np.nan
        arrays['S_cost'][:, :, 1:, 4] = np.nan
        audit.replay(arrays, with_bootstrap=False)

    def test_queried_missing_score_rejected(self):
        arrays = fixture()
        arrays['S_quality'][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Chosen action'):
            audit.replay(arrays, with_bootstrap=False)

    def test_queried_missing_score_cpu_rejected(self):
        arrays = fixture()
        arrays['S_cost'][0, 0, 0, 4] = np.nan
        with self.assertRaisesRegex(ValueError, 'Chosen action'):
            audit.replay(arrays, with_bootstrap=False)

    def test_bad_edge_order_rejected(self):
        arrays = fixture()
        arrays['editable_edges'][0, [0, 1]] = arrays['editable_edges'][0, [1, 0]]
        with self.assertRaisesRegex(ValueError, 'graph mismatch'):
            audit.validate_arrays(arrays)

    def test_bad_dirty_rejected(self):
        arrays = fixture()
        arrays['dirty'][0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'graph mismatch'):
            audit.validate_arrays(arrays)

    def test_bad_mask_rejected(self):
        arrays = fixture()
        arrays['B_masks'][0, 0] = 1 - arrays['B_masks'][0, 0]
        with self.assertRaisesRegex(ValueError, 'graph mismatch'):
            audit.validate_arrays(arrays)

    def test_nonbinary_or_nonfinite_differences_rejected(self):
        for value in (.5, np.nan, np.inf):
            arrays = fixture()
            arrays['C_delta'][0, 0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, 'invalid differences'):
                audit.validate_arrays(arrays)

    def test_wrong_fixed_call_count_rejected(self):
        arrays = fixture()
        arrays['S_cost'][0, 0, 0, 2] = 2
        with self.assertRaisesRegex(ValueError, 'fixed calls'):
            audit.validate_arrays(arrays)

    def test_wrong_score_count_rejected(self):
        arrays = fixture()
        arrays['C_increment_cost'][0, 0, 0, 3] = 0
        with self.assertRaisesRegex(ValueError, 'score count'):
            audit.validate_arrays(arrays)

    def test_negative_cost_rejected(self):
        arrays = fixture()
        arrays['I_cost'][0, 4] = -.1
        with self.assertRaisesRegex(ValueError, 'CPU cost'):
            audit.validate_arrays(arrays)

    def test_uniform_expected_cost_not_enumeration_sum(self):
        arrays = good_first_column()
        result = audit.replay(arrays, with_bootstrap=False)
        for k in (1, 2):
            row = method_row(result, k=k, method='uniform_pair_expectation', field='logical_policy_costs')
            for e in range(8):
                expected_input = (9000 + e) + 4000 + k * (100 + 2.5) + 9000 + 10 * k / 6
                self.assertAlmostEqual(row['episode_actual_costs'][e][0], expected_input, places=8)
                self.assertEqual(round(row['episode_actual_costs'][e][2]), 252 + 108 + 24 * k + 216)
                self.assertEqual(round(row['episode_actual_costs'][e][3]), 84 + 12 + 12 * k + 24)

    def test_currentstay_and_source_skip_feedback_cost(self):
        result = audit.replay(fixture(), with_bootstrap=False)
        stay = method_row(result, method='current_stay', field='logical_policy_costs')
        source = method_row(result, method='source_only', field='logical_policy_costs')
        self.assertEqual(stay['episode_actual_costs'][0], [9000., 1000., 216., 24., .024])
        self.assertEqual(source['episode_actual_costs'][0][:4], [18000., 2200., 468., 108.])

    def test_main_conservative_cap_and_full_cost_reference(self):
        cap = audit.budget(2, 'candidate')
        self.assertEqual((cap['total_token_cap'], cap['score_count_cap']), (5210112, 144))
        self.assertEqual(audit.budget(2, 'full_revalidation_high_cost')['score_count_cap'], 192)
        self.assertGreater(audit.budget(2, 'full_revalidation_high_cost')['total_token_cap'], cap['total_token_cap'])

    def test_actual_tokens_exceed_cap_rejected(self):
        arrays = fixture()
        arrays['S_cost'][0, 0, 0, 0] = 9000000
        with self.assertRaisesRegex(ValueError, 'token cap exceeded'):
            audit.replay(arrays, with_bootstrap=False)

    def test_bootstrap_constant_eight_clusters(self):
        indices = np.random.default_rng(69173).integers(0, 8, size=(10000, 8))
        result = audit.bootstrap([.125] * 8, indices)
        self.assertEqual((result['mean'], result['ci95']), (.125, [.125, .125]))
        self.assertEqual(result['independent_episode_count'], 8)

    def test_bootstrap_against_algebraic_endpoint_fixture(self):
        # Only one episode has value 1. Each replicate is its multiplicity / 8.
        indices = np.random.default_rng(69173).integers(0, 8, size=(10000, 8))
        actual = audit.bootstrap([1., 0., 0., 0., 0., 0., 0., 0.], indices)
        counts = sorted(sum(int(i) == 0 for i in row) for row in indices)
        def independent_percentile(p):
            position = 9999 * p
            i = int(position)
            return (counts[i] + (counts[i + 1] - counts[i]) * (position - i)) / 8
        self.assertEqual(actual['mean'], .125)
        self.assertEqual(actual['ci95'], [independent_percentile(.025), independent_percentile(.975)])

    def test_full_bootstrap_summary_and_nochange_zero_contrast(self):
        result = audit.replay(good_first_column())
        for k in (1, 2):
            record = result['summary']['contrasts'][f'no_change/k{k}/candidate-minus-no_graph_ablation']
            self.assertEqual(record['ci95'], [0., 0.])
            self.assertEqual(record['mean'], 0.)
        self.assertEqual(result['summary']['primary'], result['summary']['contrasts'][audit.PRIMARY])

    def test_comparator_detects_material_numeric_and_key_mismatch(self):
        c = audit.Comparator()
        c.compare({'x': [0., 1.]}, {'x': [0., 1.01]})
        self.assertEqual(len(c.mismatches), 1)
        c.compare({'a': 1}, {'b': 1})
        self.assertEqual(len(c.mismatches), 2)

    def test_incomplete_receipt_stops_before_npz_access(self):
        with tempfile.TemporaryDirectory(prefix='r1-replay-synth-') as temp:
            private = Path(temp) / 'private'
            private.mkdir()
            (private / 'COMPLETE.json').write_text(json.dumps({'status': 'RUNNING'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Official scoring incomplete'):
                audit.load_bound_inputs(temp)

    def test_no_import_of_production_policy(self):
        import ast
        tree = ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or '')
        self.assertFalse(any('r1_policy' in name or 'r1_adapter' in name for name in imports))


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticChecks)
    names = [test.id().split('.')[-1] for test in suite]
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
              'tests_run': result.testsRun, 'test_names': names,
              'failures': [(case.id(), message) for case, message in result.failures],
              'errors': [(case.id(), message) for case, message in result.errors],
              'real_R1_IC_read': False, 'real_R1_S_read': False, 'new_model_calls': 0,
              'evidence_tier': 'synthetic engineering verification only; real replay not yet run',
              'source_sha256': audit.digest_bytes(Path(audit.__file__).read_bytes()),
              'tests_sha256': audit.digest_bytes(Path(__file__).read_bytes())}
    (Path(__file__).parent / 'SYNTHETIC_CHECKS.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
