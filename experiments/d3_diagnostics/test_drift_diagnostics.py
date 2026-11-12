"""Synthetic D3 tests only; this module never opens real experiment arrays."""

import itertools
import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    from d3_diagnostics import drift_diagnostics as d3
except ImportError:
    from d3_code import drift_diagnostics as d3


class GraphOnlyTests(unittest.TestCase):
    def test_analytic_credit_design_equals_potential_difference(self):
        rng = np.random.default_rng(311)
        bits = rng.integers(0, 2, size=(4, 2, 12))
        for order in (1, 2):
            design = d3.credit_design(bits, order)
            for e in range(12):
                absent, present = bits.copy(), bits.copy()
                absent[..., e], present[..., e] = 0, 1
                expected = d3.graph_features(present, order)-d3.graph_features(absent, order)
                np.testing.assert_array_equal(design[..., e, :], expected)
            self.assertTrue(np.all(design[..., 0] == 0))

    def test_graph_pair_credit_interaction_is_symmetric(self):
        beta = np.zeros(79)
        pair = d3.PAIRS.index((2, 7))
        beta[13+pair] = .7
        bits = np.zeros((1, 2, 12))
        bits[:, 1, 7] = 1
        pred = d3.predict_credits(beta, bits)
        self.assertEqual(pred[0, 1, 2]-pred[0, 0, 2], .7)
        bits = np.zeros((1, 2, 12)); bits[:, 1, 2] = 1
        pred = d3.predict_credits(beta, bits)
        self.assertEqual(pred[0, 1, 7]-pred[0, 0, 7], .7)

    def test_first_order_fit_equals_shrunk_B0_and_has_zero_drift(self):
        rng = np.random.default_rng(312)
        bits = rng.integers(0, 2, size=(9, 2, 12))
        tune_bits = rng.integers(0, 2, size=(4, 2, 12))
        targets = rng.integers(-1, 2, size=(9, 2, 12, 5)).astype(float)
        tune = rng.integers(-1, 2, size=(4, 2, 12, 5)).astype(float)
        model = d3.fit_selector(bits, targets, tune_bits, tune, order=1)
        audit = d3.first_order_degeneracy(model, targets)
        self.assertLess(audit['coefficient_identity_max_abs'], 1e-12)
        self.assertLess(audit['constant_coefficient_abs'], 1e-12)
        np.testing.assert_array_equal(d3.drift_scores(d3.predict_credits(model.beta, tune_bits, order=1)), 0)
        self.assertFalse(model.diagnostics['candidate_eligible'])
        self.assertEqual(len(model.alpha_trials), 8)

    def test_second_order_fit_predicts_synthetic_shared_graph_effects(self):
        rng = np.random.default_rng(313)
        beta = rng.normal(0, .1, size=79); beta[0] = 0
        train = rng.integers(0, 2, size=(14, 2, 12))
        tune = rng.integers(0, 2, size=(5, 2, 12))
        train_y = np.repeat(d3.predict_credits(beta, train)[..., None], 5, axis=-1)
        tune_y = np.repeat(d3.predict_credits(beta, tune)[..., None], 5, axis=-1)
        model = d3.fit_selector(train, train_y, tune, tune_y)
        error = d3.predict_credits(model.beta, tune)-tune_y.mean(axis=-1)
        self.assertLess(np.sqrt(np.mean(error**2)), .002)
        self.assertEqual(model.beta.shape, (79,))
        self.assertEqual(model.diagnostics['effective_credit_parameter_dimension'], 78)
        self.assertEqual(model.diagnostics['weight_sum'], 14)
        self.assertFalse(model.diagnostics['refit_after_tune'])
        self.assertTrue(model.diagnostics['candidate_eligible'])

    def test_alpha_exact_tie_uses_maximum_grid_value(self):
        bits = np.zeros((2, 2, 12)); targets = np.zeros((2, 2, 12, 5))
        model = d3.fit_selector(bits, targets, bits, targets)
        self.assertEqual(model.alpha, 1000.)
        np.testing.assert_array_equal(model.beta, 0)


class DriftEvaluationTests(unittest.TestCase):
    def test_top2_exact_ties_use_small_edge_indices(self):
        scores = np.zeros((3, 12))
        scores[0, [2, 7, 8]] = 1
        scores[1, [0, 11]] = 3
        np.testing.assert_array_equal(d3.select_top2(scores), [[2, 7], [0, 11], [0, 1]])

    def test_negative_cross_seed_estimate_is_not_clipped(self):
        targets = np.zeros((2, 2, 12, 5))
        targets[:, 1, 0, 0] = 1; targets[:, 1, 0, 1] = -1
        z = d3.squared_drift_by_edge(targets)
        np.testing.assert_allclose(z[:, 0], -.1)
        scores = np.zeros((2, 12)); scores[:, 0] = 2; scores[:, 1] = 1
        result = d3.evaluate_selector(scores, targets)
        self.assertLess(result['summary']['selected_squared_drift_mean'], 0)
        self.assertEqual(result['summary']['negative_squared_drift_estimate_coordinates'], 2)
        self.assertFalse(result['summary']['negative_estimates_clipped'])

    def test_cross_seed_drift_equals_mean_square_minus_sample_noise(self):
        rng = np.random.default_rng(314)
        targets = rng.integers(-1, 2, size=(6, 2, 12, 5)).astype(float)
        delta = targets[:, 1]-targets[:, 0]
        expected = delta.mean(axis=-1)**2-delta.var(axis=-1, ddof=1)/5
        np.testing.assert_allclose(d3.squared_drift_by_edge(targets), expected, atol=4e-16)
        # Swapping anchors changes delta's sign, not the squared-drift target.
        np.testing.assert_allclose(d3.squared_drift_by_edge(targets[:, ::-1]), expected, atol=4e-16)

    def test_uniform_top2_expectation_is_all_edge_mean(self):
        rng = np.random.default_rng(315)
        values = rng.normal(size=(4, 12))  # negative values must also work
        exhaustive = np.mean([values[:, [i, j]].mean(axis=1) for i, j in itertools.combinations(range(12), 2)], axis=0)
        np.testing.assert_allclose(exhaustive, values.mean(axis=1), atol=1e-15)

    def test_same_question_labels_cannot_change_selected_indices(self):
        rng = np.random.default_rng(316)
        scores = rng.random((3, 12))
        labels = rng.integers(-1, 2, size=(3, 2, 12, 5)).astype(float)
        a = d3.evaluate_selector(scores, labels)
        b = d3.evaluate_selector(scores, -labels[:, ::-1, ::-1])
        np.testing.assert_array_equal(a['selected_indices'], b['selected_indices'])
        self.assertEqual(a['summary']['question_count'], 3)
        self.assertEqual(a['summary']['enrichment']['question_count'], 3)
        self.assertEqual(a['summary']['enrichment']['bootstrap_unit'], 'question')
        self.assertFalse(a['summary']['same_question_targets_entered_selection'])

    def test_question_level_enrichment_and_bootstrap_repeatability(self):
        targets = np.zeros((3, 2, 12, 5))
        targets[:, 1, 0] = np.array([1, .5, .25])[:, None]
        scores = np.zeros((3, 12)); scores[:, 0] = 1
        a = d3.evaluate_selector(scores, targets)
        expected = np.array([1, .25, .0625])*(1/2-1/12)
        np.testing.assert_allclose(a['per_question']['enrichment'], expected)
        self.assertAlmostEqual(a['summary']['enrichment']['estimate'], expected.mean())
        b = d3.bootstrap_question_mean(expected)
        self.assertEqual(a['summary']['enrichment'], b)
        self.assertEqual(b['resamples'], 10000)
        self.assertFalse(b['independent_confirmation'])

    def test_no_drift_has_zero_enrichment_and_zero_interval(self):
        targets = np.ones((4, 2, 12, 5))
        result = d3.evaluate_selector(np.zeros((4, 12)), targets)
        self.assertEqual(result['summary']['enrichment']['estimate'], 0)
        self.assertEqual(result['summary']['enrichment']['interval95'], [0, 0])
        self.assertEqual(result['summary']['all_scores_exactly_tied_questions'], 4)


class SpendingGateTests(unittest.TestCase):
    @staticmethod
    def summary(estimate=.03, lower=.001, selected=.05):
        return {'question_count': 120, 'selected_squared_drift_mean': selected,
                'enrichment': {'estimate': estimate, 'interval95': [lower, .08]},
                'action_gain': -1, 'global_MSE_improvement': False}

    def gate(self, summary, **overrides):
        args = dict(integrity_ok=True, nomination_frozen=True, candidate_name='D3_graph_only_order2_R3')
        args.update(overrides)
        return d3.spending_gate(summary, **args)

    def test_pass_is_bounded_and_not_conditional_on_action_or_MSE(self):
        result = self.gate(self.summary())
        self.assertTrue(result['pass'])
        self.assertEqual(result['independent_pilot_gpu_hour_cap'], 16)
        self.assertEqual(result['formal_round_cap'], 1)
        self.assertFalse(result['independent_confirmation'])
        self.assertFalse(result['action_gain_gate_used'])
        self.assertFalse(result['global_MSE_gate_used'])

    def test_nonpositive_or_inconclusive_evidence_stops(self):
        for summary in (self.summary(estimate=0), self.summary(lower=0), self.summary(lower=-.01),
                        self.summary(selected=0), self.summary(estimate=float('nan'))):
            result = self.gate(summary)
            self.assertFalse(result['pass'])
            self.assertEqual(result['status'], 'STOP_CURRENT_CANDIDATE_ROUTE')
            self.assertEqual(result['independent_pilot_gpu_hour_cap'], 0)
            self.assertFalse(result['fallback_allowed'])

    def test_controls_or_unfrozen_provenance_cannot_pass_gate(self):
        for kwargs in ({'candidate_name': 'fixed_old_C2_R3'}, {'candidate_name': 'graph_only_order1_R3'},
                       {'integrity_ok': False}, {'nomination_frozen': False}):
            self.assertFalse(self.gate(self.summary(), **kwargs)['pass'])
        wrong = self.summary(); wrong['question_count'] = 240
        self.assertFalse(self.gate(wrong)['pass'])

    def test_malformed_arrays_fail_closed(self):
        with self.assertRaises(ValueError): d3.credit_design(np.ones((2, 2, 11)))
        with self.assertRaises(ValueError): d3.graph_features(np.full(12, .5))
        with self.assertRaises(ValueError): d3.squared_drift_by_edge(np.zeros((2, 2, 12, 4)))
        with self.assertRaises(ValueError): d3.select_top2(-np.ones((2, 12)))
        with self.assertRaises(ValueError): d3.drift_scores(np.full((2, 2, 12), np.nan))

    def test_negative_point_is_not_automatically_reverse_evidence(self):
        uncertain = self.summary(estimate=-.01, lower=-.05)
        result = self.gate(uncertain)
        self.assertEqual(result['evidence_classification'], 'INCONCLUSIVE')
        self.assertEqual(result['point_direction'], 'negative')
        reverse = self.summary(estimate=-.03, lower=-.05)
        reverse['enrichment']['interval95'][1] = -.001
        self.assertEqual(self.gate(reverse)['evidence_classification'], 'REVERSE_DEVELOPMENT_SIGNAL')

    def test_invalid_interval_is_not_scientific_negative_evidence(self):
        summary = self.summary()
        summary['enrichment']['interval95'] = [.02, -.01]
        result = self.gate(summary)
        self.assertFalse(result['pass'])
        self.assertEqual(result['evidence_classification'], 'INVALID_OR_UNVERIFIED')


if __name__ == '__main__':
    unittest.main(verbosity=2)
