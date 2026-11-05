"""Synthetic-only regressions for pooled descriptive uncertainty diagnostics."""
import copy
import unittest

import numpy as np

from decision_uncertainty import uncertainty_diagnostics


def fixture(question_count=1):
    predictions = np.ones((question_count, 2, 12, 3), dtype=float)
    targets = np.ones((question_count, 2, 12, 5), dtype=float)
    leverage = np.ones((question_count, 2, 12, 2), dtype=float)
    return predictions, targets, leverage


class DecisionUncertaintyTests(unittest.TestCase):
    def test_model_order_leverage_summary_and_known_perfect_association(self):
        predictions, targets, leverage = fixture()
        predictions.reshape(-1, 3)[:12, 1] = -1
        leverage.reshape(-1, 2)[:12, 0] = 2
        leverage[..., 1] = 7
        result = uncertainty_diagnostics(predictions, targets, leverage)
        c1, c2 = result["models"]["C1"], result["models"]["C2"]
        self.assertEqual(c1["leverage"], {"minimum": 1., "maximum": 2., "mean": 1.5, "coordinates": 24})
        self.assertAlmostEqual(c1["leverage_absolute_observed_residual_spearman"], 1.0)
        sign = c1["observed_nonzero_mean_sign_diagnostic"]
        self.assertEqual(sign["positive_observed_sign_disagreements"], 12)
        self.assertEqual(sign["negative_observed_sign_agreements"], 12)
        self.assertEqual(sign["leverage_roc_auc"], 1.0)
        self.assertEqual(sign["negative_normalized_margin_roc_auc"], 1.0)
        self.assertEqual(c2["leverage"]["mean"], 7.)
        self.assertEqual(c2["observed_nonzero_mean_sign_diagnostic"]["positive_observed_sign_disagreements"], 0)
        self.assertFalse(result["B0_leverage_defined"])
        self.assertFalse(result["confidence_calibrated"])
        self.assertFalse(result["true_sign_error_rate_estimated"])
        self.assertFalse(result["certificate_provided"])
        self.assertFalse(result["model_selection_performed"])
        self.assertFalse(result["scientific_gate_defined"])

    def test_auc_average_ties_matches_explicit_positive_negative_pair_count(self):
        predictions, targets, leverage = fixture()
        predictions.reshape(-1, 3)[:12, 1] = -1
        scores = np.asarray([1.] * 6 + [2.] * 6 + [1.] * 6 + [3.] * 6)
        leverage[..., 0] = scores.reshape(1, 2, 12)
        positive_scores, negative_scores = scores[:12], scores[12:]
        pairs = [(left > right) + .5 * (left == right) for left in positive_scores for right in negative_scores]
        expected = sum(pairs) / len(pairs)
        self.assertEqual(expected, .375)
        result = uncertainty_diagnostics(predictions, targets, leverage)
        sign = result["models"]["C1"]["observed_nonzero_mean_sign_diagnostic"]
        self.assertEqual(sign["leverage_roc_auc"], expected)
        self.assertEqual(sign["negative_normalized_margin_roc_auc"], expected)

    def test_constant_scores_auc_is_half_with_both_classes(self):
        predictions, targets, leverage = fixture()
        predictions.reshape(-1, 3)[:12, 1] = -1
        result = uncertainty_diagnostics(predictions, targets, leverage)
        sign = result["models"]["C1"]["observed_nonzero_mean_sign_diagnostic"]
        self.assertEqual(sign["leverage_roc_auc"], .5)
        self.assertEqual(sign["negative_normalized_margin_roc_auc"], .5)

    def test_each_single_class_and_empty_subset_return_none_auc(self):
        for prediction_value in (1., -1.):
            with self.subTest(prediction_value=prediction_value):
                predictions, targets, leverage = fixture()
                predictions[..., 1:] = prediction_value
                result = uncertainty_diagnostics(predictions, targets, leverage)
                for model in result["models"].values():
                    sign = model["observed_nonzero_mean_sign_diagnostic"]
                    self.assertIsNone(sign["leverage_roc_auc"])
                    self.assertIsNone(sign["negative_normalized_margin_roc_auc"])
                    self.assertEqual(sign["coordinates"], 24)
        predictions, targets, leverage = fixture()
        targets.fill(0)
        result = uncertainty_diagnostics(predictions, targets, leverage)
        for model in result["models"].values():
            sign = model["observed_nonzero_mean_sign_diagnostic"]
            self.assertEqual(sign["coordinates"], 0)
            self.assertEqual(sign["positive_observed_sign_disagreements"], 0)
            self.assertEqual(sign["negative_observed_sign_agreements"], 0)
            self.assertEqual(sign["excluded_zero_observed_mean_coordinates"], 24)
            self.assertIsNone(sign["leverage_roc_auc"])
            self.assertIsNone(sign["negative_normalized_margin_roc_auc"])

    def test_spearman_returns_none_if_either_or_both_rank_vectors_constant(self):
        for constant_leverage, constant_residual in ((True, False), (False, True), (True, True)):
            with self.subTest(constant_leverage=constant_leverage, constant_residual=constant_residual):
                predictions, targets, leverage = fixture()
                if not constant_leverage:
                    leverage[..., 0] = np.arange(24).reshape(1, 2, 12)
                if not constant_residual:
                    predictions[..., 1] = np.arange(24).reshape(1, 2, 12)
                result = uncertainty_diagnostics(predictions, targets, leverage)
                self.assertIsNone(result["models"]["C1"]["leverage_absolute_observed_residual_spearman"])

    def test_zero_observed_mean_is_excluded_even_with_nonzero_seed_differences(self):
        predictions, targets, leverage = fixture()
        targets.reshape(-1, 5)[0] = [-1, 1, 0, 0, 0]
        targets.reshape(-1, 5)[1] = [0, 0, 0, 0, 0]
        predictions.reshape(-1, 3)[:2, 1:] = -100
        result = uncertainty_diagnostics(predictions, targets, leverage)
        for model in result["models"].values():
            sign = model["observed_nonzero_mean_sign_diagnostic"]
            self.assertEqual(sign["coordinates"], 22)
            self.assertEqual(sign["excluded_zero_observed_mean_coordinates"], 2)
            self.assertEqual(sign["positive_observed_sign_disagreements"], 0)
            self.assertEqual(sign["negative_observed_sign_agreements"], 22)

    def test_zero_prediction_is_disagreement_and_margin_uses_prediction_magnitude(self):
        predictions, targets, leverage = fixture()
        predictions.reshape(-1, 3)[:12, 1] = -.1
        predictions.reshape(-1, 3)[0, 1] = 0
        leverage[..., 0] = 0
        result = uncertainty_diagnostics(predictions, targets, leverage)
        sign = result["models"]["C1"]["observed_nonzero_mean_sign_diagnostic"]
        self.assertEqual(sign["positive_observed_sign_disagreements"], 12)
        self.assertEqual(sign["negative_observed_sign_agreements"], 12)
        self.assertEqual(sign["leverage_roc_auc"], .5)
        self.assertEqual(sign["negative_normalized_margin_roc_auc"], 1.0)
        self.assertIsNone(result["models"]["C1"]["leverage_absolute_observed_residual_spearman"])

    def test_b0_values_do_not_define_a_proxy_or_change_c1_c2_results_and_inputs_unchanged(self):
        inputs = fixture(2)
        originals = copy.deepcopy(inputs)
        baseline = uncertainty_diagnostics(*inputs)
        for value, original in zip(inputs, originals):
            np.testing.assert_array_equal(value, original)
        modified = copy.deepcopy(inputs)
        modified[0][..., 0] = -1e200
        repeated = uncertainty_diagnostics(*modified)
        self.assertEqual(baseline, repeated)
        self.assertEqual(baseline["question_count"], 2)
        self.assertEqual(baseline["coordinates"], 48)

    def test_invalid_shapes_empty_question_axis_and_negative_leverage_rejected(self):
        invalid = (
            (np.ones((1, 2, 12, 2)), np.ones((1, 2, 12, 5)), np.ones((1, 2, 12, 2))),
            (np.ones((1, 2, 12, 3)), np.ones((2, 2, 12, 5)), np.ones((1, 2, 12, 2))),
            (np.ones((1, 2, 12, 3)), np.ones((1, 2, 12, 5)), np.ones((1, 2, 12, 3))),
            (np.ones((0, 2, 12, 3)), np.ones((0, 2, 12, 5)), np.ones((0, 2, 12, 2))),
            (1., np.ones((1, 2, 12, 5)), np.ones((1, 2, 12, 2))),
        )
        for values in invalid:
            with self.subTest(shapes=[np.shape(value) for value in values]):
                with self.assertRaises(ValueError):
                    uncertainty_diagnostics(*values)
        predictions, targets, leverage = fixture()
        leverage[0, 0, 0, 0] = -.1
        with self.assertRaises(ValueError):
            uncertainty_diagnostics(predictions, targets, leverage)

    def test_nonfinite_or_nonreal_inputs_rejected(self):
        for array_index in range(3):
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(array=array_index, value=invalid):
                    values = fixture()
                    values[array_index].flat[0] = invalid
                    with self.assertRaises(ValueError):
                        uncertainty_diagnostics(*values)
        values = list(fixture())
        values[0] = values[0].astype(complex)
        with self.assertRaises(ValueError):
            uncertainty_diagnostics(*values)

    def test_finite_inputs_with_overflowing_derived_score_fail_explicitly(self):
        predictions, targets, leverage = fixture()
        predictions[..., 1] = 1e308
        leverage.fill(0)
        with self.assertRaisesRegex(ValueError, "arithmetic exceeded"):
            uncertainty_diagnostics(predictions, targets, leverage)


if __name__ == "__main__":
    unittest.main()
