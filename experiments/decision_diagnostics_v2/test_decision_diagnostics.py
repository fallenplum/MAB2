"""Synthetic regressions for signed-credit direction and one-toggle decisions."""
import copy
import json
import unittest

import numpy as np

import decision_diagnostics as diagnostic


def arrays(questions=2):
    return np.zeros((questions, 2, 12, 3)), np.zeros((questions, 2, 12, 5)), np.zeros((questions, 2, 12), dtype=int)


class DecisionDiagnosticsTests(unittest.TestCase):
    def test_existing_edge_uses_negative_credit_as_positive_deletion_gain(self):
        predictions, observed, bits = arrays()
        bits[:, :, 0] = 1
        predictions[:, :, 0, :] = -.8
        predictions[:, :, 1, :] = .6
        observed[:, :, 0, :] = -.5
        observed[:, :, 1, :] = .2
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        for model in diagnostic.MODELS:
            policy = result["policies"][model]
            self.assertEqual(policy["mean_observed_gain"], .5)
            self.assertEqual(policy["selection_support"]["deletion_count"], 4)
            self.assertEqual(policy["selection_support"]["addition_count"], 0)
            self.assertEqual(policy["selection_support"]["action_counts_edge0_through_edge11_then_stay"][0], 4)

    def test_all_negative_predicted_toggle_gains_stay_even_if_observed_edge_wins(self):
        predictions, observed, bits = arrays()
        predictions[:] = -.1
        observed[:, :, 0, :] = 1
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        for model in diagnostic.MODELS:
            self.assertEqual(result["policies"][model]["mean_observed_gain"], 0)
            self.assertEqual(result["policies"][model]["selection_support"]["stay_count"], 4)
        self.assertEqual(result["observed_oracle"]["mean_observed_selected_max_gain"], 1)

    def test_zero_gain_stays_and_strictly_positive_tie_uses_lowest_edge(self):
        predictions, observed, bits = arrays()
        predictions[:, :, 3, 0] = .4
        predictions[:, :, 7, 0] = .4
        predictions[:, :, 8, 1] = np.nextafter(0.0, 1.0)
        observed[:, :, 3, :] = -.6
        observed[:, :, 7, :] = .9
        observed[:, :, 8, :] = .3
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        self.assertEqual(result["policies"]["B0"]["selection_support"]["action_counts_edge0_through_edge11_then_stay"][3], 4)
        self.assertLess(result["policies"]["B0"]["mean_observed_gain"], 0)
        self.assertEqual(result["policies"]["C1"]["selection_support"]["addition_count"], 4)
        self.assertEqual(result["policies"]["C2"]["selection_support"]["stay_count"], 4)

    def test_uniform_baseline_includes_all_twelve_toggles_and_stay(self):
        predictions, observed, bits = arrays()
        observed[:, :, 0, :] = 1
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        random = result["policies"]["UNIFORM_13_ACTIONS"]
        self.assertAlmostEqual(random["mean_observed_gain"], 1 / 13)
        self.assertNotAlmostEqual(random["mean_observed_gain"], 1 / 12)
        self.assertEqual(random["selection_support"]["stay_rate"], 1 / 13)
        self.assertAlmostEqual(sum(random["selection_support"]["action_counts_edge0_through_edge11_then_stay"]), 4)
        self.assertEqual(result["policies"]["ZERO_STAY"]["mean_observed_gain"], 0)
        self.assertEqual(result["observed_target"], "Q_s(mask1)-Q_s(mask0), averaged over five fixed observed seeds")

    def test_random_addition_deletion_expected_support_respects_actual_bits(self):
        predictions, observed, bits = arrays()
        bits[:, :, :3] = 1
        observed[:] = .25
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        policy = result["policies"]["UNIFORM_13_ACTIONS"]
        self.assertAlmostEqual(policy["mean_observed_gain"], (9 - 3) * .25 / 13)
        support = policy["selection_support"]
        self.assertAlmostEqual(support["deletion_rate"], 3 / 13)
        self.assertAlmostEqual(support["addition_rate"], 9 / 13)
        self.assertAlmostEqual(support["deletion_rate"] + support["addition_rate"] + support["stay_rate"], 1)

    def test_three_class_confusion_is_existence_credit_not_toggle_direction(self):
        predictions, observed, bits = arrays()
        bits[:] = 1
        observed[:, :, 0, :] = -1
        observed[:, :, 1, :] = 1
        predictions[:, :, 0, :] = -1
        predictions[:, :, 1, :] = 1
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        sign = result["sign_diagnostics"]["models"]["B0"]
        self.assertEqual(sign["confusion_counts"], [[4, 0, 0], [0, 40, 0], [0, 0, 4]])
        self.assertEqual(sign["agreement_rate"], 1)
        self.assertEqual(sign["opposite_nonzero_sign_count"], 0)

    def test_spearman_uses_average_tie_ranks_and_preserves_negative_correlations(self):
        predictions, observed, bits = arrays()
        values = np.repeat([-1., 0., 1.], 4)
        observed[:] = values[None, None, :, None]
        predictions[..., 0] = values
        predictions[..., 1] = -values
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        self.assertEqual(diagnostic._average_ranks(np.array([3., 1., 1., 2.])).tolist(), [4., 1.5, 1.5, 3.])
        self.assertAlmostEqual(result["spearman"]["models"]["B0"]["overall"]["mean_defined_spearman"], 1)
        self.assertAlmostEqual(result["spearman"]["models"]["C1"]["overall"]["mean_defined_spearman"], -1)
        self.assertEqual(result["spearman"]["models"]["C2"]["overall"]["undefined_pairs"], 4)

    def test_all_zero_spearman_is_undefined_and_confusion_keeps_full_support(self):
        predictions, observed, bits = arrays()
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        for model in diagnostic.MODELS:
            rank = result["spearman"]["models"][model]["overall"]
            self.assertIsNone(rank["mean_defined_spearman"])
            self.assertEqual(rank["undefined_pairs"], 4)
            self.assertEqual(rank["both_constant_pairs"], 4)
            self.assertEqual(result["sign_diagnostics"]["models"][model]["observed_support"]["zero"], 48)
            self.assertIsNone(result["sign_diagnostics"]["models"][model]["by_observed_sign"]["positive"]["agreement_rate"])

    def test_spearman_ranks_existence_credit_before_mask_direction(self):
        predictions, observed, bits = arrays()
        values = np.arange(-6., 6.) / 10
        predictions[:] = values[:, None]
        observed[:] = (values + .3)[None, None, :, None]
        bits[:, :, ::2] = 1
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        for model in diagnostic.MODELS:
            self.assertAlmostEqual(result["spearman"]["models"][model]["overall"]["mean_defined_spearman"], 1)

    def test_observed_oracle_averages_seeds_before_maximizing_actions(self):
        predictions, observed, bits = arrays()
        observed[:, :, 0, :] = [1, -1, 0, 0, 0]
        observed[:, :, 1, :] = .2
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        self.assertAlmostEqual(result["observed_oracle"]["mean_observed_selected_max_gain"], .2)
        self.assertEqual(result["observed_oracle"]["selection_support"]["action_counts_edge0_through_edge11_then_stay"][1], 4)

    def test_whole_question_bootstrap_averages_both_anchors_before_resampling(self):
        predictions, observed, bits = arrays(3)
        predictions[:, :, 0, :] = 1
        gains = np.array([[1., -1.], [.8, -.6], [-.2, .6]])
        observed[:, :, 0, :] = gains[..., None]
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        expected_question = gains.mean(axis=1)
        rng = np.random.default_rng(49173)
        indices = rng.integers(0, 3, size=(10000, 3))
        expected_interval = np.quantile(expected_question[indices].mean(axis=1), [.025, .975], method="linear")
        comparison = result["paired_comparisons"]["B0_minus_ZERO_STAY"]
        self.assertAlmostEqual(comparison["estimate"], expected_question.mean())
        np.testing.assert_allclose(comparison["interval"], expected_interval, rtol=0, atol=0)
        self.assertEqual(comparison["question_count"], 3)
        self.assertEqual(comparison["bootstrap_samples"], 10000)
        self.assertEqual(comparison["bootstrap_seed"], 49173)
        self.assertLessEqual(comparison["interval"][1], .2 + 1e-15)

    def test_all_eight_comparisons_keep_negative_effect_and_quality_only_scope(self):
        predictions, observed, bits = arrays()
        predictions[:, :, 0, :2] = 1
        predictions[:, :, 1, 2] = 1
        observed[:, :, 0, :] = .5
        observed[:, :, 1, :] = -.5
        result = diagnostic.decision_diagnostics(predictions, observed, bits)
        self.assertEqual(len(result["paired_comparisons"]), 8)
        self.assertEqual(result["paired_comparisons"]["C2_minus_B0"]["estimate"], -1)
        self.assertEqual(result["paired_comparisons"]["C2_minus_C1"]["interval"], [-1, -1])
        self.assertEqual(result["paired_comparisons"]["C2_minus_ZERO_STAY"]["estimate"], -.5)
        self.assertEqual(result["observed_oracle"]["mean_observed_selected_max_gain"], .5)
        self.assertTrue(result["observed_oracle"]["maximum_selection_bias_present"])
        self.assertFalse(result["observed_oracle"]["true_oracle_or_true_regret_claimed"])
        self.assertFalse(result["limitations"]["net_budget_benefit_claimed"])
        self.assertFalse(result["limitations"]["sequential_rollout_claimed"])

    def test_inputs_unchanged_json_serializable_and_fixed_rng_replay(self):
        inputs = arrays()
        before = copy.deepcopy(inputs)
        for value in inputs:
            value.flags.writeable = False
        state = np.random.get_state()
        first = diagnostic.decision_diagnostics(*inputs)
        second = diagnostic.decision_diagnostics(*inputs)
        self.assertEqual(json.dumps(first, sort_keys=True, allow_nan=False), json.dumps(second, sort_keys=True, allow_nan=False))
        for actual, expected in zip(inputs, before):
            np.testing.assert_array_equal(actual, expected)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_invalid_shape_nonfinite_values_and_nonbinary_masks_rejected(self):
        inputs = arrays()
        cases = [(inputs[0][..., :2], inputs[1], inputs[2]),
                 (inputs[0], inputs[1][..., :4], inputs[2]),
                 (inputs[0], inputs[1], inputs[2] + .5)]
        predictions, observed, bits = arrays()
        predictions[0, 0, 0, 0] = np.nan
        cases.append((predictions, observed, bits))
        predictions, observed, bits = arrays()
        observed[0, 0, 0, 0] = np.inf
        cases.append((predictions, observed, bits))
        for case in cases:
            with self.subTest(shapes=[value.shape for value in case]), self.assertRaises(ValueError):
                diagnostic.decision_diagnostics(*case)


if __name__ == "__main__":
    unittest.main()
