"""Targeted CPU regressions for the frozen G2 predictor contract."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("g2_predictor_under_test", HERE / "predictor.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def fixture():
    masks, contrasts, phi, rows = {}, [], {}, {"train": [], "tune": [], "test": []}
    for partition, qids in (("train", ("train0", "train1")), ("tune", ("tune0", "tune1")),
                            ("test", ("test0", "test1"))):
        for qi, qid in enumerate(qids):
            phi[qid] = [(-.5 if qi == 0 else .5), 0, 0, 0, 0, 0, 0, 0, 1]
            unique = {}
            for anchor_id, anchor_bit in (("p025", 0), ("p075", 1)):
                for edge in range(12):
                    left, right = [anchor_bit] * 12, [anchor_bit] * 12
                    left[edge], right[edge] = 0, 1
                    identifiers = []
                    for side, mask in ((0, left), (1, right)):
                        identifier = f"{qid}_{anchor_id}_{edge}_{side}"
                        masks[identifier] = mask
                        unique.setdefault(tuple(mask), identifier)
                        identifiers.append(identifier)
                    contrasts.append({"question_id": qid, "anchor_id": anchor_id, "edge": edge,
                                      "mask0_id": identifiers[0], "mask1_id": identifiers[1]})
            for mask, identifier in unique.items():
                for seed in p.SEEDS:
                    quality = .1 + .02 * sum(mask) + .1 * mask[0] * mask[1] + .04 * phi[qid][0] * mask[2] + .01 * (seed - 2703)
                    rows[partition].append({"question_id": qid, "mask_id": identifier, "seed": seed,
                                            "quality": quality, "partition": partition})
    return rows, phi, masks, contrasts


class FeatureAndRidgeTests(unittest.TestCase):
    def test_frozen_feature_dimensions_binary_basis_and_phi_major_order(self):
        mask = [1, 0, 1] + [0] * 9
        h = p.graph_features(mask, 2)
        self.assertEqual(h.shape, (79,))
        self.assertEqual(p.graph_features(mask, 1).shape, (13,))
        self.assertEqual(h[13], 0)  # pair (0,1)
        self.assertEqual(h[14], 1)  # pair (0,2)
        phi = [2, 3, 0, 0, 0, 0, 0, 0, 1]
        vector = p.feature_vector(phi, mask, 2)
        self.assertEqual(vector.shape, (711,))
        np.testing.assert_array_equal(vector[:79], 2 * h)
        np.testing.assert_array_equal(vector[79:158], 3 * h)
        np.testing.assert_array_equal(vector[-79:], h)
        self.assertEqual(p.feature_vector(phi, mask, 1).shape, (117,))

    def test_invalid_phi_constant_and_mask_coordinates_fail(self):
        for phi in ([0] * 9, [0] * 8, [float("nan")] + [0] * 7 + [1]):
            with self.subTest(phi=phi), self.assertRaises(p.PredictorError):
                p.feature_vector(phi, [0] * 12)
        for mask in ([0] * 11, [0] * 11 + [True], [0] * 11 + [2]):
            with self.subTest(mask=mask), self.assertRaises(p.PredictorError):
                p.graph_features(mask)

    def test_per_question_weights_sum_to_one_despite_alias_design_sizes(self):
        zero, bit = (0,) * 12, (1,) + (0,) * 11
        quality = {("a", zero): {"quality": .2}, ("a", bit): {"quality": .4},
                   ("b", zero): {"quality": .8}}
        phi = {qid: [0] * 8 + [1] for qid in ("a", "b")}
        matrix, targets, weights, counts = p._training_arrays(quality, phi, 1)
        np.testing.assert_array_equal(weights, [.5, .5, 1])
        self.assertEqual(counts, {"a": 2, "b": 1})
        self.assertEqual(matrix.shape, (3, 117))

    def test_ridge_penalizes_constant_and_matches_weighted_sse_normal_equation(self):
        theta = p.ridge_path(np.ones((3, 1)), np.ones(3), np.ones(3), alphas=(1.0,))[1.0]
        self.assertAlmostEqual(theta[0], .75)
        x = np.asarray([[1, 0], [1, 2], [1, 4]], dtype=float)
        y, weights, alpha = np.asarray([.2, .4, .8]), np.asarray([.5, .5, 1]), .1
        actual = p.ridge_path(x, y, weights, alphas=(alpha,))[alpha]
        expected = np.linalg.solve(x.T @ (weights[:, None] * x) + alpha * np.eye(2), x.T @ (weights * y))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_alpha_tie_tolerance_selects_larger_alpha_only_within_absolute_tolerance(self):
        self.assertEqual(p.select_alpha({.0001: 1.0, 1000.0: 1.0 + .5e-12}), 1000)
        self.assertEqual(p.select_alpha({.0001: 1.0, 1000.0: 1.0 + 2e-12}), .0001)
        self.assertEqual(p.ALPHA_GRID, (.0001, .001, .01, .1, 1., 10., 100., 1000.))


class QualityAndPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.phi, cls.masks, cls.contrasts = fixture()
        cls.parameters = p.fit_and_select(cls.rows["train"], cls.rows["tune"], cls.phi, cls.masks, cls.contrasts)

    def test_same_mask_aliases_yield_one_mean_quality_row_with_five_seeds(self):
        source = copy.deepcopy(self.rows["train"])
        original = source[0]
        masks = dict(self.masks)
        masks["another_alias"] = masks[original["mask_id"]]
        source.append({**original, "mask_id": "another_alias"})
        quality, ignored = p.aggregate_quality(source, masks, partition="train")
        self.assertEqual(len(quality), 52)
        self.assertEqual(ignored, 1)
        key = (original["question_id"], tuple(masks[original["mask_id"]]))
        expected = np.mean([r["quality"] for r in source[:-1] if r["question_id"] == key[0]
                            and tuple(masks[r["mask_id"]]) == key[1]])
        self.assertAlmostEqual(quality[key]["quality"], expected)

    def test_conflicting_alias_missing_seed_and_partition_contamination_fail(self):
        for fault in ("conflicting", "seed", "partition"):
            rows = copy.deepcopy(self.rows["train"])
            if fault == "conflicting": rows.append({**rows[0], "quality": rows[0]["quality"] + .1})
            elif fault == "seed": rows.pop(0)
            else: rows[0]["partition"] = "test"
            with self.subTest(fault=fault), self.assertRaises(p.PredictorError):
                p.aggregate_quality(rows, self.masks, partition="train")

    def test_duplicate_incomplete_or_wrong_direction_contrasts_fail(self):
        qids = {"train0"}
        rows = [row for row in self.contrasts if row["question_id"] in qids]
        variants = [rows + [rows[0]], rows[:-1]]
        reversed_rows = copy.deepcopy(rows)
        reversed_rows[0]["mask0_id"], reversed_rows[0]["mask1_id"] = reversed_rows[0]["mask1_id"], reversed_rows[0]["mask0_id"]
        variants.append(reversed_rows)
        for value in variants:
            with self.assertRaises(p.PredictorError):
                p.normalize_contrasts(value, self.masks, question_ids=qids)

    def test_edges_cannot_silently_use_different_graph_backgrounds_under_one_anchor_name(self):
        rows = p.normalize_contrasts(self.contrasts, self.masks, question_ids={"train0"})
        rows[0]["mask0"], rows[0]["mask1"] = list(rows[0]["mask0"]), list(rows[0]["mask1"])
        edge = rows[0]["edge"]
        other = (edge + 1) % 12
        rows[0]["mask0"][other] = 1 - rows[0]["mask0"][other]
        rows[0]["mask1"][other] = 1 - rows[0]["mask1"][other]
        with self.assertRaisesRegex(p.PredictorError, "share one original anchor"):
            p.normalize_contrasts(rows, self.masks)

    def test_fit_freezes_train_only_rows_and_independently_selected_model_dimensions(self):
        value = self.parameters
        self.assertEqual(value["train_unique_graph_rows"], 52)
        self.assertEqual(value["train_question_ids"], ["train0", "train1"])
        self.assertEqual(value["tune_question_ids"], ["tune0", "tune1"])
        self.assertEqual(len(value["models"]["C1"]["coefficients"]), 117)
        self.assertEqual(len(value["models"]["C2"]["coefficients"]), 711)
        self.assertTrue(value["all_coefficients_penalized"])
        for name in ("C1", "C2"):
            self.assertEqual(len(value["models"][name]["alpha_scores"]), 8)
            self.assertIn(value["models"][name]["alpha"], p.ALPHA_GRID)

    def test_baseline_signed_train_credit_mean_has_equal_question_anchor_seed_weight(self):
        quality, unused = p.aggregate_quality(self.rows["train"], self.masks)
        contrasts = p.normalize_contrasts(self.contrasts, self.masks, question_ids={"train0", "train1"})
        targets = p._contrast_targets(quality, contrasts)
        for edge in range(12):
            indices = [i for i, row in enumerate(contrasts) if row["edge"] == edge]
            self.assertAlmostEqual(self.parameters["B0"][edge], targets[indices].mean())

    def test_heldout_prediction_uses_no_test_quality_and_does_not_refit_on_tune(self):
        contrasts = [row for row in self.contrasts if row["question_id"].startswith("test")]
        before = p.digest(self.parameters)
        with mock.patch.object(p, "aggregate_quality", side_effect=AssertionError("Prediction accessed quality")), \
             mock.patch.object(p, "ridge_path", side_effect=AssertionError("Prediction refit ridge")):
            predictions = p.predict(self.parameters, self.phi, self.masks, contrasts)
        self.assertEqual(len(predictions), 48)
        self.assertEqual(p.digest(self.parameters), before)
        self.assertTrue(all(set(row["predictions"]) == {"B0", "C1", "C2"} for row in predictions))

    def test_frozen_parameter_mutation_is_detected(self):
        value = copy.deepcopy(self.parameters)
        value["models"]["C2"]["coefficients"][0] += .1
        with self.assertRaisesRegex(p.PredictorError, "parameters changed"):
            p.predict(value, self.phi, self.masks, self.contrasts)

    def test_training_or_tune_questions_cannot_enter_heldout_prediction(self):
        for partition in ("train", "tune"):
            contrasts = [row for row in self.contrasts if row["question_id"].startswith(partition)]
            with self.subTest(partition=partition), self.assertRaisesRegex(p.PredictorError, "overlap training or tuning"):
                p.predict(self.parameters, self.phi, self.masks, contrasts)


class EvaluationGateAndRiskTests(unittest.TestCase):
    def test_missing_or_wrong_freeze_rejects_before_iterating_test_quality(self):
        class GoldTrap:
            def __iter__(self):
                raise AssertionError("Unfrozen heldout quality was consumed")
        for freeze in ({}, {"state": "PREDICTIONS_FROZEN", "predictions_sha256": "wrong"}):
            with self.assertRaises(p.PredictorError):
                p.evaluate(GoldTrap(), [], {}, prediction_freeze=freeze)

    def test_cross_seed_risk_may_be_negative_and_paired_difference_cancels_noise(self):
        rows, phi, masks, contrasts = fixture()
        test_contrasts = [row for row in contrasts if row["question_id"].startswith("test")]
        predictions = []
        for row in p.normalize_contrasts(test_contrasts, masks):
            predictions.append({**row, "mask0": list(row["mask0"]), "mask1": list(row["mask1"]),
                                "predictions": {"B0": 0.5, "C1": .1, "C2": 0.0}})
        # All masks share the same seed-dependent question shock except edge0,
        # whose flip credit alternates +/-1. Single-model corrected risk is
        # then negative, while every model comparison shares its noise term.
        quality = []
        unique = {(row["question_id"], tuple(masks[row["mask_id"]])) for row in rows["test"]}
        for qid, mask in unique:
            for index, seed in enumerate(p.SEEDS):
                y = float(mask[0]) if index % 2 else float(1 - mask[0])
                quality.append({"question_id": qid, "graph_mask": list(mask), "seed": seed,
                                "quality": y, "partition": "test"})
        marker = {"state": "PREDICTIONS_FROZEN", "predictions_sha256": p.digest(predictions)}
        result = p.evaluate(quality, predictions, masks, prediction_freeze=marker)
        self.assertLess(result["mean_cross_seed_risk"]["C2"], 0)
        self.assertEqual(result["scientific_gate"], "G2_PASS")
        self.assertEqual(result["primary"]["bootstrap_samples"], 10000)
        self.assertEqual(result["primary"]["bootstrap_seed"], 39173)
        self.assertEqual(result["primary"]["question_count"], 2)

    def test_negative_paired_gain_is_preserved_and_fails_scientific_gate(self):
        rows, phi, masks, contrasts = fixture()
        predictions = []
        for row in p.normalize_contrasts([c for c in contrasts if c["question_id"].startswith("test")], masks):
            predictions.append({**row, "mask0": list(row["mask0"]), "mask1": list(row["mask1"]),
                                "predictions": {"B0": 0.0, "C1": 0.0, "C2": 3.0}})
        marker = {"state": "PREDICTIONS_FROZEN", "predictions_sha256": p.digest(predictions)}
        result = p.evaluate(rows["test"], predictions, masks, prediction_freeze=marker)
        self.assertLess(result["primary"]["estimate"], 0)
        self.assertEqual(result["scientific_gate"], "INSUFFICIENT_PREDICTIVE_GAIN")
        self.assertGreater(max(row["predictions"]["C2"] for row in predictions), 1)


if __name__ == "__main__":
    unittest.main()
