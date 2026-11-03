"""Statistical-contract tests: noise bias, clustering, scope, and input validity."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_pilot.py"
SPEC = importlib.util.spec_from_file_location("analyze_pilot", MODULE_PATH)
stats = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stats
SPEC.loader.exec_module(stats)


PROTOCOL = "a" * 64
PLAN = "b" * 64


def synthetic_plan(n_questions=200, families=None, seeds=(11, 23, 37, 53, 71)):
    families = families or stats.GateConfig().required_families
    sources = {name: {"sha256": "f" * 64, "bytes": 100}
               for name in ("src/graph_credit/executor.py", "src/graph_credit/backend.py",
                            "src/graph_credit/scoring.py", "scripts/run_pilot.py", "scripts/analyze_pilot.py")}
    return {
        "schema": "graph_credit_pilot_plan_v1", "synthetic_unit_test_fixture": True,
        "question_ids": [f"q{q}" for q in range(n_questions)],
        "question_payload_sha256": ["c" * 64] * n_questions,
        "data_sha256": "d" * 64, "seeds": list(seeds), "edge_pairs": [[0, 1]],
        "base_masks": [{"name": f, "mask": [0] * 12} for f in families],
        "edge_order": [[a, b] for a in range(4) for b in range(4) if a != b],
        "config": {"graph": {"rounds": 2, "n_workers": 4, "message_tokens": 128}},
        "protocol_hash": PROTOCOL, "model_identity": {"synthetic": True},
        "model_manifest": [{"file": "unit_test_model", "sha256": "e" * 64}],
        "feedback_timing": "all_assigned_outputs_frozen_before_labels_opened",
        "sampling_pairing": "same_question_seed_node_across_all_graphs",
        "qualification_only": False,
        "source_manifest": sources,
        "source_manifest_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "scorer_identity": {"callable": "graph_credit.scoring:score_item", "version": "synthetic",
                            "module_sha256": "f" * 64, "module_bytes": 100},
    }


def write_plan(path, plan):
    path.write_bytes((json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observations(n_questions=3, families=("full",), seeds=(11, 23, 37, 53, 71),
                 plan_hash=PLAN, interaction=1):
    """Synthetic labelled metadata, never presented as language-model evidence."""
    rows = []
    for q in range(n_questions):
        for family in families:
            for seed in seeds:
                for cell in stats.CELLS:
                    # A pure AND response yields a unit graph interaction.
                    quality = float(cell == "11") if interaction else 0.0
                    mask = [0] * 12
                    mask[0], mask[1] = int(cell[0]), int(cell[1])
                    rows.append({
                        "question_id": f"q{q}", "task": "gsm8k", "split": "dev",
                        "context_id": family + "_" + hashlib.sha256(json.dumps([0] * 12, separators=(",", ":")).encode()).hexdigest()[:12], "context_family": family,
                        "edge_e": 0, "edge_f": 1, "seed": seed, "cell": cell,
                        "mask_e": int(cell[0]), "mask_f": int(cell[1]),
                        "quality": quality, "graph_mask": mask,
                        "protocol_hash": PROTOCOL, "frozen_plan_sha256": plan_hash,
                    })
    return rows


def config(**kwargs):
    return stats.GateConfig(bootstrap_samples=1000, **kwargs)


class InteractionEstimatorTests(unittest.TestCase):
    def test_cross_product_estimator_and_negative_estimates_are_retained(self):
        values = np.array([1.0, -1.0, 0.0, 1.0, -1.0])
        direct = np.mean([values[i] * values[j] for i in range(5) for j in range(5) if i != j])
        got = stats.interaction_statistics(values)
        self.assertAlmostEqual(got["signal_squared_unbiased"], direct)
        self.assertLess(got["signal_squared_unbiased"], 0)
        self.assertAlmostEqual(got["observed_mean_interaction_squared"],
                               got["signal_squared_unbiased"] + got["noise_variance_of_mean"])

    def test_null_generation_noise_does_not_create_positive_signal_bias(self):
        # Independent fair +/-1 seed noise, a known zero-mean null. Raw squared
        # sample means have positive bias; cross-seed products do not.
        rng = np.random.default_rng(918)
        noise = rng.choice([-1.0, 1.0], size=(30000, 5))
        signal = ((noise.sum(axis=1) ** 2) - (noise ** 2).sum(axis=1)) / 20
        self.assertLess(abs(signal.mean()), 0.01)
        self.assertGreater(np.mean(noise.mean(axis=1) ** 2), 0.18)

    def test_constant_signal_has_zero_estimated_seed_noise(self):
        got = stats.interaction_statistics([0.5] * 5)
        self.assertEqual(got["signal_squared_unbiased"], 0.25)
        self.assertEqual(got["single_seed_noise_variance"], 0)

    def test_one_seed_cannot_estimate_signal_squared(self):
        with self.assertRaises(ValueError):
            stats.interaction_statistics([1.0])


class QuestionBootstrapTests(unittest.TestCase):
    def test_stratum_sizes_and_question_weights_are_preserved(self):
        values = {("a", "q1"): 0.0, ("a", "q2"): 1.0, ("b", "q3"): 4.0}
        strata = {key: key[0] for key in values}
        result = stats.stratified_question_bootstrap(values, strata, samples=1000, seed=7, confidence=.95)
        self.assertAlmostEqual(result["estimate"], 5 / 3)
        self.assertEqual(result["strata"], {"a": 2, "b": 1})
        self.assertEqual(result["resampling_unit"], "whole_question")
        self.assertGreaterEqual(result["interval"][0], 4 / 3)
        self.assertLessEqual(result["interval"][1], 2)

    def test_additional_contexts_of_same_question_do_not_add_bootstrap_units(self):
        rows = observations(n_questions=3, families=("full", "chain"))
        for row in rows:
            if row["question_id"] == "q0":
                row["quality"] = 0
        first = stats.analyze_observations(rows, config())
        extra = []
        for row in rows:
            if row["question_id"] == "q0":
                copy = deepcopy(row)
                copy["context_id"] += ":extra"
                extra.append(copy)
        second = stats.analyze_observations(rows + extra, config())
        self.assertEqual(first["primary_signal"], second["primary_signal"])
        self.assertEqual(second["primary_signal"]["n_questions"], 3)


class GateAndIntegrityTests(unittest.TestCase):
    def test_small_high_signal_run_is_only_pilot_even_if_config_relaxes_scale(self):
        result = stats.analyze_observations(observations(), config(min_questions=1, min_seeds=2,
                                                                  require_plan_verification=False,
                                                                  required_families=("full",)))
        self.assertEqual(result["scientific_gate"]["status"], "PILOT_ONLY")
        self.assertEqual(result["primary_signal"]["interval"], [1.0, 1.0])

    def test_complete_one_seed_run_is_pilot_not_partial_execution(self):
        result = stats.analyze_observations(observations(seeds=(11,)), config())
        self.assertEqual(result["scientific_gate"]["status"], "PILOT_ONLY")
        self.assertEqual(result["support"]["incomplete_group_count"], 0)

    def test_full_metadata_gate_passes_only_after_plan_hash_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen_plan.json"
            digest = write_plan(path, synthetic_plan())
            rows = observations(n_questions=200, families=stats.GateConfig().required_families,
                                plan_hash=digest)
            result = stats.analyze_observations(rows, config(expected_protocol_hash=PROTOCOL), plan_path=path)
        self.assertEqual(result["scientific_gate"]["status"], "PASS")
        self.assertTrue(result["provenance"]["plan_hash_verified"])
        self.assertEqual(result["support"]["complete_question_count"], 200)
        self.assertEqual(result["support"]["minimum_complete_seed_count"], 5)
        self.assertEqual(len(result["support"]["by_context_edge_pair"]), 5)

    def test_true_zero_signal_is_not_passed_by_full_sample_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen_plan.json"
            digest = write_plan(path, synthetic_plan())
            rows = observations(n_questions=200, families=stats.GateConfig().required_families,
                                plan_hash=digest, interaction=0)
            result = stats.analyze_observations(rows, config(), plan_path=path)
        self.assertEqual(result["scientific_gate"]["status"], "INSUFFICIENT_SIGNAL")
        self.assertEqual(result["primary_signal"]["interval"], [0.0, 0.0])

    def test_missing_cell_or_label_is_incomplete_and_never_silently_discarded(self):
        rows = observations()
        rows[0]["quality"] = None
        result = stats.analyze_observations(rows, config())
        self.assertEqual(result["scientific_gate"]["status"], "INCOMPLETE")
        self.assertEqual(result["support"]["unscored_rows"], 1)
        self.assertEqual(result["groups"][0]["missing_or_unscored_cells"], {"11": ["00"]})

    def test_protocol_mixing_invalidates_even_small_pilot(self):
        rows = observations()
        rows[-1]["protocol_hash"] = "c" * 64
        result = stats.analyze_observations(rows, config())
        self.assertEqual(result["scientific_gate"]["status"], "INVALID")

    def test_hash_matching_empty_plan_does_not_pass_design_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen_plan.json"
            digest = write_plan(path, {})
            rows = observations(n_questions=200, families=stats.GateConfig().required_families, plan_hash=digest)
            result = stats.analyze_observations(rows, config(), plan_path=path)
        self.assertEqual(result["scientific_gate"]["status"], "INVALID")
        self.assertTrue(any("missing required design fields" in e for e in result["validation_errors"]))

    def test_dropping_whole_planned_edge_pair_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen_plan.json"
            plan = synthetic_plan()
            plan["edge_pairs"].append([2, 3])
            digest = write_plan(path, plan)
            # Every observed 2x2 is complete, but a whole planned pair was omitted.
            rows = observations(n_questions=200, families=stats.GateConfig().required_families, plan_hash=digest)
            result = stats.analyze_observations(rows, config(), plan_path=path)
        self.assertEqual(result["scientific_gate"]["status"], "INCOMPLETE")
        coverage = result["provenance"]["complete_frozen_design_coverage"]
        self.assertEqual(coverage["missing_group_count"], 1000)
        self.assertFalse(coverage["verified"])

    def test_missing_whole_seed_block_is_not_hidden_by_five_remaining_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen_plan.json"
            plan = synthetic_plan(seeds=(11, 23, 37, 53, 71, 89))
            digest = write_plan(path, plan)
            rows = observations(n_questions=200, families=stats.GateConfig().required_families, plan_hash=digest)
            result = stats.analyze_observations(rows, config(), plan_path=path)
        self.assertEqual(result["scientific_gate"]["status"], "INCOMPLETE")
        self.assertEqual(result["provenance"]["complete_frozen_design_coverage"]["missing_group_count"], 1000)

    def test_observed_constant_background_must_still_equal_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen_plan.json"
            digest = write_plan(path, synthetic_plan(n_questions=3, families=("full",)))
            rows = observations(plan_hash=digest)
            for row in rows:
                row["graph_mask"][7] = 1
            result = stats.analyze_observations(rows, config(), plan_path=path)
        self.assertEqual(result["scientific_gate"]["status"], "INVALID")
        self.assertTrue(any("actual graph mask differs from frozen plan" in e for e in result["validation_errors"]))

    def test_scorer_identity_and_source_manifest_are_bound_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen_plan.json"
            plan = synthetic_plan(n_questions=3, families=("full",))
            plan["scorer_identity"]["module_sha256"] = "0" * 64
            digest = write_plan(path, plan)
            result = stats.analyze_observations(observations(plan_hash=digest), config(), plan_path=path)
        self.assertEqual(result["scientific_gate"]["status"], "INVALID")
        self.assertTrue(any("scorer identity differs" in e for e in result["validation_errors"]))

    def test_non_intervention_edge_changes_invalidate_counterfactual(self):
        rows = observations()
        rows[0]["graph_mask"][7] = 1
        result = stats.analyze_observations(rows, config())
        self.assertEqual(result["scientific_gate"]["status"], "INVALID")
        self.assertTrue(any("non-intervention graph edges changed" in v for v in result["validation_errors"]))

    def test_identical_repeats_deduplicate_but_conflicting_same_seed_rows_fail(self):
        rows = observations()
        result = stats.analyze_observations(rows + [deepcopy(rows[0])], config())
        self.assertEqual(result["support"]["exact_duplicate_rows_ignored"], 1)
        bad = deepcopy(rows[0])
        bad["quality"] = 1 - bad["quality"]
        result = stats.analyze_observations(rows + [bad], config())
        self.assertEqual(result["scientific_gate"]["status"], "INVALID")

    def test_plan_tampering_and_quality_nan_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps({"modified": True}), encoding="utf-8")
            result = stats.analyze_observations(observations(), config(), plan_path=path)
        self.assertEqual(result["scientific_gate"]["status"], "INVALID")
        rows = observations()
        rows[0]["quality"] = float("nan")
        result = stats.analyze_observations(rows, config())
        self.assertEqual(result["scientific_gate"]["status"], "INVALID")

    def test_unknown_config_keys_do_not_silently_change_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.json"
            path.write_text('{"min_question":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                stats._load_config(path, {})


if __name__ == "__main__":
    unittest.main()
