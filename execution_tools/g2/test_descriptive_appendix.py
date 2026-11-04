"""Synthetic-only appendix formula, selection and label-access regressions."""
from contextlib import contextmanager
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments/g2"))
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("g2_descriptive_under_test", HERE / "descriptive_appendix.py")
appendix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appendix)
import predictor


def encoded(value):
    return (appendix.canonical(value) + "\n").encode()


class MemoryWorkspace:
    def __init__(self):
        self.files, self.label_reads = {}, 0
        self.label_name, self.event_name = "data/heldout.jsonl", ""

    def path(self, name):
        workspace = self
        class MemoryPath:
            def exists(self):
                return name in workspace.files
            def read_bytes(self):
                return workspace.read(name)
            def open(self, mode):
                if mode != "rb":
                    raise AssertionError("Input file opened with write permission")
                class Handle(io.BytesIO):
                    def fileno(self):
                        return 71
                    def read(self, *args):
                        if name == workspace.label_name:
                            if workspace.event_name not in workspace.files:
                                raise AssertionError("Heldout data read before its persistent first-access event")
                            workspace.label_reads += 1
                        return super().read(*args)
                return Handle(workspace.read(name))
        return MemoryPath()

    def read(self, name):
        if name not in self.files:
            raise FileNotFoundError(name)
        return self.files[name]

    def json(self, name):
        return appendix.decode(self.read(name))

    def write(self, name, value, *, exclusive=False):
        raw = encoded(value)
        if exclusive and name in self.files and self.files[name] != raw:
            raise appendix.IntegrityError("Immutable memory artifact changed")
        self.files[name] = raw
        return appendix.bytes_sha(raw)

    @contextmanager
    def lease(self, name):
        yield


def coordinates_fixture(question_count=4, *, all_zero=False, rare=False):
    rows = []
    for question in range(question_count):
        for family in ("high", "low"):
            for edge in range(12):
                targets = ([-1, 1, 0, 0, 0] if question == 0 and edge == 0 else
                           [1] * 5 if question == 2 else [0] * 5)
                if all_zero or rare:
                    targets = [0] * 5
                    if rare and question == 0 and family == "high" and edge == 0:
                        targets = [-1, 1, 0, 0, 0]
                predictions = ({"ZERO": 0.0, "B0": .2, "C1": .1, "C2": 0.0} if question == 0 else
                               {"ZERO": 0.0, "B0": .1, "C1": .2, "C2": .3} if question == 1 else
                               {"ZERO": 0.0, "B0": 1.0, "C1": .8, "C2": 1.0})
                mean = math.fsum(targets) / 5
                risks = {}
                for name, prediction in predictions.items():
                    errors = np.asarray(targets, dtype=float) - prediction
                    risks[name] = float((errors.sum() ** 2 - errors @ errors) / 20)
                rows.append({"qid": "synthetic_q_" + str(question), "family": family, "edge": edge,
                             "targets": targets, "risk": risks,
                             "observed_squared_error": {name: (prediction - mean) ** 2 for name, prediction in predictions.items()}})
    analysis = {"question_risks": {}, "scientific_gate": "INSUFFICIENT_PREDICTIVE_GAIN"}
    for qid in sorted({row["qid"] for row in rows}):
        chosen = [row for row in rows if row["qid"] == qid]
        analysis["question_risks"][qid] = {name: math.fsum(row["risk"][name] for row in chosen) / 24 for name in ("B0", "C1", "C2")}
    analysis["mean_cross_seed_risk"] = {name: math.fsum(row[name] for row in analysis["question_risks"].values()) / question_count for name in ("B0", "C1", "C2")}
    for key, reference in (("primary", "B0"), ("first_order_diagnostic", "C1")):
        values = [risks[reference] - risks["C2"] for risks in analysis["question_risks"].values()]
        analysis[key] = predictor.question_bootstrap(values)
    return rows, analysis


def accepted_fixture():
    workspace = MemoryWorkspace()
    plan = {"run_id": "synthetic", "output_dir": "results/g2/synthetic", "protocol_hash": "protocol",
            "features": {"sha256": "features", "receipt_sha256": "feature_receipt"},
            "source_manifest": {}, "pipeline_source_manifest": {}, "label_release": {"test": workspace.label_name},
            "partitions": {"synthetic_q_" + str(index): "test" for index in range(120)}}
    context = {"plan": plan, "plan_sha256": "plan", "plan_path": plan["output_dir"] + "/frozen_plan.json",
               "base": plan["output_dir"] + "/descriptive_appendix", "source_manifest_sha256": appendix.digest({}),
               "pipeline_source_manifest_sha256": appendix.digest({}), "scorer_path": "scoring.py", "scorer_sha256": "scorer",
               "families": ["high", "low"]}
    workspace.event_name = context["base"] + "/DESCRIPTIVE_FIRST_LABEL_ACCESS.json"
    labels_raw = b"".join(encoded({"id": qid, "answer": "1"}) for qid in plan["partitions"])
    workspace.files[workspace.label_name] = labels_raw
    base = plan["output_dir"]
    workspace.write(context["base"] + "/DESCRIPTIVE_APPENDIX_PLAN.json", {**appendix.recipe_value(context), "created_unix": 1700000000.0})
    # The gate tests isolate artifact binding using synthetic metadata, without
    # pretending this skeleton is a scientifically accepted generated dataset.
    workspace.files[context["plan_path"]] = b"synthetic plan bytes"
    common = {"plan_sha256": "plan", "source_manifest_sha256": appendix.digest({}), "pipeline_source_manifest_sha256": appendix.digest({})}
    observations = b""
    workspace.files[base + "/observations.jsonl"] = observations
    global_marker = {"schema": "graph_credit_g2_outputs_frozen_v1", "state": "GLOBAL_OUTPUTS_FROZEN", **common,
                     "observations": 51970, "completed_units": 2000, "labels_read": False, "first_gold_opened_unix": None,
                     "features_sha256": "features", "features_receipt_sha256": "feature_receipt",
                     "artifacts": {"observations": {"path": base + "/observations.jsonl", "sha256": appendix.bytes_sha(observations), "bytes": 0}}}
    global_hash = workspace.write(base + "/GLOBAL_OUTPUTS_FROZEN.json", global_marker)
    prediction_raw, parameter_raw = encoded([]), encoded({})
    workspace.files[base + "/heldout_predictions.json"] = prediction_raw
    workspace.files[base + "/parameters.json"] = parameter_raw
    prediction = {"schema": "graph_credit_g2_predictions_frozen_v1", "state": "PREDICTIONS_FROZEN", **common,
                  "global_outputs_frozen_sha256": global_hash, "feature_sha256": "features", "features_receipt_sha256": "feature_receipt",
                  "heldout_prediction_rows": 2880, "heldout_labels_read": False, "predictions_sha256": appendix.bytes_sha(prediction_raw),
                  "predictions_semantic_sha256": appendix.digest([]), "parameters_sha256": appendix.bytes_sha(parameter_raw)}
    prediction_hash = workspace.write(base + "/PREDICTIONS_FROZEN.json", prediction)
    release = {"state": "HELDOUT_LABELS_RELEASED", "plan_sha256": "plan", "predictions_frozen_sha256": prediction_hash,
               "heldout_labels_sha256": appendix.bytes_sha(labels_raw)}
    release_hash = workspace.write(base + "/HELDOUT_LABEL_RELEASE.json", release)
    analysis = {"schema": "graph_credit_g2_heldout_evaluation_v1", **common, "predictions_frozen_sha256": prediction_hash,
                "scientific_gate": "INSUFFICIENT_PREDICTIVE_GAIN", "heldout_label_release_sha256": release_hash,
                "heldout_labels_sha256": appendix.bytes_sha(labels_raw)}
    analysis["analysis_semantic_sha256"] = appendix.digest(analysis)
    analysis_hash = workspace.write(base + "/G2_ANALYSIS.json", analysis)
    local_hash = workspace.write(base + "/LOCAL_MIRROR_COMPLETE.json", {})
    acceptance = {"schema": "graph_credit_g2_independent_acceptance_v1", "status": "ACCEPTED_WITH_DOCUMENTED_LIMITATIONS",
                  "g2_stage_complete": True, "scientific_completion": True, "project_complete": False,
                  "run_id": "synthetic", "plan_sha256": "plan", "scientific_gate": analysis["scientific_gate"],
                  "artifact_sha256": {"global_outputs_frozen": global_hash, "predictions_frozen": prediction_hash,
                                      "analysis": analysis_hash, "local_mirror_complete": local_hash},
                  "reference_source_pins": {"original_scorer": "scorer"}}
    for name in ("integrity", "score_replay", "ridge_replay", "prediction_replay", "statistics", "costs"):
        acceptance[name] = {"status": "PASS"}
    acceptance_name = "artifacts/audit/G2_INDEPENDENT_ACCEPTANCE.json"
    workspace.write(acceptance_name, acceptance)
    return workspace, context, acceptance_name


class DescriptiveAppendixTests(unittest.TestCase):
    def test_cross_seed_formula_matches_ordered_pair_identity_and_preserves_negative_zero_risk(self):
        targets = [-1, 1, 0, 0, 0]
        self.assertEqual(appendix.cross_seed_risk(targets, 0.0), -.1)
        for prediction in (0.0, .1, -.3, 1.0):
            errors = np.asarray(targets) - prediction
            expected = float((errors.sum() ** 2 - errors @ errors) / 20)
            self.assertAlmostEqual(appendix.cross_seed_risk(targets, prediction), expected, places=14)

    def test_all_24_prespecified_cells_zero_reference_and_original_primary_replay(self):
        rows, analysis = coordinates_fixture()
        original = copy.deepcopy(analysis)
        result = appendix.analyze_coordinates(rows, analysis, ["high", "low"])
        self.assertEqual(analysis, original)
        self.assertEqual(len(result["family_edge"]), 24)
        self.assertTrue(result["original_question_risks_verified"])
        self.assertTrue(all(cell["support"]["questions_with_coordinates"] == 4 and cell["support"]["coordinates"] == 4
                            for cell in result["family_edge"]))
        self.assertIn("ZERO", result["overall"]["models"])
        self.assertIn("ZERO_risk_minus_C2_risk", result["overall"]["comparisons"])
        self.assertTrue(all(cell["comparisons"]["ZERO_risk_minus_C2_risk"]["multiplicity_adjustment"] == "none" for cell in result["family_edge"]))
        self.assertGreater(result["overall"]["models"]["ZERO"]["negative_coordinate_risk_count"], 0)
        self.assertEqual(result["overall"]["support"]["seed_contrasts"], len(rows) * 5)
        self.assertEqual(result["overall"]["support"]["cross_seed_pairs"], len(rows) * 10)

    def test_whole_question_bootstrap_matches_original_frozen_predictor(self):
        values = np.asarray([-.4, .1, .2, .8])
        counts = appendix.bootstrap_counts(len(values))
        samples = counts @ values / len(values)
        reference = predictor.question_bootstrap(values)
        np.testing.assert_allclose(np.quantile(samples, [.025, .975]), reference["interval"], rtol=1e-14, atol=1e-14)

    def test_observed_strata_support_and_undefined_samples_are_retained(self):
        rows, analysis = coordinates_fixture(3, rare=True)
        result = appendix.analyze_coordinates(rows, analysis, ["high", "low"])
        selected = result["observed_credit_strata"]["any_observed_difference_nonzero"]
        self.assertEqual(selected["support"]["coordinates"], 1)
        self.assertEqual(selected["support"]["questions_with_coordinates"], 1)
        self.assertEqual(selected["support"]["per_question_coordinate_count_histogram"], [{"coordinates": 0, "questions": 2}, {"coordinates": 1, "questions": 1}])
        statistic = selected["comparisons"]["ZERO_risk_minus_C2_risk"]
        self.assertGreater(statistic["undefined_empty_stratum_samples"], 0)
        self.assertGreater(statistic["valid_bootstrap_samples"], 0)
        self.assertEqual(statistic["valid_bootstrap_samples"] + statistic["undefined_empty_stratum_samples"], 10000)
        self.assertFalse(selected["conditional_true_risk_unbiasedness_claimed"])

    def test_empty_observed_stratum_is_undefined_instead_of_zero(self):
        rows, analysis = coordinates_fixture(all_zero=True)
        result = appendix.analyze_coordinates(rows, analysis, ["high", "low"])
        empty = result["observed_credit_strata"]["any_observed_difference_nonzero"]
        self.assertEqual(empty["support"]["coordinates"], 0)
        self.assertIsNone(empty["models"]["ZERO"]["mean_cross_seed_product_risk"])
        for comparison in empty["comparisons"].values():
            self.assertIsNone(comparison["estimate"])
            self.assertIsNone(comparison["interval"])
            self.assertEqual(comparison["undefined_empty_stratum_samples"], 10000)

    def test_signed_contributions_sum_to_overall_and_publish_no_question_identifiers(self):
        rows, analysis = coordinates_fixture()
        result = appendix.analyze_coordinates(rows, analysis, ["high", "low"])
        for comparison, summary in result["signed_question_contributions"].items():
            self.assertEqual(sum(group["questions"] for group in summary["groups"].values()), 4)
            self.assertAlmostEqual(math.fsum(group["contribution_to_overall_question_mean"] for group in summary["groups"].values()),
                                   result["overall"]["comparisons"][comparison]["estimate"], places=13)
        text = appendix.canonical(result)
        self.assertNotIn("synthetic_q_", text)
        self.assertNotIn('"qid"', text)
        self.assertNotIn('"question_id"', text)
        self.assertTrue(result["negative_risks_preserved"])

    def test_changed_original_question_risk_is_rejected_before_descriptive_publication(self):
        rows, analysis = coordinates_fixture()
        analysis["question_risks"]["synthetic_q_0"]["B0"] += .01
        with self.assertRaisesRegex(appendix.IntegrityError, "question risks differ"):
            appendix.analyze_coordinates(rows, analysis, ["high", "low"])

    def test_acceptance_waiting_fail_or_foreign_plan_prevents_any_label_open(self):
        for key, value in (("status", "WAITING_NOT_TERMINAL"), ("status", "FAIL"), ("plan_sha256", "foreign"), ("scientific_completion", False)):
            with self.subTest(key=key, value=value):
                workspace, context, acceptance = accepted_fixture()
                modified = workspace.json(acceptance)
                modified[key] = value
                workspace.write(acceptance, modified)
                with self.assertRaises((appendix.WaitingError, appendix.IntegrityError)), mock.patch.object(appendix, "original_scorer") as scorer:
                    appendix.run_appendix(workspace, context, acceptance)
                self.assertEqual(workspace.label_reads, 0)
                self.assertNotIn(workspace.event_name, workspace.files)
                scorer.assert_not_called()

    def test_changed_accepted_anchor_or_source_recipe_fails_before_gold(self):
        for suffix in ("GLOBAL_OUTPUTS_FROZEN.json", "PREDICTIONS_FROZEN.json", "G2_ANALYSIS.json", "LOCAL_MIRROR_COMPLETE.json"):
            with self.subTest(suffix=suffix):
                workspace, context, acceptance = accepted_fixture()
                workspace.files[context["plan"]["output_dir"] + "/" + suffix] += b" "
                with self.assertRaises(appendix.IntegrityError):
                    appendix.run_appendix(workspace, context, acceptance)
                self.assertEqual(workspace.label_reads, 0)
        workspace, context, acceptance = accepted_fixture()
        name = context["base"] + "/DESCRIPTIVE_APPENDIX_PLAN.json"
        recipe = workspace.json(name)
        recipe["source_sha256"] = "different code"
        workspace.write(name, recipe)
        with self.assertRaises(appendix.IntegrityError):
            appendix.run_appendix(workspace, context, acceptance)
        self.assertEqual(workspace.label_reads, 0)

    def test_first_descriptive_access_is_durable_before_read_and_keeps_original_bytes(self):
        workspace, context, acceptance = accepted_fixture()
        appendix.prelabel_gate(workspace, context, acceptance)
        metadata = SimpleNamespace(st_size=len(workspace.read(workspace.label_name)), st_dev=1, st_ino=2)
        before = {name: raw for name, raw in workspace.files.items() if not name.startswith(context["base"] + "/")}
        with mock.patch.object(appendix.os, "fstat", return_value=metadata):
            first = appendix.open_labels(workspace, context)
            first_event = workspace.read(workspace.event_name)
            repeated = appendix.open_labels(workspace, context)
        self.assertEqual(len(first), 120)
        self.assertEqual(first, repeated)
        self.assertEqual(workspace.read(workspace.event_name), first_event)
        self.assertEqual(workspace.label_reads, 2)
        self.assertEqual(before, {name: raw for name, raw in workspace.files.items() if not name.startswith(context["base"] + "/")})

    def test_freeze_is_stable_and_cannot_be_created_after_any_g2_gold_event(self):
        workspace, context, acceptance = accepted_fixture()
        recipe_name = context["base"] + "/DESCRIPTIVE_APPENDIX_PLAN.json"
        first_raw = workspace.read(recipe_name)
        appendix.freeze_recipe(workspace, context)
        self.assertEqual(workspace.read(recipe_name), first_raw)
        for event in appendix.GOLD_EVENTS:
            with self.subTest(event=event):
                fresh, fresh_context, acceptance = accepted_fixture()
                del fresh.files[fresh_context["base"] + "/DESCRIPTIVE_APPENDIX_PLAN.json"]
                fresh.files[fresh_context["plan"]["output_dir"] + "/" + event] = b"{}"
                with self.assertRaises(appendix.IntegrityError):
                    appendix.freeze_recipe(fresh, fresh_context)
                self.assertEqual(fresh.label_reads, 0)

    def test_cpu_failure_receipt_records_cost_and_keeps_all_original_bytes(self):
        workspace, context, acceptance = accepted_fixture()
        original = copy.deepcopy(workspace.files)
        with mock.patch.object(appendix, "source_context", return_value=context), mock.patch.object(appendix, "run_appendix", side_effect=appendix.WaitingError("pending")):
            with self.assertRaises(appendix.WaitingError):
                appendix.execute(workspace, context["plan_path"], acceptance, "run")
        attempts = [workspace.json(path) for path in workspace.files if "/cpu_attempts/" in path]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "WAITING_FOR_INDEPENDENT_ACCEPTANCE")
        self.assertFalse(attempts[0]["labels_read"])
        self.assertGreaterEqual(attempts[0]["cpu_seconds"], 0)
        self.assertGreaterEqual(attempts[0]["wall_seconds"], 0)
        self.assertTrue(all(workspace.files[name] == raw for name, raw in original.items()))


if __name__ == "__main__":
    unittest.main()
