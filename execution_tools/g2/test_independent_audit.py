"""Directed CPU regressions with synthetic evidence and synthetic gold only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
CORE_DIRECTORY = REPOSITORY_ROOT / "experiments/g2"
SPEC = importlib.util.spec_from_file_location("independent_g2_audit_under_test", HERE / "independent_audit.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


class IndependentArithmetic(unittest.TestCase):
    def test_source_reference_pins_equal_the_five_frozen_core_files(self):
        public_hashes = json.loads((CORE_DIRECTORY / "SOURCE_HASHES.json").read_text(encoding="utf-8"))
        for name, expected in a.PIPELINE_PINS.items():
            source = public_hashes["files"][name]
            self.assertEqual(source["runtime_sha256"], expected)
            self.assertEqual(hashlib.sha256((CORE_DIRECTORY / name).read_bytes()).hexdigest(),
                             source["repository_utf8_lf_sha256"])

    def test_decimal_parser_matches_original_for_ambiguous_truncated_and_numeric_outputs(self):
        path = REPOSITORY_ROOT / "src/graph_credit/scoring.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), a.ORIGINAL_SCORER_SHA)
        spec = importlib.util.spec_from_file_location("g2_original_scorer_synthetic_only", path)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        cases = [
            ('{"answer":"42","evidence":"x"}', "42"),
            ('{"answer":42,"evidence":"unfinished', "42"),
            ('{"answer":true}', "1"), ('{"answer":[42]}', "42"),
            ('{"answer":1e3}', "1000"), ('1e3', "1000"),
            ('Answer: 42\nFinal answer: 43', "42"),
            ('{"answer":42,"answer":43}', "43"),
            ('#### $1,200.00', "1200"), ('Final answer: -0.50', "-0.50"),
            ('42 apples', "42"), ('{"answer":Infinity}', "0"),
        ]
        for output, target in cases:
            label = {"id": "synthetic", "task": "gsm8k", "answer": target}
            with self.subTest(output=output):
                self.assertEqual(a.independent_score(output, label), official.score_item(output, label))

    def test_graph_features_and_phi_major_order_are_explicit(self):
        mask = [1, 0, 1] + [0] * 9
        h = a.graph_basis(mask, 2)
        self.assertEqual(h.shape, (79,))
        self.assertEqual(h[13], 0)
        self.assertEqual(h[14], 1)
        vector = a.psi([2, 3, 0, 0, 0, 0, 0, 0, 1], mask, 2)
        self.assertEqual(vector.shape, (711,))
        np.testing.assert_array_equal(vector[:79], 2 * h)
        np.testing.assert_array_equal(vector[79:158], 3 * h)
        self.assertEqual(a.psi([0] * 8 + [1], mask, 1).shape, (117,))

    def test_independent_cholesky_matches_weighted_normal_equation_and_penalizes_constant(self):
        x = np.asarray([[1., 0., 2.], [1., 1., 0.], [1., 2., 1.], [1., 3., 4.]])
        y, weights = np.asarray([.1, .2, .5, .8]), np.asarray([.5, .5, 1., 1.])
        path = a.solve_ridge_path(x, y, weights)
        for alpha, coefficient in path.items():
            expected = np.linalg.solve(x.T @ (weights[:, None] * x) + alpha * np.eye(3), x.T @ (weights * y))
            np.testing.assert_allclose(coefficient, expected, atol=1e-12, rtol=1e-12)
        constant = a.solve_ridge_path(np.ones((3, 1)), np.ones(3), np.ones(3), alphas=(1.,))[1.]
        self.assertAlmostEqual(constant[0], .75)

    def test_alpha_ties_use_frozen_absolute_tolerance_and_larger_alpha(self):
        self.assertEqual(a.select_alpha({.0001: 1., 1000.: 1. + .5e-12}), 1000.)
        self.assertEqual(a.select_alpha({.0001: 1., 1000.: 1. + 2e-12}), .0001)

    def test_negative_cross_seed_risk_and_paired_noise_cancellation_are_preserved(self):
        credits = [1., -1., 0., 1., -1.]
        self.assertAlmostEqual(a.cross_seed_risk(0, credits), -.2)
        first, second = .4, -.1
        difference = a.cross_seed_risk(first, credits) - a.cross_seed_risk(second, credits)
        self.assertAlmostEqual(difference, (first - np.mean(credits)) ** 2 - (second - np.mean(credits)) ** 2)

    def test_whole_question_bootstrap_matches_linear_rng_draw_order(self):
        values = np.asarray([-.1, .2, -.3, .5])
        result = a.bootstrap_question_difference(values)
        generator = np.random.default_rng(39173)
        sampled = values[generator.integers(0, 4, size=(10000, 4))].mean(axis=1)
        np.testing.assert_allclose(result["interval"], np.quantile(sampled, [.025, .975]), atol=1e-12)
        self.assertEqual(result["bootstrap_samples"], 10000)
        self.assertEqual(result["bootstrap_seed"], 39173)

    def test_quality_mean_requires_five_seeds_and_one_graph_row(self):
        mask = (0,) * 12
        scored = {("q", seed, mask): float(index % 2) for index, seed in enumerate(a.SEEDS)}
        self.assertEqual(a.quality_table(scored), {("q", mask): .4})
        scored.pop(("q", 2701, mask))
        with self.assertRaises(a.AuditError):
            a.quality_table(scored)

    def test_global_freeze_must_precede_first_train_read_on_the_same_server(self):
        global_marker, prediction, analysis = {"frozen_unix": 1.}, {"created_unix": 4.}, {"completed_unix": 7.}
        events = {"FIRST_TRAIN_TUNE_LABELS_OPENED.json": {"first_gold_opened_unix": 2., "created_unix": 3.},
                  "FIRST_HELDOUT_LABELS_OPENED.json": {"first_gold_opened_unix": 5., "created_unix": 6.}}
        a.verify_server_event_order(global_marker, prediction, analysis, events)
        for value in (2.5, float("nan"), None):
            with self.subTest(frozen_unix=value), self.assertRaises(a.AuditError):
                a.verify_server_event_order({"frozen_unix": value}, prediction, analysis, events)


class IndependentEvidenceGates(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="g2ia_", dir=HERE.parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def _minimal_waiting_plan(self):
        relative = "results/g2/synthetic/frozen_plan.json"
        a.atomic_json(self.root, relative, {"schema": "graph_credit_g2_plan_v1", "status": "PLAN_FROZEN",
                                          "labels_read": False, "run_id": "synthetic", "output_dir": "results/g2/synthetic"})
        return relative

    def test_missing_or_unfinished_terminal_proof_never_opens_a_label(self):
        relative = self._minimal_waiting_plan()
        original = Path.open
        def guarded(path, *arguments, **keywords):
            if path.name.endswith("labels.jsonl"):
                raise AssertionError("Preterminal gold was opened")
            return original(path, *arguments, **keywords)
        with mock.patch.object(Path, "open", guarded):
            missing = a.audit(self.root, relative)
            a.atomic_json(self.root, "results/g2/synthetic/G2_ANALYSIS.json", {"scientific_gate": "INCOMPLETE"})
            unfinished = a.audit(self.root, relative)
        for report in (missing, unfinished):
            self.assertEqual(report["status"], "WAITING_NOT_TERMINAL")
            self.assertFalse(report["labels_read"])

    def test_safe_paths_reject_escapes_and_existing_reparse_points(self):
        for relative in ("../escape", "/absolute", "a//b", "a/./b", "C:/escape", "a\\b"):
            with self.subTest(relative=relative), self.assertRaises(a.AuditError):
                a.safe_path(self.root, relative, exists=False)
        linked = self.root / "link"
        linked.mkdir()
        original = Path.lstat
        def lstat(path):
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400) if path == linked else original(path)
        with mock.patch.object(Path, "lstat", lstat), self.assertRaises(a.AuditError):
            a.safe_path(self.root, "link/raw.json", exists=False)

    def test_parallel_snapshot_reads_each_raw_once_and_keeps_six_worker_bound(self):
        tasks, validators, counts = {}, {}, {}
        active, maximum = 0, 0
        lock = threading.Lock()
        for index in range(24):
            relative = f"raw/{index}.json"
            tasks[relative] = a.atomic_json(self.root, relative, {"index": index})
            counts[relative] = 0
            def validator(value, relative=relative):
                nonlocal active, maximum
                with lock:
                    counts[relative] += 1
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(.005)
                with lock:
                    active -= 1
                return None
            validators[relative] = validator
        checked = a.bounded_hash_verification(self.root, tasks, validators=validators)
        self.assertEqual(checked, tasks)
        self.assertTrue(all(count == 1 for count in counts.values()))
        self.assertGreater(maximum, 1)
        self.assertLessEqual(maximum, 6)
        (self.root / "raw/0.json").write_bytes(b"changed")
        with self.assertRaises(a.AuditError):
            a.bounded_hash_verification(self.root, tasks, validators=validators)

    def test_actual_first_audit_read_follows_durable_event_and_failure_reports_access(self):
        partitions = {"a": "train", "b": "tune", "c": "test"}
        context = {"root": self.root, "base": "out", "plan_hash": "a" * 64, "global_hash": "b" * 64,
                   "prediction_hash": "c" * 64, "analysis_hash": "d" * 64, "local_hash": "e" * 64,
                   "verified_hashes": {}, "plan": {"partitions": partitions, "label_release": {}}, "label_hashes": {}}
        for partition, qid in (("train", "a"), ("tune", "b"), ("test", "c")):
            relative = f"data/{partition}_labels.jsonl"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = (json.dumps({"id": qid, "answer": "4", "task": "gsm8k"}) + "\n").encode()
            path.write_bytes(content)
            context["plan"]["label_release"][partition] = relative
            context["label_hashes"][partition] = hashlib.sha256(content).hexdigest()
        original = Path.open
        test_case = self
        class Handle:
            def __init__(self, handle): self.handle = handle
            def __enter__(self): return self
            def __exit__(self, *arguments): return self.handle.__exit__(*arguments)
            def __getattr__(self, name): return getattr(self.handle, name)
            def read(self, *arguments):
                test_case.assertTrue((self_root / "out/AUDIT_FIRST_G2_LABELS_OPENED.json").exists())
                return self.handle.read(*arguments)
        self_root = self.root
        def guarded(path, *arguments, **keywords):
            handle = original(path, *arguments, **keywords)
            return Handle(handle) if path.name.endswith("labels.jsonl") else handle
        with mock.patch.object(Path, "open", guarded), self.assertRaises(a.AuditError):
            # One-row fixtures intentionally fail the real 200/80/120 coverage
            # after the first authorized read, proving truthful read tracking.
            a.open_audit_labels(context)
        self.assertTrue(context["audit_labels_read"])
        original_event = (self.root / "out/AUDIT_FIRST_G2_LABELS_OPENED.json").read_bytes()
        foreign = copy.deepcopy(context)
        foreign["plan_hash"] = "f" * 64
        with self.assertRaises(a.AuditError):
            a.open_audit_labels(foreign)
        self.assertEqual((self.root / "out/AUDIT_FIRST_G2_LABELS_OPENED.json").read_bytes(), original_event)

    def test_unit_mirror_receipt_binds_actual_claim_done_and_only_planned_members(self):
        qid, seed, worker, run = "q", 2701, "gpu0", "synthetic"
        uid, base = a.unit_id(qid, seed), "results/g2/synthetic"
        unit_relative = base + "/segments/" + worker + "/units/" + uid
        masks = [[0] * 12, [1] + [0] * 11]
        unit = {"question_id": qid, "seed": seed, "mask_requests": [{"mask": mask} for mask in masks]}
        references, global_rows, members = [], [], {}
        for index, mask in enumerate(masks):
            mid, mask_hash = "".join(map(str, mask)), a.digest(mask)
            raw_hash, execution_hash = str(index + 1) * 64, str(index + 3) * 64
            references.append({"mask_sha256": mask_hash,
                               "raw_path": f"units/{uid}/raw/{mid}.json", "raw_sha256": raw_hash,
                               "execution_path": f"units/{uid}/executions/{mid}.json", "execution_sha256": execution_hash})
            global_rows.append({"question_id": qid, "seed": seed, "worker_id": worker, "unit_id": uid,
                                "mask_id": mid, "mask_sha256": mask_hash,
                                "raw_path": unit_relative + f"/raw/{mid}.json", "raw_sha256": raw_hash,
                                "execution_path": unit_relative + f"/executions/{mid}.json", "execution_sha256": execution_hash})
            members[f"raw/{mid}.json"] = {"sha256": raw_hash, "bytes": 10}
            members[f"executions/{mid}.json"] = {"sha256": execution_hash, "bytes": 20}
        marker = {"schema": "g2_collection_unit_outputs_v1", "question_id": qid, "seed": seed,
                  "plan_sha256": "a" * 64, "protocol_hash": "b" * 64, "labels_read": False,
                  "first_gold_opened_unix": None, "observations": 2, "rows": references}
        marker_hash = a.atomic_json(self.root, unit_relative + "/UNIT_OUTPUTS_FROZEN.json", marker)
        unit_plan_hash = a.atomic_json(self.root, unit_relative + "/UNIT_PLAN.json", {})
        for name, expected in (("UNIT_PLAN.json", unit_plan_hash), ("UNIT_OUTPUTS_FROZEN.json", marker_hash)):
            members[name] = {"sha256": expected, "bytes": (self.root / unit_relative / name).stat().st_size}
        shared = {"question_id": qid, "seed": seed, "worker_id": worker, "plan_sha256": "a" * 64, "labels_read": False}
        claim_relative = base + "/claims/" + uid + ".json"
        claim_hash = a.atomic_json(self.root, claim_relative, {"schema": "g2_collection_claim_v1", **shared})
        done = {"schema": "g2_collection_done_v1", "unit_id": uid, "claim_sha256": claim_hash,
                "unit_frozen_sha256": marker_hash, "unit_frozen_path": f"segments/{worker}/units/{uid}/UNIT_OUTPUTS_FROZEN.json",
                "observations": 2, **shared}
        done_relative = base + "/claims/" + uid + ".done.json"
        done_hash = a.atomic_json(self.root, done_relative, done)
        receipt = {"schema": "graph_credit_g2_mirrored_unit_v1", "run_id": run, "unit_id": uid,
                   "question_id": qid, "seed": seed, "worker_id": worker, "plan_sha256": "a" * 64,
                   "done_sha256": done_hash, "claim_sha256": claim_hash, "unit_frozen_sha256": marker_hash,
                   "unit_relative_path": unit_relative, "members": members,
                   "labels_read": False, "final_full_verification_required": True}
        receipt_relative = "artifacts/private/g2/" + run + "/mirrored_units/" + uid + ".json"
        a.atomic_json(self.root, receipt_relative, receipt)
        context = {"root": self.root, "base": base, "plan_hash": "a" * 64,
                   "plan": {"run_id": run, "protocol_hash": "b" * 64, "qualification": {"worker_ids": [worker]}},
                   "by_unit": {uid: global_rows}}
        snapshot = a._unit_snapshot(context, uid, unit)
        self.assertEqual(len(snapshot["validators"]), 4)
        for fault in ("claim", "member", "path"):
            modified = copy.deepcopy(receipt)
            if fault == "claim": modified["claim_sha256"] = "0" * 64
            elif fault == "member": modified["members"]["leftover.tmp"] = {"sha256": "0" * 64, "bytes": 0}
            else: modified["unit_relative_path"] = "../escape"
            a.atomic_json(self.root, receipt_relative, modified)
            with self.subTest(fault=fault), self.assertRaises(a.AuditError):
                a._unit_snapshot(context, uid, unit)

    def test_costs_preserve_unknown_failed_cpu_and_feature_preparation_costs(self):
        ledger_relative = "out/physical_calls.jsonl"
        path = self.root / ledger_relative
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"call_id": "failed", "status": "FAILED", "input_tokens": 7,
                                    "output_tokens": None, "gpu_seconds": None, "elapsed_seconds": .5}) + "\n")
        a.atomic_json(self.root, "features.json", {"created_unix": 29.376149892807007})
        a.atomic_json(self.root, "artifacts/audit/G2_FEATURE_PREPARATION_LAUNCH.json", {"launch": {"launched_unix": 10.}})
        formal_metrics = {"input_tokens": {"known_sum": 7., "unknown_calls": 0, "complete_total": 7.},
                          "output_tokens": {"known_sum": 0., "unknown_calls": 1, "complete_total": None},
                          "gpu_seconds": {"known_sum": 0., "unknown_calls": 1, "complete_total": None},
                          "elapsed_seconds": {"known_sum": .5, "unknown_calls": 0, "complete_total": .5}}
        zero_metrics = {name: {"known_sum": 0., "unknown_calls": 0, "complete_total": 0.} for name in formal_metrics}
        a.atomic_json(self.root, "out/G2_COST_SUMMARY.json", {"generation": {
            "formal": {"calls": 1, "successful_calls": 0, "failed_calls": 1, "metrics": formal_metrics},
            "qualification": {"calls": 0, "successful_calls": 0, "failed_calls": 0, "metrics": zero_metrics}}})
        a.atomic_json(self.root, "out/cpu_attempts/interrupted.json", {"plan_sha256": "a" * 64,
                      "mode": "predict", "state": "RUNNING", "cpu_seconds": None})
        context = {"root": self.root, "base": "out", "plan_hash": "a" * 64,
                   "plan": {"formal_call_upper_bound": 124000, "features": {"receipt": "features.json"}},
                   "execution_summary": {"fresh_ids": set(), "referenced_ids": set()},
                   "cost_audit": {"formal_calls": 1, "referenced_calls": 0, "orphan_call_ids": ["failed"],
                                  "failed_call_ids": ["failed"], "physical_ledgers": [{"path": ledger_relative}],
                                  "qualification_ledgers": [], "inference_retries": 0, "unrecorded_inflight_call_upper_bound": 0}}
        result = a.cost_replay(context)
        self.assertIsNone(result["formal"]["metrics"]["output_tokens"]["complete_total"])
        self.assertEqual(result["unknown_cpu_attempts"], 1)
        self.assertIsNone(result["complete_cpu_attempt_seconds"])
        self.assertTrue(result["preprocessing_cost_cpu_unknown"])
        self.assertIsNone(result["preprocessing"]["cpu_seconds"])
        self.assertAlmostEqual(result["preprocessing"]["observed_launch_to_feature_receipt_wall_interval_seconds"], 19.376149892807007)

    def test_negative_scientific_gate_can_pass_arithmetic_acceptance_without_advancing_g3(self):
        context = {"root": self.root, "plan": {"run_id": "synthetic"}, "plan_hash": "a" * 64,
                   "base": "out", "global_hash": "b" * 64, "prediction_hash": "c" * 64,
                   "analysis_hash": "d" * 64, "local_hash": "e" * 64, "access_events": {}}
        def labels(ctx):
            ctx["audit_labels_read"] = True
            return {}, {}
        with mock.patch.object(a, "terminal_metadata", return_value=context), \
                mock.patch.object(a, "verify_mirror_snapshot", return_value={"status": "PASS"}), \
                mock.patch.object(a, "open_audit_labels", side_effect=labels), \
                mock.patch.object(a, "score_replay", return_value=({}, {"status": "PASS", "mismatch_count": 0})), \
                mock.patch.object(a, "ridge_and_prediction_replay", return_value=({}, {"status": "PASS"})), \
                mock.patch.object(a, "statistics_replay", return_value={"status": "PASS", "scientific_gate": "INSUFFICIENT_PREDICTIVE_GAIN"}), \
                mock.patch.object(a, "cost_replay", return_value={"status": "PASS"}), \
                mock.patch.object(a, "check_hash", return_value="a" * 64):
            result = a.audit(self.root, "unused.json")
        self.assertEqual(result["status"], "ACCEPTED_WITH_DOCUMENTED_LIMITATIONS")
        self.assertEqual(result["scientific_gate"], "INSUFFICIENT_PREDICTIVE_GAIN")
        self.assertEqual(result["G3_contract_state"], "STOP_G3_UNDER_CURRENT_CONTRACT")
        self.assertTrue(result["scientific_completion"])
        self.assertTrue(result["g2_stage_complete"])
        self.assertFalse(result["project_complete"])


if __name__ == "__main__":
    unittest.main()
