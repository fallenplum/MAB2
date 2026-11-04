"""Native floating-point order recovery, with unchanged frozen validators."""
import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "g2"))
sys.path.insert(0, str(HERE))

import freeze_compat as compat
import test_collector as regression

collector, native, FakeBackend = regression.collector, regression.native, regression.FakeBackend

TEMP_ROOT = Path(__file__).resolve().parent.parent


class VariedBackend(FakeBackend):
    def generate(self, messages, seed, max_new_tokens, config):
        generation = super().generate(messages, seed, max_new_tokens, config)
        aggregate = max_new_tokens == config.aggregate_tokens
        generation = dataclasses.replace(generation, elapsed_seconds=1.0 if aggregate else .1,
                                         gpu_seconds=.8 if aggregate else .01)
        self.calls[-1].update(elapsed_seconds=generation.elapsed_seconds, gpu_seconds=generation.gpu_seconds)
        return generation


def example(rounds=2):
    question = {"id": "synthetic", "question": "Synthetic closed-input arithmetic."}
    backend = VariedBackend()
    executor = native.Executor(backend, native.GraphConfig(rounds=rounds))
    result = executor.execute(question, [1] * 12, 2701)
    value = json.loads(collector.canonical(result.to_dict()))
    row = {"answer": result.answer, "service_output": result.nodes["aggregate"].generation.text,
           "service_message_metadata": result.nodes["aggregate"].message_metadata,
           "physical_cost": result.cost(), "logical_service_cost": result.cost(False), "execution_id": result.execution_id}
    context = {"module": native, "config": executor.config, "questions": {question["id"]: question},
               "plan": {"protocol_hash": executor.protocol_hash, "model": {"identity": backend.identity}}}
    unit = {"question_id": question["id"], "seed": 2701}
    item = {"mask": [1] * 12, "reference_mask": None}
    return backend, executor, result, value, row, context, unit, item


class FreezeCompatibilityTests(unittest.TestCase):
    def prepared_recovery(self, temporary):
        fixture = regression.CollectorRegressions()
        fixture.setUp()
        with mock.patch("test_collector.FakeBackend", VariedBackend):
            context, rows, segment = fixture.prepared_output(temporary)
        context["collector"] = collector
        output = context["output"]
        collector.atomic_json(output / "STATUS.json", {"phase": "FAILED", "error": "original failure"})
        collector.atomic_json(output / "STOP_REQUESTED.json", {"schema": "graph_credit_g2_stop_v1", "run_id": "g2_test",
                              "plan_sha256": "plan", "reason": "Collector global output freeze failed", "labels_read": False})
        binding = {"run_id": "g2_test", "plan_sha256": "plan", "source_manifest_sha256": "source",
                   "pipeline_source_manifest_sha256": "pipeline", "protocol_hash": context["plan"]["protocol_hash"],
                   "wrapper_sha256": collector.file_digest(Path(compat.__file__)), "labels_read": False,
                   "failure_evidence": {"failure_status": collector.artifact(context["root"], output / "STATUS.json"),
                                        "stop_request": collector.artifact(context["root"], output / "STOP_REQUESTED.json")}}
        return context, rows, segment, binding

    def test_sorted_json_reproduces_exact_failure_and_native_order_recovers(self):
        backend, executor, result, value, row, context, unit, item = example()
        old = collector.execution_from_dict(native, value)
        self.assertEqual(next(iter(old.nodes)), "aggregate")
        self.assertNotEqual(old.cost(), row["physical_cost"])
        ledger = {call["call_id"]: call for call in backend.calls}
        with self.assertRaisesRegex(ValueError, "Raw answer or physical/logical cost"):
            collector.validate_execution(context, unit, item, row, old, {}, ledger, set())
        restored, changed = compat.compare_roundtrip(collector, native, value, row, 2)
        self.assertTrue(changed)
        self.assertEqual(restored.to_dict(), result.to_dict())
        consumed = set()
        collector.validate_execution(context, unit, item, row, restored, {}, ledger, consumed)
        self.assertEqual(consumed, set(ledger))

    def test_two_and_three_rounds_preserve_all_payload_fields_and_original_object(self):
        for rounds in (2, 3):
            with self.subTest(rounds=rounds):
                backend, executor, result, value, row, context, unit, item = example(rounds)
                before = copy.deepcopy(value)
                original = collector.execution_from_dict(native, value)
                restored = compat.ordered_deserializer(lambda module, _: original, native, value, rounds)
                self.assertEqual(list(restored.nodes), list(result.nodes))
                self.assertEqual(value, before)
                for name in result.nodes:
                    self.assertEqual(restored.nodes[name].integrity_value(), result.nodes[name].integrity_value())
                    self.assertEqual(restored.nodes[name].output_digest, result.nodes[name].output_digest)

    def test_extra_or_missing_node_rejected_before_original_deserializer(self):
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                value = example()[3]
                if mutation == "missing":
                    del value["nodes"]["r1:w0"]
                else:
                    value["nodes"]["unrequested"] = copy.deepcopy(value["nodes"]["aggregate"])
                original = mock.Mock()
                with self.assertRaisesRegex(ValueError, "missing or extra"):
                    compat.ordered_deserializer(original, native, value, 2)
                original.assert_not_called()

    def test_answer_integer_and_float_cost_tamper_rejected_without_tolerance(self):
        for mutation in ("answer", "integer", "integer_type", "float", "stored_cost"):
            with self.subTest(mutation=mutation):
                backend, executor, result, value, row, context, unit, item = example()
                if mutation == "answer":
                    row["answer"] += "0"
                elif mutation == "integer":
                    row["physical_cost"]["calls"] += 1
                elif mutation == "integer_type":
                    row["physical_cost"]["calls"] = float(row["physical_cost"]["calls"])
                    value["physical_cost"]["calls"] = row["physical_cost"]["calls"]
                elif mutation == "float":
                    row["logical_service_cost"]["gpu_seconds"] += 1e-12
                else:
                    value["physical_cost"]["gpu_seconds"] += 1e-12
                with self.assertRaises(ValueError):
                    compat.compare_roundtrip(collector, native, value, row, 2)

    def test_single_flip_keeps_anchor_lineage_and_physical_calls_consumed_once(self):
        backend, executor, first, value, row, context, unit, item = example()
        ledger = {call["call_id"]: call for call in backend.calls}
        restored, _ = compat.compare_roundtrip(collector, native, value, row, 2)
        consumed = set()
        collector.validate_execution(context, unit, item, row, restored, {}, ledger, consumed)
        second = executor.execute(context["questions"][unit["question_id"]], [0] + [1] * 11, unit["seed"], base=first)
        value2 = json.loads(collector.canonical(second.to_dict()))
        row2 = {**row, "answer": second.answer, "service_output": second.nodes["aggregate"].generation.text,
                "service_message_metadata": second.nodes["aggregate"].message_metadata,
                "physical_cost": second.cost(), "logical_service_cost": second.cost(False), "execution_id": second.execution_id}
        recovered, _ = compat.compare_roundtrip(collector, native, value2, row2, 2)
        ledger = {call["call_id"]: call for call in backend.calls}
        collector.validate_execution(context, unit, {"mask": [0] + [1] * 11, "reference_mask": [1] * 12}, row2,
                                     recovered, {tuple(first.mask): restored}, ledger, consumed)
        self.assertEqual(len(consumed), 11)
        self.assertEqual(recovered.cost()["calls"], 2)
        self.assertEqual(recovered.to_dict(), second.to_dict())

    def test_node_time_or_fingerprint_tamper_still_fails_original_validator(self):
        for mutation in ("time", "fingerprint"):
            with self.subTest(mutation=mutation):
                backend, executor, result, value, row, context, unit, item = example()
                if mutation == "time":
                    value["nodes"]["r1:w0"]["generation"]["gpu_seconds"] += 1e-12
                else:
                    value["nodes"]["r1:w0"]["input_fingerprint"] = "foreign"
                restored = compat.ordered_deserializer(collector.execution_from_dict, native, value, 2)
                with self.assertRaises(ValueError):
                    collector.validate_execution(context, unit, item, row, restored, {},
                                                 {call["call_id"]: call for call in backend.calls}, set())

    def test_hooks_keep_original_stop_and_validator_and_restore_on_exception(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            context = {"collector": collector, "output": Path(temporary), "config": native.GraphConfig()}
            original = collector.execution_from_dict, collector.check_stop, collector.validate_execution, collector.atomic_json
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                with compat.compatibility_hooks(context, {}):
                    self.assertIs(collector.check_stop, original[1])
                    self.assertIs(collector.validate_execution, original[2])
                    raise RuntimeError("deliberate")
            self.assertEqual((collector.execution_from_dict, collector.check_stop, collector.validate_execution, collector.atomic_json), original)

    def test_new_stop_prevents_global_publication_and_receipt(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            output = Path(temporary)
            context = {"collector": collector, "output": output, "config": native.GraphConfig(), "plan": {"run_id": "test"}, "plan_hash": "plan"}
            collector.atomic_json(output / "STOP_REQUESTED.json", {"run_id": "test", "plan_sha256": "plan"})
            callback = mock.Mock()
            with compat.compatibility_hooks(context, {}, callback), self.assertRaises(collector.StopRequested):
                collector.atomic_json(output / "GLOBAL_OUTPUTS_FROZEN.json", {})
            callback.assert_not_called()
            self.assertFalse((output / "GLOBAL_OUTPUTS_FROZEN.json").exists())

    def test_complete_original_freeze_runs_after_order_recovery_with_bytes_unchanged(self):
        fixture = regression.CollectorRegressions()
        fixture.setUp()
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            with mock.patch("test_collector.FakeBackend", VariedBackend):
                context, rows, segment = fixture.prepared_output(temporary)
            context["collector"] = collector
            before = {path.relative_to(context["root"]).as_posix(): path.read_bytes()
                      for path in context["root"].rglob("*") if path.is_file()}
            with self.assertRaisesRegex(ValueError, "Raw answer or physical/logical cost"):
                collector.freeze_outputs(context)
            with mock.patch.object(compat, "bindings", return_value={}):
                with compat.compatibility_hooks(context, {}):
                    report = collector.freeze_outputs(context)
            self.assertEqual(report["observations"], 2)
            audit = collector.read_json(context["output"] / "PHYSICAL_COST_AUDIT.json")
            self.assertEqual(audit["formal_calls"], audit["referenced_calls"])
            for relative, content in before.items():
                self.assertEqual((context["root"] / relative).read_bytes(), content)

    def test_full_preflight_runs_original_validator_for_every_graph_without_new_outputs(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            context, rows, segment, binding = self.prepared_recovery(temporary)
            with mock.patch.object(compat, "bindings", return_value=binding), mock.patch.object(
                    collector, "validate_execution", wraps=collector.validate_execution) as validator:
                report = compat.diagnose(context)
            self.assertEqual(report["observations"], 2)
            self.assertEqual(validator.call_count, 2)
            self.assertFalse((context["output"] / "GLOBAL_OUTPUTS_FROZEN.json").exists())
            self.assertTrue((context["output"] / "STOP_REQUESTED.json").exists())
            preflight = collector.read_json(context["output"] / compat.RECEIPT_FOLDER / "PREFLIGHT.json")
            self.assertEqual(preflight["original_execution_validations"], 2)
            self.assertTrue(preflight["complete_original_freeze_pending"])
            self.assertTrue(set(preflight["changed_field_counts"]) <= {
                view + "." + field for view in ("physical_cost", "logical_service_cost") for field in compat.TIMING_FIELDS})

    def test_preflight_rejects_wrong_raw_source_or_question_fingerprint_even_with_self_consistent_raw_hash(self):
        for field in ("source_manifest_sha256", "question_fingerprint"):
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
                context, rows, segment, binding = self.prepared_recovery(temporary)
                rows[0][field] = "foreign"
                raw = {name: value for name, value in rows[0].items() if name not in ("raw_path", "raw_sha256")}
                rows[0]["raw_sha256"] = collector.atomic_json(segment / rows[0]["raw_path"], raw)
                (segment / "raw_executions.jsonl").write_text("".join(collector.canonical(row) + "\n" for row in rows), encoding="utf-8")
                with mock.patch.object(compat, "bindings", return_value=binding), self.assertRaises(ValueError):
                    compat.diagnose(context)
                self.assertFalse((context["output"] / compat.RECEIPT_FOLDER / "PREFLIGHT.json").exists())
                self.assertFalse((context["output"] / "GLOBAL_OUTPUTS_FROZEN.json").exists())

    def test_authorized_recovery_archives_exact_history_before_publishing_receipt_and_status(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            context, rows, segment, binding = self.prepared_recovery(temporary)
            before = {path.relative_to(context["root"]).as_posix(): path.read_bytes()
                      for path in context["root"].rglob("*") if path.is_file()}
            original_stop, original_validator = collector.check_stop, collector.validate_execution
            with mock.patch.object(compat, "bindings", return_value=binding):
                preflight = compat.diagnose(context)
                result = compat.recover_freeze(context, preflight["preflight_sha256"], preflight["stop_sha256"])
            output, folder = context["output"], context["output"] / compat.RECEIPT_FOLDER
            self.assertEqual((folder / "STATUS.failed.json").read_bytes(), before["results/STATUS.json"])
            self.assertEqual((folder / "STOP_REQUESTED.consumed.json").read_bytes(), before["results/STOP_REQUESTED.json"])
            self.assertFalse((output / "STOP_REQUESTED.json").exists())
            self.assertIs(collector.check_stop, original_stop)
            self.assertIs(collector.validate_execution, original_validator)
            success = collector.read_json(folder / "RECOVERY_COMPLETE.json")
            status = collector.read_json(output / "STATUS.json")
            self.assertEqual(status["publisher"], "freeze_compat")
            self.assertEqual(status["phase"], "WAITING_TRAIN_TUNE_LABELS")
            self.assertEqual(success["global_outputs_frozen_sha256"], collector.file_digest(output / "GLOBAL_OUTPUTS_FROZEN.json"))
            self.assertEqual(status["recovered_from"]["recovery_receipt"]["sha256"], collector.file_digest(folder / "RECOVERY_COMPLETE.json"))
            self.assertEqual(result["observations"], 2)
            for relative, content in before.items():
                if relative not in ("results/STATUS.json", "results/STOP_REQUESTED.json"):
                    self.assertEqual((context["root"] / relative).read_bytes(), content)

    def test_foreign_preflight_or_stop_ack_never_consumes_history(self):
        for wrong in ("preflight", "stop"):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
                context, rows, segment, binding = self.prepared_recovery(temporary)
                with mock.patch.object(compat, "bindings", return_value=binding):
                    preflight = compat.diagnose(context)
                    with self.assertRaises(ValueError):
                        compat.recover_freeze(context, "foreign" if wrong == "preflight" else preflight["preflight_sha256"],
                                              "foreign" if wrong == "stop" else preflight["stop_sha256"])
                self.assertTrue((context["output"] / "STOP_REQUESTED.json").exists())
                self.assertEqual(collector.read_json(context["output"] / "STATUS.json")["phase"], "FAILED")
                self.assertFalse((context["output"] / compat.RECEIPT_FOLDER / "AUTHORIZATION.json").exists())

    def test_new_stop_after_global_prevents_success_receipt_and_recovered_status(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            context, rows, segment, binding = self.prepared_recovery(temporary)
            original_freeze = collector.freeze_outputs
            def stop_after_global(actual_context):
                result = original_freeze(actual_context)
                collector.atomic_json(context["output"] / "STOP_REQUESTED.json", {"run_id": "g2_test", "plan_sha256": "plan"})
                return result
            with mock.patch.object(compat, "bindings", return_value=binding):
                preflight = compat.diagnose(context)
                with mock.patch.object(collector, "freeze_outputs", side_effect=stop_after_global):
                    with self.assertRaises(collector.StopRequested):
                        compat.recover_freeze(context, preflight["preflight_sha256"], preflight["stop_sha256"])
            self.assertFalse((context["output"] / compat.RECEIPT_FOLDER / "RECOVERY_COMPLETE.json").exists())
            self.assertEqual(collector.read_json(context["output"] / "STATUS.json")["phase"], "FAILED")


if __name__ == "__main__":
    unittest.main()
