"""Specific collection regressions with the frozen executor and fake generations."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
reference = HERE.parent / "g2_reference" / "executor.py"
if not reference.is_file():
    reference = REPO_ROOT / "src" / "graph_credit" / "executor.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collector = load("g2_collector_tested", HERE / "collector.py")
native = load("g2_executor_test_reference", reference)


class FakeBackend:
    identity = {"backend": "g2_collection_regression_only"}
    def __init__(self, fail_at=None):
        self.calls = []
        self.physical_calls = self.calls
        self.runtime = {"regression_backend": True}
        self.fail_at = fail_at
    def encode(self, text):
        return list(map(ord, text))
    def decode(self, values):
        return "".join(map(chr, values))
    def generate(self, messages, seed, max_new_tokens, config):
        status = "FAILED" if len(self.calls) + 1 == self.fail_at else "SUCCESS"
        if status == "FAILED":
            self.calls.append({"status": status, "seed": seed})
            raise RuntimeError("Deliberate model failure with a retained paid-call record")
        answer = str(int(collector.digest([messages, seed])[:8], 16) % 100)
        text = native.canonical({"answer": answer, "evidence": "Checked.", "uncertainty": ""})
        generation = native.Generation(text, len(native.canonical(messages)), len(text), .01, .008, 1024, "eos")
        self.calls.append({"status": status, "seed": seed, "call_id": generation.call_id,
                           "prompt_sha256": native.digest(messages), "input_tokens": generation.input_tokens,
                           "output_tokens": generation.output_tokens, "elapsed_seconds": generation.elapsed_seconds,
                           "gpu_seconds": generation.gpu_seconds, "gpu_peak_allocated_bytes": generation.gpu_peak_allocated_bytes,
                           "finish_reason": generation.finish_reason})
        return generation


class CollectorRegressions(unittest.TestCase):
    def setUp(self):
        self.question = {"id": "calibration-0", "question": "Compute 2 plus 3.", "task": "gsm8k", "split": "calibration"}

    def requests(self, masks):
        return [{"mask": mask, "reference_mask": masks[i - 1] if i else None,
                 "aliases": [{"request_index": i}]} for i, mask in enumerate(masks)]

    def test_repeated_masks_have_one_committed_graph_and_retained_comparison_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend()
            executor = native.Executor(backend)
            full, empty, one = [1] * 12, [0] * 12, [1] + [0] * 11
            segment = Path(temporary)
            rows = collector.collect_unit(executor, self.question, 1701,
                                          self.requests([full, empty, full, one, empty]), segment, "plan", "sources")
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["request_indices"] for row in rows], [[0, 2], [1, 4], [3]])
            self.assertEqual([row["mask_aliases"] for row in rows],
                             [[{"request_index": 0}, {"request_index": 2}], [{"request_index": 1}, {"request_index": 4}], [{"request_index": 3}]])
            self.assertEqual(len(backend.calls), 19)
            self.assertEqual(sum(row["physical_cost"]["calls"] for row in rows), len(backend.calls))
            for row in rows:
                self.assertIsNone(row["quality"])
                self.assertFalse(row["labels_read"])
                self.assertEqual(collector.file_digest(segment / row["raw_path"]), row["raw_sha256"])
                self.assertEqual(collector.file_digest(segment / row["execution_path"]), row["execution_sha256"])
            marker = json.loads(next(segment.glob("units/*/UNIT_OUTPUTS_FROZEN.json")).read_text())
            self.assertEqual(marker["observations"], 3)
            self.assertIsNone(marker["first_gold_opened_unix"])

    def test_midunit_model_failure_preserves_committed_graph_and_ledger_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            segment = Path(temporary)
            backend = FakeBackend(fail_at=10)
            executor = native.Executor(backend)
            with self.assertRaisesRegex(RuntimeError, "Deliberate model failure"):
                collector.collect_unit(executor, self.question, 1701, self.requests([[1] * 12, [0] * 12]), segment, "plan", "sources")
            self.assertEqual(len(backend.calls), 10)
            self.assertEqual(backend.calls[-1]["status"], "FAILED")
            rows = [json.loads(line) for line in (segment / "raw_executions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["physical_cost"]["calls"], 9)
            self.assertEqual(list(segment.glob("units/*/UNIT_OUTPUTS_FROZEN.json")), [])
            with self.assertRaises(FileExistsError):
                collector.collect_unit(executor, self.question, 1701, self.requests([[1] * 12, [0] * 12]), segment, "plan", "sources")
            self.assertEqual(len(backend.calls), 10)

    def test_raw_commit_failure_keeps_paid_orphan_execution_and_does_not_resample(self):
        with tempfile.TemporaryDirectory() as temporary:
            segment = Path(temporary)
            backend = FakeBackend()
            executor = native.Executor(backend)
            original = collector.atomic_json
            def fail_raw(path, value, **options):
                if Path(path).parent.name == "raw":
                    raise OSError("Deliberate raw commit failure")
                return original(path, value, **options)
            with mock.patch.object(collector, "atomic_json", side_effect=fail_raw):
                with self.assertRaises(OSError):
                    collector.collect_unit(executor, self.question, 1701, self.requests([[1] * 12]), segment, "plan", "sources")
            self.assertEqual(len(backend.calls), 9)
            self.assertEqual(len(list(segment.glob("units/*/executions/*.json"))), 1)
            self.assertFalse((segment / "raw_executions.jsonl").exists())
            self.assertEqual(list(segment.glob("units/*/UNIT_OUTPUTS_FROZEN.json")), [])

    def test_shared_claim_has_one_owner_and_keeps_failed_claim_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            def claim(index):
                return collector.claim_unit(output, "calibration-0", 1701, f"gpu{index}", "plan")
            with ThreadPoolExecutor(max_workers=8) as pool:
                claims = list(pool.map(claim, range(8)))
            self.assertEqual(sum(path is not None for path in claims), 1)
            self.assertIsNone(collector.claim_unit(output, "calibration-0", 1701, "replacement", "plan"))
            self.assertIsNotNone(collector.claim_unit(output, "calibration-0", 1702, "gpu0", "plan"))

    def test_short_unit_claim_collision_rejects_different_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertEqual(len(collector.unit_id("calibration-0", 2701)), 20)
            with mock.patch.object(collector, "unit_id", return_value="deliberate_collision"):
                self.assertIsNotNone(collector.claim_unit(output, "calibration-0", 2701, "gpu0", "plan"))
                with self.assertRaisesRegex(ValueError, "collides"):
                    collector.claim_unit(output, "calibration-1", 2701, "gpu1", "plan")
                with self.assertRaisesRegex(ValueError, "collides"):
                    collector.claim_unit(output, "calibration-0", 2701, "gpu1", "other_plan")

    def test_single_flips_use_their_anchor_cache_and_second_anchor_uses_first_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            full = [1] * 12
            first_flip = [0] + [1] * 11
            second_flip = [1, 0] + [1] * 10
            second_anchor = [0] * 12
            second_anchor_flip = [1] + [0] * 11
            requests = [{"mask": full, "reference_mask": None, "aliases": [{"kind": "anchor", "context_id": "a"}]},
                        {"mask": first_flip, "reference_mask": full, "aliases": [{"kind": "flip", "edge": 0, "context_id": "a"}]},
                        {"mask": second_flip, "reference_mask": full, "aliases": [{"kind": "flip", "edge": 1, "context_id": "a"}]},
                        {"mask": second_anchor, "reference_mask": full, "aliases": [{"kind": "anchor", "context_id": "b"}]},
                        {"mask": second_anchor_flip, "reference_mask": second_anchor, "aliases": [{"kind": "flip", "edge": 0, "context_id": "b"}]}]
            backend = FakeBackend()
            executor = native.Executor(backend)
            rows = collector.collect_unit(executor, self.question, 2701, requests, Path(temporary), "plan", "sources")
            # Both first-anchor flips cost two calls. Chaining flips would cost three.
            self.assertEqual([row["physical_cost"]["calls"] for row in rows], [9, 2, 2, 5, 2])
            self.assertEqual(rows[2]["reference_mask"], full)
            self.assertEqual(rows[3]["reference_mask"], full)
            self.assertEqual(rows[4]["reference_mask"], second_anchor)

    def test_unavailable_anchor_and_invalid_request_fail_before_any_generation(self):
        backend = FakeBackend()
        executor = native.Executor(backend)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "reference mask"):
                collector.collect_unit(executor, self.question, 2701,
                    [{"mask": [0] * 12, "reference_mask": [1] * 12, "aliases": []}],
                    Path(temporary), "plan", "sources")
        self.assertEqual(backend.calls, [])

    def prepared_output(self, temporary):
        root = Path(temporary).resolve()
        output, backend = root / "results", FakeBackend()
        executor = native.Executor(backend)
        requests = [{"mask": [1] * 12, "reference_mask": None, "aliases": [{"kind": "anchor"}]},
                    {"mask": [0] + [1] * 11, "reference_mask": [1] * 12, "aliases": [{"kind": "single_flip", "edge": 0}]}]
        unit = {"question_id": self.question["id"], "seed": 2701, "partition": "train", "mask_requests": requests}
        plan = {"run_id": "g2_test", "protocol_hash": executor.protocol_hash, "model": {"identity": backend.identity},
                "features": {"sha256": "feature", "receipt_sha256": "receipt"}, "units": [unit],
                "qualification": {"worker_ids": ["gpu0"]}, "formal_call_upper_bound": 124000}
        context = {"root": root, "output": output, "plan": plan, "plan_hash": "plan", "source_hash": "source",
                   "pipeline_hash": "pipeline", "module": native, "config": executor.config,
                   "questions": {self.question["id"]: self.question}, "expected_graphs": 2}
        collector.atomic_json(output / "PLAN_FROZEN.json", {"state": "PLAN_FROZEN", "run_id": "g2_test", "plan_sha256": "plan",
            "source_manifest_sha256": "source", "pipeline_source_manifest_sha256": "pipeline", "features_sha256": "feature", "labels_read": False})
        qualification = output / "qualification/gpu0"
        collector.atomic_json(qualification / "QUALIFICATION.json", {"status": "PASS", "exact_replay": True,
            "run_id": "g2_test", "worker_id": "gpu0", "plan_sha256": "plan", "labels_read": False,
            "protocol_hash": executor.protocol_hash, "source_manifest_sha256": "source",
            "pipeline_source_manifest_sha256": "pipeline", "signature_sha256": collector.digest("same"),
            "model_identity": backend.identity, "physical_calls": 0})
        (qualification / "physical_calls.jsonl").touch()
        segment = output / "segments/gpu0"
        rows = collector.collect_unit(executor, self.question, 2701, requests, segment, "plan", "source", partition="train")
        for call in backend.calls:
            collector.append_json(segment / "physical_calls.jsonl", call)
        collector.atomic_json(segment / "run_manifest.json", {"worker_id": "gpu0", "run_id": "g2_test", "mode": "formal",
            "plan_sha256": "plan", "source_manifest_sha256": "source", "pipeline_source_manifest_sha256": "pipeline",
            "model_identity": backend.identity, "protocol_hash": executor.protocol_hash, "labels_read": False})
        state = {"phase": "GENERATION_COMPLETE_UNSCORED", "generation_complete": True, "labels_read": False,
                 "plan_sha256": "plan", "physical_calls": len(backend.calls)}
        collector.atomic_json(segment / "STATUS.json", state)
        claim = collector.claim_unit(output, self.question["id"], 2701, "gpu0", "plan")
        uid = collector.unit_id(self.question["id"], 2701)
        frozen = segment / "units" / uid / "UNIT_OUTPUTS_FROZEN.json"
        collector.atomic_json(claim.with_suffix(".done.json"), {"schema": "g2_collection_done_v1", "worker_id": "gpu0",
            "question_id": self.question["id"], "seed": 2701, "unit_id": uid, "plan_sha256": "plan", "labels_read": False,
            "claim_sha256": collector.file_digest(claim), "unit_frozen_path": frozen.relative_to(output).as_posix(),
            "unit_frozen_sha256": collector.file_digest(frozen), "observations": 2})
        return context, rows, segment

    def test_global_freeze_checks_complete_cache_outputs_and_physical_cost_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            context, rows, segment = self.prepared_output(temporary)
            report = collector.freeze_outputs(context)
            self.assertEqual(report["state"], "GLOBAL_OUTPUTS_FROZEN")
            self.assertEqual(report["observations"], 2)
            self.assertIsNone(report["first_gold_opened_unix"])
            audit = collector.read_json(context["output"] / "PHYSICAL_COST_AUDIT.json")
            self.assertEqual(audit["formal_calls"], 11)
            self.assertEqual(audit["referenced_calls"], 11)
            self.assertEqual(audit["orphan_call_ids"], [])
            merged = list(collector.read_jsonl(context["output"] / "observations.jsonl"))
            self.assertEqual(merged[0]["mask_id"], "111111111111")
            self.assertEqual(collector.file_digest(context["root"] / merged[1]["execution_path"]), rows[1]["execution_sha256"])
            self.assertEqual(report["artifacts"]["observations"]["sha256"], collector.file_digest(context["output"] / "observations.jsonl"))

    def test_global_freeze_retains_unreferenced_paid_call_as_orphan_without_extra_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            context, rows, segment = self.prepared_output(temporary)
            collector.append_json(segment / "physical_calls.jsonl", {"call_id": "paid_orphan", "status": "SUCCESS",
                                  "input_tokens": 7, "output_tokens": 3, "elapsed_seconds": .4})
            state = collector.read_json(segment / "STATUS.json")
            state["physical_calls"] += 1
            collector.atomic_json(segment / "STATUS.json", state)
            report = collector.freeze_outputs(context)
            audit = collector.read_json(context["output"] / "PHYSICAL_COST_AUDIT.json")
            self.assertEqual(report["observations"], 2)
            self.assertEqual(audit["formal_calls"], 12)
            self.assertEqual(audit["referenced_calls"], 11)
            self.assertEqual(audit["orphan_call_ids"], ["paid_orphan"])

    def test_global_freeze_rejects_execution_byte_tamper_and_missing_completion(self):
        for tamper in ("execution", "done"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                context, rows, segment = self.prepared_output(temporary)
                if tamper == "execution":
                    (segment / rows[1]["execution_path"]).write_text("{}", encoding="utf-8")
                else:
                    next((context["output"] / "claims").glob("*.done.json")).unlink()
                with self.assertRaises(ValueError):
                    collector.freeze_outputs(context)
                self.assertFalse((context["output"] / "GLOBAL_OUTPUTS_FROZEN.json").exists())

    def test_qualified_status_or_partial_generation_cannot_freeze_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            context, rows, segment = self.prepared_output(temporary)
            state = collector.read_json(segment / "STATUS.json")
            state.update(phase="GENERATING", generation_complete=False)
            collector.atomic_json(segment / "STATUS.json", state)
            with self.assertRaisesRegex(ValueError, "unfinished"):
                collector.freeze_outputs(context)
            self.assertFalse((context["output"] / "GLOBAL_OUTPUTS_FROZEN.json").exists())

    def test_stop_receipt_stops_after_a_committed_graph_without_extra_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            segment, backend = Path(temporary), FakeBackend()
            executor = native.Executor(backend)
            context = {"output": segment, "plan": {"run_id": "g2"}, "plan_hash": "plan"}
            def stop(row):
                collector.atomic_json(segment / "STOP_REQUESTED.json", {"run_id": "g2", "plan_sha256": "plan"})
                collector.check_stop(context)
            with self.assertRaises(collector.StopRequested):
                collector.collect_unit(executor, self.question, 2701, self.requests([[1] * 12, [0] * 12]),
                                       segment, "plan", "sources", progress=stop)
            self.assertEqual(len(backend.calls), 9)
            self.assertEqual(len(list(collector.read_jsonl(segment / "raw_executions.jsonl"))), 1)
            self.assertEqual(list(segment.glob("units/*/UNIT_OUTPUTS_FROZEN.json")), [])

    def test_missing_or_wrong_plan_freeze_fails_before_backend_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = {"output": Path(temporary), "plan_hash": "plan", "source_hash": "source", "pipeline_hash": "pipeline",
                       "plan": {"run_id": "g2", "features": {"sha256": "feature"}}}
            with mock.patch.object(collector, "load_backend") as backend:
                with self.assertRaises(FileNotFoundError):
                    collector.collect(context, "gpu0", "cuda:0")
                collector.atomic_json(context["output"] / "PLAN_FROZEN.json", {"state": "PLAN_FROZEN", "plan_sha256": "other"})
                with self.assertRaises(ValueError):
                    collector.qualify(context, "gpu0", "cuda:0")
                backend.assert_not_called()

    def test_all_26_requests_qualification_uses_89_calls_and_exact_full_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            backend, requests = FakeBackend(), []
            executor = native.Executor(backend)
            for anchor in ([1] * 12, [0] * 12):
                requests.append({"mask": anchor, "reference_mask": [1] * 12 if requests else None,
                                 "aliases": [{"kind": "anchor"}]})
                for edge in range(12):
                    mask = anchor.copy()
                    mask[edge] = 1 - mask[edge]
                    requests.append({"mask": mask, "reference_mask": anchor, "aliases": [{"edge": edge}]})
            questions_path = root / "qualification_questions.jsonl"
            questions_path.write_text(native.canonical(self.question) + "\n", encoding="utf-8")
            settings = {"questions": {"path": questions_path.name, "sha256": collector.file_digest(questions_path),
                         "payload_sha256": {self.question["id"]: native.digest(native.Executor.question_payload(self.question))}},
                        "units": [{"question_id": self.question["id"], "seed": 2700, "mask_requests": requests}],
                        "independent_full_replay_request_indices": [0, 1, 13]}
            context = {"root": root, "output": root / "out", "plan_hash": "plan", "source_hash": "source", "pipeline_hash": "pipeline",
                       "module": native, "plan": {"run_id": "g2", "features": {"sha256": "feature"},
                       "protocol_hash": executor.protocol_hash, "qualification": settings, "source_manifest": {}, "pipeline_source_manifest": {}}}
            collector.atomic_json(context["output"] / "PLAN_FROZEN.json", {"state": "PLAN_FROZEN", "run_id": "g2", "plan_sha256": "plan",
                "source_manifest_sha256": "source", "pipeline_source_manifest_sha256": "pipeline", "features_sha256": "feature", "labels_read": False})
            with mock.patch.object(collector, "load_backend", return_value=(backend, executor)), mock.patch.object(collector, "verify_manifest"):
                report = collector.qualify(context, "gpu0", "cuda:0")
            self.assertEqual(report["physical_calls"], 89)
            self.assertEqual(report["replay_checks"], [True, True, True])
            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["labels_read"])

    def test_reopening_generation_after_freeze_or_gold_access_is_rejected(self):
        for name in ("GLOBAL_OUTPUTS_FROZEN.json", "FIRST_TRAIN_TUNE_LABELS_OPENED.json", "FIRST_HELDOUT_LABELS_OPENED.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                context = {"output": Path(temporary)}
                collector.atomic_json(context["output"] / name, {})
                with self.assertRaises(FileExistsError):
                    collector.require_unreleased_outputs(context)

    @unittest.skipUnless(
        (HERE / "DESIGN_DRAFT.json").is_file(),
        "Generate the private DESIGN_DRAFT.json locally to run the real 400-question plan regression",
    )
    def test_real_draft_question_to_partition_map_and_five_seed_coverage(self):
        draft = collector.read_json(HERE / "DESIGN_DRAFT.json")
        collector.validate_design_coverage(draft)
        self.assertEqual(len({collector.unit_id(unit["question_id"], unit["seed"]) for unit in draft["units"]}), 2000)
        self.assertEqual(sum(len(collector.unique_mask_requests(unit["mask_requests"])) for unit in draft["units"]), 51970)
        draft["units"][0]["partition"] = "test" if draft["units"][0]["partition"] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "five-seed"):
            collector.validate_design_coverage(draft)


if __name__ == "__main__":
    unittest.main()
