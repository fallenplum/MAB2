"""Mock contract tests for real frozen Executor continuation; no GPU or labels."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("elastic_worker", HERE / "elastic_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import graph_credit.executor as native


class FakeBackend:
    identity = {"backend": "mock_contract_only", "model": "frozen"}
    def __init__(self):
        self.calls = []
    def encode(self, text):
        return [ord(c) for c in text]
    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)
    def generate(self, messages, seed, max_new_tokens, config):
        answer = str(int(worker.sha(worker.canonical([messages, seed]).encode())[:8], 16) % 100)
        text = native.canonical({"answer": answer, "evidence": "Checked internally.", "uncertainty": ""})
        value = native.Generation(text, len(native.canonical(messages)), len(text),
                                  (len(self.calls) + 1) * .00013, (len(self.calls) + 1) * .000017, 1024, "eos")
        self.calls.append(value)
        return value


class ElasticContracts(unittest.TestCase):
    def setUp(self):
        cfg = json.loads((PROJECT_ROOT / "configs/motivation_v3.json").read_text(encoding="utf-8-sig"))
        self.plan = {"base_masks": cfg["base_masks"], "edge_pairs": cfg["edge_pairs"], "seeds": cfg["seeds"],
                     "config": cfg, "question_ids": ["q0", "q1", "q2"]}
        self.question = {"id": "q0", "question": "Compute 3 plus 2.", "task": "gsm8k", "split": "dev"}

    def test_canonical_deserialization_preserves_all_cache_and_float_costs(self):
        executor = native.Executor(FakeBackend(), worker.native_config(self.plan, native))
        result = executor.execute(self.question, [1] * 12, 1701)
        encoded = json.loads(worker.canonical(result.to_dict()))
        self.assertEqual(next(iter(encoded["nodes"])), "aggregate")
        restored = worker.deserialize_execution(encoded, native)
        self.assertEqual(list(restored.nodes), list(result.nodes))
        self.assertEqual(restored.cost(), result.cost())
        self.assertEqual(restored.cost(False), result.cost(False))
        worker.validate_base(restored, executor, self.question, [1] * 12, 1701, native)
        self.assertEqual(restored.to_dict(), result.to_dict())

    def test_tampered_generation_or_cost_is_rejected(self):
        executor = native.Executor(FakeBackend(), worker.native_config(self.plan, native))
        encoded = json.loads(worker.canonical(executor.execute(self.question, [1] * 12, 1701).to_dict()))
        tampered = copy.deepcopy(encoded)
        tampered["nodes"]["r1:w0"]["generation"]["text"] = "tampered"
        with self.assertRaises(ValueError):
            worker.deserialize_execution(tampered, native)
        encoded["physical_cost"]["output_tokens"] += 1
        with self.assertRaises(ValueError):
            worker.deserialize_execution(encoded, native)

    def test_closed_input_validation_rejects_wrong_seed_question_and_mask(self):
        executor = native.Executor(FakeBackend(), worker.native_config(self.plan, native))
        base = executor.execute(self.question, [1] * 12, 1701)
        for question, mask, seed in ((self.question, [1] * 12, 1702),
                                     ({**self.question, "question": "Changed input"}, [1] * 12, 1701),
                                     (self.question, [0] + [1] * 11, 1701)):
            with self.assertRaises(ValueError):
                worker.validate_base(base, executor, question, mask, seed, native)

    def test_partial_unit_continues_missing_cells_exactly_and_charges_only_new_calls(self):
        steps = worker.design_steps(self.plan)
        self.assertEqual(len(steps), 60)
        original_backend = FakeBackend()
        original = native.Executor(original_backend, worker.native_config(self.plan, native))
        executions, base = [], None
        for step in steps:
            base = original.execute(self.question, worker.step_mask(step), 1701, base=base)
            executions.append(base)
        for start in (1, 7, 17, 59):
            with self.subTest(start=start), tempfile.TemporaryDirectory(dir=HERE) as temp:
                root = Path(temp)
                shard = root / "results/pilot" / worker.RUN_ID / "shard_000"
                (shard / "executions").mkdir(parents=True)
                worker.atomic(shard / "frozen_plan.json", self.plan)
                resumed_sha = worker.atomic(shard / "executions/last.json", executions[start - 1].to_dict())
                segment = shard / "elastic_segments/gpu0"
                segment.mkdir(parents=True)
                task = {"logical_shard": 0, "question_id": "q0", "seed": 1701,
                        "start_index": start, "expected_new_rows": 60 - start,
                        "resume_execution_path": "executions/last.json", "resume_execution_sha256": resumed_sha}
                backend = FakeBackend()
                executor = native.Executor(backend, worker.native_config(self.plan, native))
                rows = worker.execute_task(root, task, self.plan, self.question, executor, native, segment, "taskhash")
                self.assertEqual(len(rows), 60 - start)
                self.assertEqual(len({worker.row_key(row) for row in rows}), 60 - start)
                self.assertEqual(sum(row["physical_cost"]["calls"] for row in rows), len(backend.calls))
                for index, row in enumerate(rows, start):
                    saved = json.loads((segment / row["execution_path"]).read_text())
                    for name in saved["nodes"]:
                        self.assertEqual(saved["nodes"][name]["wire_message"], executions[index].nodes[name].wire_message)
                        self.assertEqual(saved["nodes"][name]["messages"], executions[index].nodes[name].messages)
                    self.assertIsNone(row["quality"])
                    self.assertEqual(row["elastic_cell_index"], index)
                self.assertFalse((segment / "observations.jsonl").exists())

    def test_completed_unit_wrong_shard_and_wrong_prefix_are_rejected(self):
        task = {"logical_shard": 0, "question_id": "q0", "seed": 1701, "start_index": 0, "expected_new_rows": 60}
        worker.validate_task(task, self.plan, {})
        for changed, completed in (({**task, "logical_shard": 1}, {}),
                                    ({**task, "start_index": 60, "expected_new_rows": 0}, {}),
                                    (task, {(0, "q0", 1701): 60}),
                                    ({**task, "start_index": 7, "expected_new_rows": 53}, {(0, "q0", 1701): 7})):
            with self.assertRaises(ValueError):
                worker.validate_task(changed, self.plan, completed)

    def test_atomic_claim_has_one_owner_and_never_steals_existing_or_crashed_claim(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            task = {"logical_shard": 0, "question_id": "q0", "seed": 1701, "start_index": 7, "expected_new_rows": 53}
            claim = worker.claim_unit(root, task, "gpu0", "first")
            self.assertTrue(claim.is_file())
            self.assertIsNone(worker.claim_unit(root, task, "gpu1", "second"))
            self.assertEqual(json.loads(claim.read_text())["worker_id"], "gpu0")
            claim.write_bytes(b"")  # Simulates a crash after O_EXCL, before append.
            self.assertIsNone(worker.claim_unit(root, task, "gpu2", "third"))

    def test_bootstrap_shared_global_coverage_and_static_overlap_are_enforced(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            private = root / "artifacts/private/performance"
            private.mkdir(parents=True)
            worker_path = private / "elastic_worker.py"
            worker_path.write_bytes((HERE / "elastic_worker.py").read_bytes())
            loaded_spec = importlib.util.spec_from_file_location("bootstrap_test_worker", worker_path)
            loaded = importlib.util.module_from_spec(loaded_spec)
            loaded_spec.loader.exec_module(loaded)
            source_path = root / "src/graph_credit/executor.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes((PROJECT_ROOT / "src/graph_credit/executor.py").read_bytes())
            sources = {"src/graph_credit/executor.py": {"sha256": worker.sha(source_path.read_bytes()), "bytes": source_path.stat().st_size}}
            scorer = {"callable": "frozen_scorer:score_item", "version": "mock_contract"}
            pilot = SimpleNamespace(source_manifest=lambda _: sources, scorer_identity=lambda _: scorer)
            questions = [{**self.question, "id": qid} for qid in self.plan["question_ids"]]
            data_path = root / "data/derived/motivation_questions.jsonl"
            data_path.parent.mkdir(parents=True)
            data_path.write_text("".join(worker.canonical(q) + "\n" for q in questions), encoding="utf-8")
            plan = {**self.plan, "seeds": [1701], "num_shards": 3, "intended_shards": 3,
                    "source_manifest": sources, "source_manifest_sha256": native.digest(sources),
                    "scorer_identity": scorer, "scorer": scorer["callable"],
                    "question_payload_sha256": [native.digest(native.Executor.question_payload(q)) for q in questions],
                    "data_sha256": worker.sha(data_path.read_bytes())}
            normalized = []
            for index in range(3):
                shard = root / "results/pilot" / worker.RUN_ID / f"shard_{index:03d}"
                shard.mkdir(parents=True)
                plan_sha = worker.atomic(shard / "frozen_plan.json", plan)
                (shard / "raw_executions.jsonl").write_bytes(b"")
                normalized.append({"logical_shard": index, "path": f"results/pilot/{worker.RUN_ID}/shard_{index:03d}/raw_executions.jsonl",
                                   "normalized_sha256": worker.sha(b"")})
            receipt = {"state": "COMPLETE", "run_id": worker.RUN_ID, "labels_read": False,
                       "frozen_plan_sha256": plan_sha, "files": normalized}
            receipt_path = root / "artifacts/receipt.json"
            receipt_sha = worker.atomic(receipt_path, receipt)
            scheduler = {"artifacts/private/performance/elastic_worker.py": worker.sha(worker_path.read_bytes())}
            tasks = [{"logical_shard": index, "question_id": qid, "seed": 1701, "start_index": 0, "expected_new_rows": 60}
                     for index, qid in enumerate(plan["question_ids"])]
            allocation = {"run_id": worker.RUN_ID, "scheduler_sources": scheduler,
                          "frozen_plan_sha256": plan_sha, "migration_receipt_sha256": receipt_sha,
                          "remaining_rows": 180, "workers": []}
            for owner in ("gpu0", "gpu1"):
                manifest = {"schema": "graph_credit_elastic_tasks_v1", "worker_id": owner, "run_id": worker.RUN_ID,
                            "logical_shards": 3, "labels_read": False, "frozen_plan_sha256": plan_sha,
                            "migration_receipt": "artifacts/receipt.json", "migration_receipt_sha256": receipt_sha,
                            "scheduler_sources": scheduler, "tasks": tasks, "expected_new_rows": 180,
                            "claim_mode": "shared_atomic_units"}
                task_path = private / f"tasks_{owner}.json"
                task_sha = worker.atomic(task_path, manifest)
                allocation["workers"].append({"worker_id": owner, "task_manifest": task_path.relative_to(root).as_posix(),
                                               "task_manifest_sha256": task_sha})
            worker.atomic(private / "allocation.json", allocation)
            manifest, observed_plan, observed_questions, _ = loaded.bootstrap(root, private / "tasks_gpu0.json", "gpu0", native, pilot)
            self.assertEqual(manifest["claim_mode"], "shared_atomic_units")
            self.assertEqual(observed_plan, plan)
            self.assertEqual(set(observed_questions), set(plan["question_ids"]))
            # Same overlap is rejected when claiming is not explicitly enabled.
            for item in allocation["workers"]:
                task_path = root / item["task_manifest"]
                manifest = json.loads(task_path.read_text())
                manifest.pop("claim_mode")
                item["task_manifest_sha256"] = worker.atomic(task_path, manifest)
            worker.atomic(private / "allocation.json", allocation)
            with self.assertRaisesRegex(ValueError, "owned by multiple"):
                loaded.bootstrap(root, private / "tasks_gpu0.json", "gpu0", native, pilot)


if __name__ == "__main__":
    unittest.main()
