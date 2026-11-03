"""CPU-only finalizer contract checks; never read real gold or mutate a run."""
from __future__ import annotations

import dataclasses
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("elastic_finalize_tested", HERE / "elastic_finalize.py")
finalizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = finalizer
SPEC.loader.exec_module(finalizer)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from graph_credit.executor import Executor, Generation, GraphConfig, canonical


class Backend:
    identity = {"backend": "synthetic_finalizer_contract_only"}

    def __init__(self):
        self.calls = {}

    def encode(self, text):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))

    def generate(self, messages, seed, max_new_tokens, config):
        text = canonical({"answer": "4", "evidence": "2+2", "uncertainty": ""})
        generation = Generation(text, len(canonical(messages)), len(text), .01, .008, 10, "eos")
        self.calls[generation.call_id] = {
            "call_id": generation.call_id, "status": "SUCCESS", "seed": seed,
            "input_tokens": generation.input_tokens, "output_tokens": generation.output_tokens,
            "gpu_seconds": generation.gpu_seconds, "elapsed_seconds": generation.elapsed_seconds,
            "prompt_sha256": finalizer.digest(canonical(messages).encode()), "finish_reason": "eos"}
        return generation


def execution_fixture():
    backend = Backend()
    result = Executor(backend, GraphConfig()).execute({"id": "q", "question": "2+2?"}, [1] * 12, 1701)
    row = {"execution_id": result.execution_id, "question_id": result.question_id,
           "graph_mask": list(result.mask), "seed": result.seed, "protocol_hash": result.protocol_hash,
           "answer": result.answer, "service_output": result.nodes["aggregate"].generation.text,
           "physical_cost": result.cost()}
    import json
    execution = json.loads(canonical(result.to_dict()))
    return backend, result, row, execution


def lineage_fixture(root):
    """Build scheduler/claim evidence with real worker JSON formats, no labels."""
    private = root / "artifacts/private/performance"
    private.mkdir(parents=True)
    module_path = private / "elastic_finalize.py"
    module_path.write_bytes((HERE / "elastic_finalize.py").read_bytes())
    spec = importlib.util.spec_from_file_location("finalizer_lineage_fixture", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runid = "motivation_v3_20261103"
    masks = [{"name": name, "mask": [0] * 12} for name in ("full", "chain", "star", "fixed_random", "sequential_deletion")]
    plan = {"question_ids": ["q0", "q1", "q2"], "seeds": [1701], "base_masks": masks,
            "edge_pairs": [[0, 4], [2, 8], [6, 10]], "protocol_hash": "protocol",
            "source_manifest": {"src/example.py": {"sha256": "immutable", "bytes": 1}},
            "source_manifest_sha256": "source", "scorer_identity": {"version": "fixed"},
            "model_identity": {"weights": "fixed_model"}}
    plan_hash = "frozen_plan"
    scheduler = {"artifacts/private/performance/elastic_finalize.py": module.digest(module_path.read_bytes())}
    receipt_path = root / "artifacts/audit/elastic_migration_20261103.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(module.json_bytes({"state": "COMPLETE", "frozen_plan_sha256": plan_hash, "labels_read": False}))
    receipt_hash = module.digest(receipt_path.read_bytes())
    tasks = [{"logical_shard": i, "question_id": f"q{i}", "seed": 1701, "start_index": 0, "expected_new_rows": 60} for i in range(3)]
    workers = []
    for worker_id in ("gpu0", "gpu1"):
        manifest = {"schema": "graph_credit_elastic_tasks_v1", "worker_id": worker_id, "run_id": runid,
                    "logical_shards": 3, "frozen_plan_sha256": plan_hash, "scheduler_sources": scheduler,
                    "migration_receipt_sha256": receipt_hash, "labels_read": False,
                    "claim_mode": "shared_atomic_units", "tasks": tasks, "expected_new_rows": 180}
        path = private / ("tasks_" + worker_id + ".json")
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        workers.append({"worker_id": worker_id, "task_manifest": path.relative_to(root).as_posix(),
                        "task_manifest_sha256": module.digest(path.read_bytes())})
    allocation = {"schema": "graph_credit_elastic_allocation_v1", "run_id": runid,
                  "logical_shards": 3, "frozen_plan_sha256": plan_hash, "migration_receipt_sha256": receipt_hash,
                  "scheduler_sources": scheduler, "labels_read": False, "claim_mode": "shared_atomic_units",
                  "workers": workers, "remaining_rows": 180}
    allocation_path = private / "allocation.json"
    allocation_path.write_bytes(module.json_bytes(allocation))
    gate = module._verify_allocation(root, runid, plan, plan_hash, allocation_path, {i: [] for i in range(3)})
    segments = []
    original = module.read_json(private / "tasks_gpu0.json")
    for task in tasks:
        index = task["logical_shard"]
        segment = root / "results/pilot" / runid / f"shard_{index:03d}/elastic_segments/gpu0"
        segment.mkdir(parents=True)
        (segment / "task_manifest.json").write_bytes(module.json_bytes(original))
        runtime = {"worker_id": "gpu0", "task_manifest_sha256": workers[0]["task_manifest_sha256"],
                   "frozen_plan_sha256": plan_hash, "protocol_hash": plan["protocol_hash"], "model_identity": plan["model_identity"],
                   "source_manifest": plan["source_manifest"], "source_manifest_sha256": plan["source_manifest_sha256"],
                   "scorer_identity": plan["scorer_identity"], "scheduler_sources": scheduler,
                   "migration_receipt_sha256": receipt_hash, "labels_read": False}
        (segment / "run_manifest.json").write_bytes(module.json_bytes(runtime))
        (segment / "STATUS.json").write_bytes(module.json_bytes({"worker_id": "gpu0", "task_manifest_sha256": workers[0]["task_manifest_sha256"],
                      "phase": "GENERATION_COMPLETE_UNSCORED", "generation_complete": True, "scored": False,
                      "labels_read": False, "segment_rows": 60}))
        key = (index, task["question_id"], task["seed"])
        claim_id = module.digest(module.canonical(list(key)).encode())
        claims = root / "results/pilot" / runid / "elastic_claims"
        claims.mkdir(exist_ok=True)
        claim_path = claims / (claim_id + ".json")
        claim_path.write_bytes(module.json_bytes({"schema": "graph_credit_elastic_unit_claim_v1", "worker_id": "gpu0", "task": task,
                               "task_manifest_sha256": workers[0]["task_manifest_sha256"], "labels_read": False}))
        done_path = claims / (claim_id + ".done.json")
        done_path.write_bytes(module.json_bytes({"schema": "graph_credit_elastic_unit_completion_v1", "worker_id": "gpu0", "task": task,
                              "task_manifest_sha256": workers[0]["task_manifest_sha256"], "claim_sha256": module.digest(claim_path.read_bytes()),
                              "new_rows": 60, "segment": segment.relative_to(root).as_posix(),
                              "last_execution_sha256": f"{59:064x}", "labels_read": False}))
        rows = [{"question_id": task["question_id"], "seed": task["seed"], "logical_shard": index,
                 "elastic_task_manifest_sha256": workers[0]["task_manifest_sha256"], "elastic_unit_start_index": 0,
                 "elastic_cell_index": offset, "context_family": context["name"], "edge_e": e, "edge_f": f, "cell": cell,
                 "execution_sha256": f"{offset:064x}"} for offset, (context, e, f, cell) in enumerate(gate["steps"])]
        segments.append((index, segment, rows, claim_path, done_path))
    return module, runid, plan, plan_hash, gate, segments


class FinalizerContracts(unittest.TestCase):
    def test_segment_manifest_objects_and_exclusive_claim_lineage_validate(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            module, runid, plan, plan_hash, gate, segments = lineage_fixture(root)
            self.assertNotEqual((root / "artifacts/private/performance/tasks_gpu0.json").read_bytes(),
                                (segments[0][1] / "task_manifest.json").read_bytes())
            for index, directory, rows, _, _ in segments:
                module._verify_segment(root, runid, plan, plan_hash, index, directory, gate)
                for row in rows:
                    module._verify_elastic_row(root, runid, index, directory, row, gate)
            module._finish_lineage(gate)
            module._unchanged_lineage(gate)
            self.assertEqual(len(gate["unit_rows"]), 3)

    def test_runtime_model_or_claim_owner_or_completion_tampering_rejected(self):
        for kind in ("model", "claim_owner", "missing_done", "row_task_hash", "row_index"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=HERE) as temporary:
                root = Path(temporary)
                module, runid, plan, plan_hash, gate, segments = lineage_fixture(root)
                index, directory, rows, claim, done = segments[0]
                if kind == "model":
                    runtime = module.read_json(directory / "run_manifest.json")
                    runtime["model_identity"] = {"weights": "wrong_model"}
                    (directory / "run_manifest.json").write_bytes(module.json_bytes(runtime))
                    with self.assertRaises(module.FinalizationError):
                        module._verify_segment(root, runid, plan, plan_hash, index, directory, gate)
                    continue
                module._verify_segment(root, runid, plan, plan_hash, index, directory, gate)
                if kind == "claim_owner":
                    value = module.read_json(claim)
                    value["worker_id"] = "gpu1"
                    claim.write_bytes(module.json_bytes(value))
                elif kind == "missing_done":
                    done.rename(done.with_name(done.name + ".withheld"))
                elif kind == "row_task_hash":
                    rows[0]["elastic_task_manifest_sha256"] = "unallocated"
                elif kind == "row_index":
                    rows[0]["elastic_cell_index"] = 1
                with self.assertRaises((module.FinalizationError, FileNotFoundError)):
                    module._verify_elastic_row(root, runid, index, directory, rows[0], gate)

    def test_omitted_unit_or_changed_scheduler_is_not_accepted(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            module, runid, plan, plan_hash, gate, segments = lineage_fixture(root)
            for index, directory, rows, _, _ in segments[:2]:
                module._verify_segment(root, runid, plan, plan_hash, index, directory, gate)
                for row in rows:
                    module._verify_elastic_row(root, runid, index, directory, row, gate)
            with self.assertRaises(module.FinalizationError):
                module._finish_lineage(gate)
            (root / "artifacts/private/performance/tasks_gpu0.json").write_text("{}")
            with self.assertRaises(module.FinalizationError):
                module._unchanged_lineage(gate)

    def test_path_escape_absolute_and_windows_spellings_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            (root / "ok.json").write_text("{}")
            self.assertEqual(finalizer.safe_relative(root, "ok.json"), root / "ok.json")
            for bad in ("../ok.json", "/ok.json", "C:/ok.json", "..\\ok.json"):
                with self.subTest(path=bad), self.assertRaises(finalizer.FinalizationError):
                    finalizer.safe_relative(root, bad)

    def test_json_duplicate_members_and_partial_line_rejected(self):
        with self.assertRaises(finalizer.FinalizationError):
            finalizer.decode_json(b'{"seed":1,"seed":2}')
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            path = Path(temporary) / "raw.jsonl"
            path.write_bytes(b'{"x":1}')
            with self.assertRaises(finalizer.FinalizationError):
                finalizer.read_jsonl(path)

    def test_frozen_design_has_exactly_60000_unique_keys(self):
        plan = {"question_ids": [f"q{i}" for i in range(200)],
                "seeds": [1701, 1702, 1703, 1704, 1705], "edge_pairs": [[0, 4], [2, 8], [6, 10]],
                "num_shards": 3, "intended_shards": 3,
                "base_masks": [{"name": name, "mask": [0] * 12} for name in (
                    "full", "chain", "star", "fixed_random", "sequential_deletion")]}
        self.assertEqual(len(finalizer._design(plan)[2]), 60000)
        plan["seeds"] = [1701]
        with self.assertRaises(finalizer.FinalizationError):
            finalizer._design(plan)

    def test_actual_execution_node_seed_prompt_and_cost_reconcile(self):
        backend, result, row, execution = execution_fixture()
        used = set()
        finalizer._verify_execution(execution, row, result.question_fingerprint, backend.calls, used)
        self.assertEqual(used, set(backend.calls))
        with self.assertRaises(finalizer.FinalizationError):
            finalizer._verify_execution(execution, row, result.question_fingerprint, backend.calls, used)

    def test_service_output_or_paid_prompt_tampering_rejected(self):
        backend, result, row, execution = execution_fixture()
        row["service_output"] = "changed after gold"
        with self.assertRaises(finalizer.FinalizationError):
            finalizer._verify_execution(execution, row, result.question_fingerprint, backend.calls, set())
        row["service_output"] = result.nodes["aggregate"].generation.text
        next(iter(backend.calls.values()))["prompt_sha256"] = "0" * 64
        with self.assertRaises(finalizer.FinalizationError):
            finalizer._verify_execution(execution, row, result.question_fingerprint, backend.calls, set())

    def test_orphan_failed_and_unrecorded_costs_are_not_erased(self):
        backend, result, row, execution = execution_fixture()
        calls = list(backend.calls.values())
        failed = {"call_id": "failed", "status": "FAILED", "input_tokens": 100,
                  "output_tokens": None, "elapsed_seconds": 1.2}
        used = {calls[0]["call_id"]}
        summary = finalizer._paid_summary(calls + [failed], used, 3)
        self.assertEqual(summary["recorded_success_all_paid"]["calls"], 9)
        self.assertEqual(summary["recorded_success_associated_with_fresh_nodes"]["calls"], 1)
        self.assertEqual(summary["recorded_success_orphaned_from_completed_graphs"]["calls"], 8)
        self.assertEqual(summary["recorded_failed_calls"], 1)
        self.assertIsNone(summary["failed_call_costs"][0]["output_tokens"])
        self.assertIsNone(summary["migration_unrecorded_inflight"]["gpu_seconds"])
        self.assertIsNone(summary["migration_unrecorded_inflight"]["exact_count"])

    def test_backup_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            path = Path(temporary) / "original_raw_executions.jsonl"
            finalizer._exclusive_backup(path, b"first\n")
            finalizer._exclusive_backup(path, b"first\n")
            with self.assertRaises(finalizer.FinalizationError):
                finalizer._exclusive_backup(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")

    def test_migration_prefix_preserves_original_tail_and_normalizes_only_complete_rows(self):
        full = b'{"row":1}\r\n{"row":2}\n{"incomplete":'
        rows, normalized, tail = finalizer._normalize_migration_backup(full)
        self.assertEqual(rows, 2)
        self.assertEqual(normalized, b'{"row":1}\n{"row":2}\n')
        self.assertEqual(tail, b'{"incomplete":')
        self.assertEqual(full, b'{"row":1}\r\n{"row":2}\n{"incomplete":')
        with self.assertRaises(finalizer.FinalizationError):
            finalizer._normalize_migration_backup(b'{broken\n{"row":2}\n')

    def test_partial_and_forged_complete_receipts_do_not_trigger_rescore(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            parent = root / "results/pilot/run"
            parent.mkdir(parents=True)
            path = parent / "ELASTIC_FINALIZATION.json"
            for receipt in ({"phase": "COMMITTING_BEFORE_LABELS"}, {"phase": "COMPLETE", "committed_artifact_hashes": {}}):
                path.write_bytes(finalizer.json_bytes(receipt))
                with self.assertRaises(finalizer.FinalizationError):
                    finalizer.finalize(root, "run", commit=True)


if __name__ == "__main__":
    unittest.main()
