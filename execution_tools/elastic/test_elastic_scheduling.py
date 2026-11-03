"""CPU-only tests for frozen-design scheduling; never invokes process or GPU APIs.

Run beside the two modules with:
    python -m unittest -v test_elastic_scheduling
All fixture files are isolated under this file's authorized staging directory.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


migrate = load_module("elastic_migrate_under_test", "elastic_migrate.py")
pool = load_module("elastic_supervisor_under_test", "elastic_supervisor.py")


def json_line(value):
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ValidPrefixTests(unittest.TestCase):
    def test_rejects_bad_record_in_middle_even_if_tail_is_valid(self):
        with self.assertRaisesRegex(ValueError, "before final"):
            migrate.valid_prefix(json_line({"a": 1}) + b'{broken}\n' + json_line({"a": 3}))

    def test_preserves_incomplete_final_record_bytes(self):
        first = json_line({"question_id": "q1", "text": "完整"})
        tail = b'{"question_id":"q2","text":"unfinished'
        records, normalized, retained_tail = migrate.valid_prefix(first + tail)
        self.assertEqual(records, [{"question_id": "q1", "text": "完整"}])
        self.assertEqual(normalized, first)
        self.assertEqual(retained_tail, tail)

    def test_preserves_incomplete_utf8_tail_without_silent_data_loss(self):
        first = json_line({"a": 1})
        tail = b'{"text":"\xe4\xb8'
        records, normalized, retained_tail = migrate.valid_prefix(first + tail)
        self.assertEqual(records, [{"a": 1}])
        self.assertEqual(normalized, first)
        self.assertEqual(retained_tail, tail)

    def test_normalizes_crlf_and_preserves_all_object_rows(self):
        data = b'{"a":1}\r\n{"a":2}\r\n'
        rows, normalized, tail = migrate.valid_prefix(data)
        self.assertEqual(rows, [{"a": 1}, {"a": 2}])
        self.assertEqual(normalized, b'{"a":1}\n{"a":2}\n')
        self.assertEqual(tail, b"")

    def test_rejects_nonobject_rows_including_last_record(self):
        for row in ([], None, 1, "row"):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "Non-object"):
                migrate.valid_prefix(json_line({"ok": True}) + json_line(row))

    def test_empty_ledger_is_empty_prefix(self):
        self.assertEqual(migrate.valid_prefix(b""), ([], b"", b""))


class TaskPrefixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="elastic_pure_tests_", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.plan = {
            "question_ids": [f"q{i}" for i in range(7)],
            "seeds": [2024, 2025],
            "base_masks": [{"name": f"context_{i}"} for i in range(3)],
            "edge_pairs": [[f"edge_{i}", f"edge_{i + 5}"] for i in range(5)],
        }
        self.shard_index = 0
        self.shard = self.root / "results/pilot" / migrate.RUN_ID / "shard_000"
        self.execution = self.shard / "executions" / "last.json"
        self.execution.parent.mkdir(parents=True)
        self.execution_bytes = b'{"cache":"saved frozen execution"}\n'
        self.execution.write_bytes(self.execution_bytes)
        self.execution_hash = hashlib.sha256(self.execution_bytes).hexdigest()
        self.rows = []
        for qid in ("q0", "q3", "q6"):
            for seed in self.plan["seeds"]:
                for context, pair, cell in migrate.design_steps(self.plan):
                    self.rows.append({
                        "question_id": qid, "seed": seed,
                        "context_family": context["name"],
                        "edge_e": pair[0], "edge_f": pair[1], "cell": cell,
                        "execution_path": "executions/last.json",
                        "execution_sha256": self.execution_hash,
                    })

    def tasks(self, rows, shard_index=0):
        return migrate.make_tasks(self.root, self.plan, shard_index, rows)

    def test_design_has_exact_frozen_cell_order_and_sixty_steps(self):
        steps = migrate.design_steps(self.plan)
        self.assertEqual(len(steps), 60)
        self.assertEqual([step[2] for step in steps[:4]], ["11", "01", "00", "10"])
        self.assertEqual(steps[0][0]["name"], "context_0")
        self.assertEqual(steps[4][1], self.plan["edge_pairs"][1])
        self.assertEqual(steps[20][0]["name"], "context_1")

    def test_empty_prefix_assigns_only_original_logical_shard(self):
        tasks = self.tasks([])
        self.assertEqual([(t["question_id"], t["seed"]) for t in tasks],
                         [(qid, seed) for qid in ("q0", "q3", "q6") for seed in (2024, 2025)])
        self.assertTrue(all(t["logical_shard"] == 0 for t in tasks))
        self.assertTrue(all(t["start_index"] == 0 and t["expected_new_rows"] == 60 for t in tasks))
        self.assertEqual(sum(t["expected_new_rows"] for t in tasks), len(self.rows))

    def test_full_prefix_has_no_remaining_tasks(self):
        self.assertEqual(self.tasks(self.rows), [])

    def test_complete_unit_is_not_regenerated(self):
        tasks = self.tasks(self.rows[:60])
        self.assertEqual(len(tasks), 5)
        self.assertEqual((tasks[0]["question_id"], tasks[0]["seed"]), ("q0", 2025))
        self.assertEqual(tasks[0]["start_index"], 0)
        self.assertNotIn("resume_execution_path", tasks[0])

    def test_partial_unit_resumes_exact_next_index_and_saved_execution(self):
        tasks = self.tasks(self.rows[:65])
        first = tasks[0]
        self.assertEqual((first["question_id"], first["seed"]), ("q0", 2025))
        self.assertEqual(first["start_index"], 5)
        self.assertEqual(first["expected_new_rows"], 55)
        self.assertEqual(first["resume_execution_path"], "executions/last.json")
        self.assertEqual(first["resume_execution_sha256"], self.execution_hash)
        self.assertEqual(sum(t["expected_new_rows"] for t in tasks), len(self.rows) - 65)
        self.assertTrue(all(t["start_index"] == 0 for t in tasks[1:]))

    def test_rejects_duplicate_and_reordered_keys(self):
        duplicate = copy.deepcopy(self.rows[:3])
        duplicate[1] = copy.deepcopy(duplicate[0])
        reordered = copy.deepcopy(self.rows[:3])
        reordered[0], reordered[1] = reordered[1], reordered[0]
        for rows in (duplicate, reordered):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "exact frozen-design prefix"):
                self.tasks(rows)

    def test_rejects_each_wrong_key_component(self):
        for field, bad in (("question_id", "q1"), ("seed", 42),
                           ("context_family", "wrong"), ("edge_e", "wrong"),
                           ("edge_f", "wrong"), ("cell", "10")):
            rows = copy.deepcopy(self.rows[:1])
            rows[0][field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "exact frozen-design prefix"):
                self.tasks(rows)

    def test_rejects_records_beyond_full_expected_design(self):
        with self.assertRaisesRegex(ValueError, "exact frozen-design prefix"):
            self.tasks(self.rows + [self.rows[0]])

    def test_rejects_missing_row_identity(self):
        row = copy.deepcopy(self.rows[0])
        del row["cell"]
        with self.assertRaises((KeyError, ValueError)):
            self.tasks([row])

    def test_rejects_wrong_partial_resume_hash(self):
        row = copy.deepcopy(self.rows[0])
        row["execution_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "missing or corrupt"):
            self.tasks([row])

    def test_rejects_changed_partial_resume_execution(self):
        self.execution.write_bytes(b'{"cache":"tampered"}')
        with self.assertRaisesRegex(ValueError, "missing or corrupt"):
            self.tasks(self.rows[:1])

    def test_rejects_missing_partial_resume_execution(self):
        self.execution.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.tasks(self.rows[:1])

    def test_rejects_resume_path_outside_original_shard(self):
        other = self.root / "outside.json"
        other.write_bytes(self.execution_bytes)
        row = copy.deepcopy(self.rows[0])
        row["execution_path"] = str(other)
        with self.assertRaisesRegex(ValueError, "missing or corrupt"):
            self.tasks([row])

    def test_rejects_unexpected_design_unit_size(self):
        self.plan["edge_pairs"] = self.plan["edge_pairs"][:4]
        with self.assertRaisesRegex(ValueError, "Unexpected unit size"):
            self.tasks([])


class PoolProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="elastic_pool_tests_", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_row_count_missing_and_empty_ledgers(self):
        path = self.root / "rows.jsonl"
        self.assertEqual(pool.row_count(path), 0)
        path.write_bytes(b"")
        self.assertEqual(pool.row_count(path), 0)

    def test_row_count_counts_only_newline_committed_records(self):
        path = self.root / "rows.jsonl"
        path.write_bytes(json_line({"a": 1}) + json_line({"a": 2}) + b'{"a":3')
        self.assertEqual(pool.row_count(path), 2)
        path.write_bytes(json_line({"a": 1}) + b'{"a":3}')
        self.assertEqual(pool.row_count(path), 1)

    def test_row_count_handles_record_crossing_io_chunk_boundary(self):
        path = self.root / "rows.jsonl"
        path.write_bytes(json_line({"text": "x" * (1048576 + 17)}) + json_line({"a": 2}) + b'{')
        self.assertEqual(pool.row_count(path), 2)

    def test_progress_sums_segments_without_losing_frozen_identity(self):
        originals = []
        for i in range(3):
            shard = self.root / "results/pilot" / pool.RUN_ID / f"shard_{i:03d}"
            prior = {
                "run_id": pool.RUN_ID, "logical_shard": i,
                "frozen_plan_sha256": f"frozen-plan-{i}",
                "source_manifest_sha256": "locked-source-manifest",
                "protocol_hash": "locked-protocol",
                "source_manifest": {"src/engine.py": {"sha256": "immutable"}},
                "original_frozen_plan": {"path": "frozen_plan.json", "sha256": "locked"},
                "phase": "PREVIOUS", "outputs_frozen": 999,
                "generation_complete": True, "scored": True,
            }
            originals.append(copy.deepcopy(prior))
            write_json(shard / "STATUS.json", prior)
            (shard / "raw_executions.jsonl").write_bytes(json_line({"legacy": i}) + b'{"tail":')
            (shard / "physical_calls.jsonl").write_bytes(json_line({"call": 1}) + json_line({"call": 2}))
            segment = shard / "elastic_segments" / "gpu4_unit1"
            segment.mkdir(parents=True)
            (segment / "raw_executions.jsonl").write_bytes(json_line({"new": 1}) + json_line({"new": 2}) + b'{')
            (segment / "physical_calls.jsonl").write_bytes(json_line({"call": 3}) + b'{')
            (shard / "elastic_segments" / "not_a_segment.txt").write_text("ignored")
        summaries = pool.progress(self.root)
        self.assertEqual(len(summaries), 3)
        for i, prior in enumerate(originals):
            status = json.loads((self.root / "results/pilot" / pool.RUN_ID / f"shard_{i:03d}" / "STATUS.json").read_text())
            for field in ("run_id", "logical_shard", "frozen_plan_sha256", "source_manifest_sha256",
                          "protocol_hash", "source_manifest", "original_frozen_plan"):
                self.assertEqual(status[field], prior[field], field)
            self.assertEqual(status["outputs_frozen"], 3)
            self.assertEqual(status["successful_physical_calls"], 3)
            self.assertEqual(status["scheduler_revision"], "elastic_v1")
            self.assertEqual(status["phase"], "GENERATING")
            self.assertFalse(status["generation_complete"])
            self.assertFalse(status["scored"])
            self.assertFalse(status["scientific_completion"])


if __name__ == "__main__":
    unittest.main()
