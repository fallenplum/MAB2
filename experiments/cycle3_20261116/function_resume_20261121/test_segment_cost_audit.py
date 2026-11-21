"""Exercise the cost reconciliation on synthetic request traces."""
import json
from pathlib import Path
import tempfile
import unittest

from segment_cost_audit import digest, summarize


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class SegmentCostTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        worker = self.root / "worker_0"
        worker.mkdir()
        unit = worker / "unit_000"
        unit.mkdir()
        self.log = worker / "qwen_requests.jsonl"
        self.close = self.root / "CLOSURE.json"
        save(self.close, {"status": "PARTIAL_OR_FAILED", "all_workers_exited": True, "quality_seen": False,
                          "actual_GPU_hours": 0.5, "residence_seconds_by_worker": [1800]})
        self.generation = {"call_id": "native-1", "input_tokens": 3, "output_tokens": 2, "gpu_seconds": 0.1}
        self.start = {"event": "START", "request_id": "journal-1", "model": "qwen",
                      "context": {"question_fingerprint": "task-1", "request_fingerprint": "request",
                                  "model": "qwen", "request_id": "trace-1"}}
        self.end = dict(self.start, event="RETURNED", generation=self.generation,
                        native_events=[dict(self.generation, status="SUCCESS")])
        self.other = dict(self.start, request_id="journal-2", context=dict(
            self.start["context"], question_fingerprint="task-2", request_id="trace-2"))
        m = unit / "MATRIX.json"
        save(m, {"hidden_test_access": False, "question_fingerprint": "task-1", "conditions": [{}],
                 "request_trace": [{"request_id": "trace-1", "model": "qwen",
                                    "request_fingerprint": "request", "generation": self.generation}]})
        save(unit / "COMPLETE.json", {"matrix_sha256": digest(m)})

    def run_audit(self, *rows):
        header = {"event": "HEADER", "model": "qwen"}
        self.log.write_text("\n".join(json.dumps(r) for r in (header, *rows)), encoding="utf-8")
        manifest = self.root / "ARCHIVE_MANIFEST.json"
        save(manifest, {f.relative_to(self.root).as_posix(): digest(f) for f in self.root.rglob("*")
                        if f.is_file() and f != manifest})
        return summarize(self.root, digest(manifest))

    def test_known_cost_and_unresolved_cost(self):
        audit = self.run_audit(self.start, self.end, self.other)
        self.assertEqual(audit["matrix_requests_matched"], 1)
        self.assertEqual(audit["physical_journal_counts"]["completed_tasks"]["output_tokens"], 2)
        self.assertEqual(audit["physical_journal_counts"]["interrupted_tasks"], {"started": 1, "unresolved_starts": 1})

    def test_orphan_return(self):
        with self.assertRaisesRegex(ValueError, "Orphan"):
            self.run_audit(self.end)

    def test_unmatched_complete_trace(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.run_audit(self.start)

    def test_conflicting_native_count(self):
        self.end["native_events"][0]["output_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "Native generation differs"):
            self.run_audit(self.start, self.end)

    def test_complete_closure_with_interrupted_task(self):
        closure = json.loads(self.close.read_bytes())
        save(self.close, dict(closure, status="COMPLETE_UNSCORED"))
        with self.assertRaisesRegex(ValueError, "interrupted"):
            self.run_audit(self.start, self.end, self.other)


if __name__ == "__main__":
    unittest.main()
