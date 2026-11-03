"""CPU-only deferred-admission tests; no subprocesses, GPU queries, or gold."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("elastic_supervisor_tested", HERE / "elastic_supervisor.py")
pool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pool
SPEC.loader.exec_module(pool)


class Process:
    def __init__(self, pid):
        self.pid = pid
        self.code = None

    def poll(self):
        return self.code


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=HERE)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.worker = {"gpu": 3, "worker_id": "gpu3", "task_manifest": "tasks_gpu3.json",
                       "deferred_qualification": True}
        self.spawned, self.started, self.queries = [], [], []
        self.available = True
        self.free = 11999

    def spawn(self, command, gpu, label):
        process = Process(8000 + len(self.spawned))
        self.spawned.append((command, gpu, label, process))
        return process

    def memory(self):
        self.queries.append(True)
        return {3: self.free}

    def controller(self, **kwargs):
        callbacks = dict(unclaimed=lambda: self.available, memory=self.memory, verify=lambda: None,
                         qualify=lambda: {"status": "PASS"}, resume=lambda: {"status": "PASS"})
        callbacks.update(kwargs)
        return pool.DeferredAdmission(self.root, {}, self.worker, self.spawn, self.started.append, **callbacks)

    def test_low_memory_does_not_launch_and_queries_are_60_seconds_apart(self):
        admission = self.controller()
        admission.tick(formal_running=True, now=0)
        admission.tick(formal_running=True, now=59)
        self.assertEqual(len(self.queries), 1)
        self.assertFalse(self.spawned)
        self.free = 12000
        admission.tick(formal_running=True, now=60)
        self.assertEqual(len(self.queries), 2)
        self.assertEqual(admission.info["phase"], "QUALIFYING")
        self.assertFalse(self.started)

    def test_start_requires_qualification_then_resume_and_immutable_source_check(self):
        self.free = 12000
        verifies = []
        admission = self.controller(verify=lambda: verifies.append(True))
        admission.tick(formal_running=True, now=0)
        self.assertEqual(len(self.spawned), 1)
        command, gpu, _, process = self.spawned[0]
        self.assertEqual(gpu, 3)
        self.assertIn("--qualification-only", command)
        self.assertIn("elastic_qualification_gpu3_20261103", command)
        self.assertNotIn("--labels", command)
        process.code = 0
        admission.tick(formal_running=True, now=1)
        self.assertEqual(admission.info["phase"], "RESUME_QUALIFYING")
        self.assertEqual(len(self.spawned), 2)
        self.assertFalse(self.started)
        self.spawned[-1][3].code = 0
        admission.tick(formal_running=True, now=2)
        self.assertEqual(self.started, [self.worker])
        self.assertEqual(admission.info["phase"], "FORMAL_WORKER_STARTED")
        self.assertIsNone(admission.info["pending_qualification_pid"])
        self.assertEqual(len(verifies), 3)

    def test_qualification_process_failure_excludes_card_without_any_retry(self):
        self.free = 12000
        admission = self.controller()
        admission.tick(formal_running=True, now=0)
        self.spawned[-1][3].code = 2
        admission.tick(formal_running=True, now=1)
        admission.tick(formal_running=True, now=3600)
        self.assertEqual(admission.info["phase"], "EXCLUDED")
        self.assertEqual(len(self.spawned), 1)
        self.assertFalse(self.started)

    def test_signature_failure_or_worker_hash_mismatch_never_starts_worker(self):
        def reject():
            raise ValueError("signature or worker SHA mismatch")
        self.free = 12000
        admission = self.controller(qualify=reject)
        admission.tick(formal_running=True, now=0)
        self.spawned[-1][3].code = 0
        admission.tick(formal_running=True, now=1)
        self.assertEqual(admission.info["phase"], "EXCLUDED")
        self.assertFalse(self.started)
        self.assertEqual(len(self.spawned), 1)

    def test_no_unclaimed_units_skips_new_qualification(self):
        self.available = False
        self.free = 50000
        admission = self.controller()
        admission.tick(formal_running=True, now=0)
        self.assertEqual(admission.info["phase"], "SKIPPED_NO_UNCLAIMED_WORK")
        self.assertFalse(self.spawned)
        self.assertFalse(self.queries)

    def test_formal_work_done_settles_existing_probe_without_starting_resume(self):
        self.free = 12000
        admission = self.controller()
        admission.tick(formal_running=True, now=0)
        admission.tick(formal_running=False, now=1)
        self.assertIsNotNone(admission.process)
        self.spawned[-1][3].code = 0
        admission.tick(formal_running=False, now=2)
        self.assertEqual(admission.info["phase"], "QUALIFIED_BUT_NO_UNCLAIMED_WORK")
        self.assertEqual(len(self.spawned), 1)
        self.assertFalse(self.started)

    def test_work_claimed_while_probe_runs_does_not_start_resume_or_formal_worker(self):
        self.free = 12000
        admission = self.controller()
        admission.tick(formal_running=True, now=0)
        self.available = False
        self.spawned[-1][3].code = 0
        admission.tick(formal_running=True, now=1)
        self.assertEqual(admission.info["phase"], "QUALIFIED_BUT_NO_UNCLAIMED_WORK")
        self.assertEqual(len(self.spawned), 1)

    def test_read_only_memory_telemetry_error_does_not_exclude_or_start(self):
        def fail_query():
            raise RuntimeError("nvidia-smi temporarily unavailable")
        admission = self.controller(memory=fail_query)
        admission.tick(formal_running=True, now=0)
        self.assertEqual(admission.info["phase"], "WAITING_FOR_FREE_MEMORY")
        self.assertFalse(self.spawned)
        self.assertIn("temporarily unavailable", admission.info["last_memory_query_error"])

    def test_existing_qualification_directory_is_never_overwritten_or_retried(self):
        (self.root / "results/pilot/elastic_qualification_gpu3_20261103").mkdir(parents=True)
        self.free = 12000
        admission = self.controller()
        admission.tick(formal_running=True, now=0)
        self.assertEqual(admission.info["phase"], "EXCLUDED")
        self.assertFalse(self.spawned)


class SharedClaimTests(unittest.TestCase):
    def test_unclaimed_lookup_matches_original_worker_unit_key(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            task = {"logical_shard": 2, "question_id": "q", "seed": 1703}
            manifest = {"claim_mode": "shared_atomic_units", "tasks": [task]}
            (root / "tasks.json").write_text(json.dumps(manifest), encoding="utf-8")
            allocation = {"claim_mode": "shared_atomic_units", "workers": [{"task_manifest": "tasks.json"}]}
            self.assertTrue(pool.has_unclaimed_units(root, allocation))
            encoded = json.dumps([2, "q", 1703], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            folder = root / "results/pilot" / pool.RUN_ID / "elastic_claims"
            folder.mkdir(parents=True)
            (folder / (hashlib.sha256(encoded).hexdigest() + ".json")).write_text("{}")
            self.assertFalse(pool.has_unclaimed_units(root, allocation))


if __name__ == "__main__":
    unittest.main()
