"""Small synthetic CPU fixtures for the read-only terminal mirror verifier."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mirror_terminal_verify_under_test", HERE / "mirror_terminal_verify.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def write(root, relative, value, *, raw=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if raw else (verify.canonical(value) + "\n").encode())
    return verify.file_sha(path)


class MirrorContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mirror_cpu_tests_", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = "results/pilot/" + verify.RUN_ID

    def test_missing_file_is_waiting_and_has_root_relative_path(self):
        with self.assertRaises(verify.MirrorWaitingError) as caught:
            verify.check_hash(self.root, "missing.json", "0" * 64)
        self.assertIsInstance(caught.exception, FileNotFoundError)
        self.assertEqual(caught.exception.relative_path, "missing.json")

    def test_hash_corruption_has_retransfer_path(self):
        write(self.root, "changed.json", {})
        with self.assertRaises(verify.MirrorCorruptionError) as caught:
            verify.check_hash(self.root, "changed.json", "0" * 64)
        self.assertEqual(caught.exception.relative_path, "changed.json")

    def test_safe_paths_reject_traversal_absolute_windows_and_gold(self):
        for path in ("../escape.json", "/absolute.json", "C:/escape", "a\\b.json", "./x", "a//b", "motivation_labels.jsonl"):
            with self.subTest(path=path), self.assertRaises(verify.MirrorCorruptionError):
                verify.safe_file(self.root, path)

    def test_duplicate_json_members_rejected(self):
        write(self.root, "dupe.json", b'{"a":1,"a":2}', raw=True)
        with self.assertRaises(verify.MirrorCorruptionError):
            verify._json_file(self.root, "dupe.json")

    def test_finalization_missing_or_incomplete_waits_without_gold(self):
        with self.assertRaises(verify.MirrorWaitingError):
            verify.verify_terminal(self.root)
        write(self.root, self.run + "/ELASTIC_FINALIZATION.json", {"phase": "SCORING_FROZEN_OUTPUTS"})
        with self.assertRaises(verify.MirrorWaitingError):
            verify.verify_terminal(self.root)

    def test_old_canonical_input_hash_maps_only_to_normalized_backup(self):
        old = b'{"old":1}\n'
        write(self.root, self.run + "/shard_000/original_raw_executions.jsonl", old, raw=True)
        write(self.root, self.run + "/shard_000/raw_executions.jsonl", b'{"merged":2}\n', raw=True)
        segment = "shard_000/elastic_segments/gpu4/raw_executions.jsonl"
        segment_hash = write(self.root, self.run + "/" + segment, b'{"segment":3}\n', raw=True)
        receipt = {"input_hashes": {"shard_000/raw_executions.jsonl": verify.digest(old), segment: segment_hash}}
        verify.verify_input_hashes(self.root, self.run, receipt)
        self.assertEqual(verify.original_input_relative(segment), segment)
        write(self.root, self.run + "/" + segment, b'{"segment":4}\n', raw=True)
        with self.assertRaises(verify.MirrorCorruptionError):
            verify.verify_input_hashes(self.root, self.run, receipt)

    def test_committed_artifact_manifest_must_cover_all_canonical_and_backups(self):
        receipt = {"committed_artifact_hashes": {}}
        with self.assertRaises(verify.MirrorCorruptionError):
            verify.verify_committed(self.root, self.run, receipt)
        hashes = {relative: write(self.root, self.run + "/" + relative, {})
                  for relative in verify.required_artifacts()}
        verify.verify_committed(self.root, self.run, {"committed_artifact_hashes": hashes})
        write(self.root, self.run + "/shard_002/STATUS.json", {"changed": True})
        with self.assertRaises(verify.MirrorCorruptionError):
            verify.verify_committed(self.root, self.run, {"committed_artifact_hashes": hashes})

    def _observation_fixture(self):
        plan = {"question_ids": ["q0", "q1", "q2"], "seeds": [1701],
                "edge_pairs": [[0, 4]], "base_masks": [{"name": "full", "mask": [1] * 12}],
                "protocol_hash": "a" * 64}
        plan_hash = "b" * 64
        context_id = "full_" + verify.digest(verify.canonical([1] * 12).encode())[:12]
        rows = []
        for index in range(3):
            for cell in ("00", "01", "10", "11"):
                path = f"executions/{cell}.json"
                sha = write(self.root, f"{self.run}/shard_{index:03d}/{path}", {"synthetic": cell})
                rows.append({"question_id": f"q{index}", "context_id": context_id,
                             "context_family": "full", "edge_e": 0, "edge_f": 4, "seed": 1701,
                             "cell": cell, "quality": 1.0, "protocol_hash": plan["protocol_hash"],
                             "frozen_plan_sha256": plan_hash, "execution_path": path,
                             "execution_sha256": sha})
        self._write_observations(rows)
        return plan, plan_hash, rows

    def _write_observations(self, rows):
        write(self.root, self.run + "/all_observations.jsonl",
              b"".join((verify.canonical(row) + "\n").encode() for row in rows), raw=True)

    def test_twelve_synthetic_rows_verify_all_references_with_progress(self):
        plan, plan_hash, rows = self._observation_fixture()
        progress = []
        counts, files = verify.verify_observations(self.root, self.run, plan, plan_hash,
                                                   expected_counts=(4, 4, 4), callback=progress.append)
        self.assertEqual(counts, (4, 4, 4))
        self.assertEqual(files, 12)
        self.assertEqual(progress[-1]["observations"], 12)

    def test_duplicate_wrong_design_and_row_hash_rejected(self):
        for fault in ("duplicate", "protocol", "plan", "hash", "traversal", "key"):
            plan, plan_hash, rows = self._observation_fixture()
            if fault == "duplicate": rows[-1] = copy.deepcopy(rows[0])
            elif fault == "protocol": rows[0]["protocol_hash"] = "0" * 64
            elif fault == "plan": rows[0]["frozen_plan_sha256"] = "0" * 64
            elif fault == "hash": rows[0]["execution_sha256"] = "0" * 64
            elif fault == "traversal": rows[0]["execution_path"] = "../escape.json"
            else: rows[0]["seed"] = 999
            self._write_observations(rows)
            with self.subTest(fault=fault), self.assertRaises(verify.MirrorCorruptionError):
                verify.verify_observations(self.root, self.run, plan, plan_hash, expected_counts=(4, 4, 4))

    def test_missing_execution_is_waiting(self):
        plan, plan_hash, rows = self._observation_fixture()
        (self.root / self.run / "shard_000/executions/00.json").unlink()
        with self.assertRaises(verify.MirrorWaitingError) as caught:
            verify.verify_observations(self.root, self.run, plan, plan_hash, expected_counts=(4, 4, 4))
        self.assertTrue(caught.exception.relative_path.endswith("shard_000/executions/00.json"))

    def test_jsonl_uncommitted_tail_rejected(self):
        write(self.root, "tail.jsonl", b'{"a":1}\n{"a":2}', raw=True)
        with self.assertRaises(verify.MirrorCorruptionError):
            list(verify._jsonl(self.root, "tail.jsonl"))

    def _analysis(self, gate="INSUFFICIENT_SIGNAL"):
        return {"scientific_gate": {"status": gate}, "validation_errors": [],
                "support": {"input_rows": 60000, "exact_duplicate_rows_ignored": 0,
                            "unscored_rows": 0, "complete_question_count": 200, "incomplete_group_count": 0},
                "provenance": {"plan_hash_verified": True, "supplied_plan_sha256": "a" * 64,
                               "frozen_plan_hashes": ["a" * 64], "protocol_hashes": ["b" * 64],
                               "observations_sha256": "c" * 64,
                               "complete_frozen_design_coverage": {"verified": True, "missing_group_count": 0,
                                                                   "expected_scored_rows": 60000}}}

    def test_negative_scientific_gate_is_a_valid_completed_integrity_result(self):
        for gate in ("PASS", "INSUFFICIENT_SIGNAL"):
            write(self.root, self.run + "/G1_ANALYSIS.json", self._analysis(gate))
            self.assertEqual(verify.verify_analysis(self.root, self.run, "a" * 64, "b" * 64, "c" * 64), gate)

    def test_analysis_missing_coverage_or_wrong_input_hash_rejected(self):
        for fault in ("coverage", "observations", "support"):
            value = self._analysis()
            if fault == "coverage": value["provenance"]["complete_frozen_design_coverage"]["verified"] = False
            elif fault == "observations": value["provenance"]["observations_sha256"] = "d" * 64
            else: value["support"]["input_rows"] = 59999
            write(self.root, self.run + "/G1_ANALYSIS.json", value)
            with self.subTest(fault=fault), self.assertRaises(verify.MirrorCorruptionError):
                verify.verify_analysis(self.root, self.run, "a" * 64, "b" * 64, "c" * 64)

    def test_unfinished_analysis_is_waiting(self):
        write(self.root, self.run + "/G1_ANALYSIS.json", self._analysis("INCOMPLETE"))
        with self.assertRaises(verify.MirrorWaitingError):
            verify.verify_analysis(self.root, self.run, "a" * 64, "b" * 64, "c" * 64)

    def test_migration_normalization_retains_partial_tail(self):
        raw = b'{"ok":1}\r\n{"partial":'
        self.assertEqual(verify._normalize_backup(raw, "backup"), (1, b'{"ok":1}\n', b'{"partial":'))
        with self.assertRaises(verify.MirrorCorruptionError):
            verify._normalize_backup(b'{bad}\n{"ok":1}\n', "backup")

    def _static_fixture(self):
        sources = {}
        for relative in ("src/graph_credit/backend.py", "src/graph_credit/scoring.py",
                         "scripts/run_pilot.py", "configs/gate.json"):
            content = b"# synthetic source bytes only\n"
            sha = write(self.root, relative, content, raw=True)
            sources[relative] = {"bytes": len(content), "sha256": sha}
        model_root = "models/Qwen2.5-3B-Instruct/"
        template = "synthetic template"
        model_files = {}
        for name, content in (("model.safetensors", b"synthetic, not real weights"),
                              ("tokenizer_config.json", (verify.canonical({"chat_template": template}) + "\n").encode())):
            sha = write(self.root, model_root + name, content, raw=True)
            model_files[name] = {"sha256": sha, "bytes": len(content)}
        return {"source_manifest": sources, "source_manifest_sha256": verify.digest(verify.canonical(sources).encode()),
                "model_manifest": model_files,
                "model_identity": {"model_files": copy.deepcopy(model_files),
                                   "backend_code_sha256": sources["src/graph_credit/backend.py"]["sha256"],
                                   "tokenizer_chat_template_sha256": verify.digest(template.encode())},
                "scorer_identity": {"callable": "graph_credit.scoring:score_item",
                                    "module_sha256": sources["src/graph_credit/scoring.py"]["sha256"],
                                    "module_bytes": sources["src/graph_credit/scoring.py"]["bytes"]}}

    def test_static_guard_hashes_actual_source_and_model_bytes_without_backend_import(self):
        plan = self._static_fixture()
        before_torch = "torch" in sys.modules
        verify.verify_static_guard(self.root, plan)
        self.assertEqual("torch" in sys.modules, before_torch)
        write(self.root, "models/Qwen2.5-3B-Instruct/model.safetensors", b"tampered", raw=True)
        with self.assertRaises(verify.MirrorCorruptionError) as caught:
            verify.verify_static_guard(self.root, plan)
        self.assertEqual(caught.exception.relative_path, "models/Qwen2.5-3B-Instruct/model.safetensors")

    def test_static_guard_rejects_source_inventory_and_tokenizer_identity_drift(self):
        plan = self._static_fixture()
        write(self.root, "scripts/extra.py", b"# extra frozen code", raw=True)
        with self.assertRaises(verify.MirrorCorruptionError):
            verify.verify_static_guard(self.root, plan)
        (self.root / "scripts/extra.py").unlink()
        plan["model_identity"]["tokenizer_chat_template_sha256"] = "0" * 64
        with self.assertRaises(verify.MirrorCorruptionError):
            verify.verify_static_guard(self.root, plan)

    def test_migration_fullbyte_backups_and_original_mapping_verify_without_gold(self):
        files = []
        for index in range(3):
            for name in ("raw_executions.jsonl", "physical_calls.jsonl"):
                canonical_relative = f"{self.run}/shard_{index:03d}/{name}"
                backup_relative = f"{self.run}/shard_{index:03d}/migration_20261103/{name}.original_bytes"
                fullbytes = b'{"legacy":1}\r\n{"unfinished":'
                normalized, tail = b'{"legacy":1}\n', b'{"unfinished":'
                write(self.root, backup_relative, fullbytes, raw=True)
                write(self.root, f"{self.run}/shard_{index:03d}/original_{name}", normalized, raw=True)
                files.append({"path": canonical_relative, "backup_path": backup_relative,
                              "logical_shard": index, "backup_sha256": verify.digest(fullbytes),
                              "normalized_sha256": verify.digest(normalized), "valid_rows": 1,
                              "partial_tail_bytes": len(tail), "partial_tail_sha256": verify.digest(tail)})
        relative = "artifacts/audit/elastic_migration_20261103.json"
        migration = {"schema": "graph_credit_elastic_migration_v1", "state": "COMPLETE", "labels_read": False,
                     "run_id": verify.RUN_ID, "logical_shards": 3, "frozen_plan_sha256": "a" * 64,
                     "source_manifest_sha256": "b" * 64, "files": files}
        sha = write(self.root, relative, migration)
        receipt = {"run_id": verify.RUN_ID, "migration_receipt": {"path": relative, "sha256": sha}}
        self.assertEqual(verify.verify_migration(self.root, self.run, receipt, "a" * 64, "b" * 64), (relative, sha))
        write(self.root, files[0]["backup_path"], b"changed", raw=True)
        with self.assertRaises(verify.MirrorCorruptionError):
            verify.verify_migration(self.root, self.run, receipt, "a" * 64, "b" * 64)

    def test_allocation_binds_six_private_sources_and_task_manifest_bytes(self):
        scheduler = {relative: write(self.root, relative, b"# scheduler code fixture\n", raw=True)
                     for relative in verify.SCHEDULER_FILES}
        manifest = {"schema": "graph_credit_elastic_tasks_v1", "worker_id": "gpu4", "run_id": verify.RUN_ID,
                    "logical_shards": 3, "frozen_plan_sha256": "a" * 64,
                    "migration_receipt_sha256": "b" * 64, "scheduler_sources": scheduler,
                    "labels_read": False, "claim_mode": "shared_atomic_units", "tasks": [], "expected_new_rows": 0}
        task_relative = verify.PRIVATE + "tasks_gpu4.json"
        task_sha = write(self.root, task_relative, manifest)
        allocation = {"schema": "graph_credit_elastic_allocation_v1", "run_id": verify.RUN_ID,
                      "logical_shards": 3, "frozen_plan_sha256": "a" * 64,
                      "migration_receipt_sha256": "b" * 64, "scheduler_sources": scheduler,
                      "labels_read": False, "claim_mode": "shared_atomic_units",
                      "workers": [{"worker_id": "gpu4", "task_manifest": task_relative,
                                   "task_manifest_sha256": task_sha}]}
        relative = verify.PRIVATE + "allocation.json"
        allocation_sha = write(self.root, relative, allocation)
        receipt = {"run_id": verify.RUN_ID,
                   "scheduler_lineage": {"path": "/remote/experiment/" + relative, "sha256": allocation_sha},
                   "scheduler_input_hashes": {**scheduler, relative: allocation_sha, task_relative: task_sha},
                   "finalizer_source_sha256": scheduler[verify.PRIVATE + "elastic_finalize.py"]}
        self.assertEqual(verify.verify_allocation(self.root, receipt, "a" * 64, "b" * 64), allocation_sha)
        write(self.root, verify.PRIVATE + "elastic_worker.py", b"# changed", raw=True)
        with self.assertRaises(verify.MirrorCorruptionError) as caught:
            verify.verify_allocation(self.root, receipt, "a" * 64, "b" * 64)
        self.assertEqual(caught.exception.relative_path, verify.PRIVATE + "elastic_worker.py")


if __name__ == "__main__":
    unittest.main()
