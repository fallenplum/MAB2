"""Operational decision tests; no SSH, GPU, labels or experimental writes."""
import copy
import importlib.util
from pathlib import Path
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

spec = importlib.util.spec_from_file_location("watch_g1", Path(__file__).with_name("watch_g1.py"))
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def fixture():
    shards = []
    for index, count in enumerate((20100, 20100, 19800)):
        shards.append({"index": index, "status": {"phase": watch.COMPLETE_PHASE, "generation_complete": True,
                        "scored": True, "outputs_frozen": count, "frozen_plan_sha256": "plan"},
                       "executor_version": "unrolled_closed_input_v3", "protocol_hash": "protocol",
                       "plan_protocol_hash": "protocol", "source_manifest_sha256": "source", "plan_source_manifest_sha256": "source",
                       "files": {"frozen_plan.json": {"exists": True, "sha256": "plan"}, "physical_calls.jsonl": {"bytes": 200}},
                       "outputs_frozen": {"outputs": count}, "scoring": {"diagnostic_scoring_count": count},
                       "last_physical_call": {"status": "SUCCESS", "started_unix": 10, "elapsed_seconds": 1}})
    artifacts = {name: {"bytes": 20, "sha256": "artifact"} for name in ("all_observations.jsonl", "G1_ANALYSIS.json")}
    return {"observed_unix": 20000, "local": {"coordinator": {"stage": "REMOTE_TERMINAL_MIRRORED"},
            "shards": copy.deepcopy(shards), "artifacts": copy.deepcopy(artifacts)},
            "remote": {"connection_status": "OK", "main": {"stage": watch.REVIEW_STAGE, "expected_graph_observations": 60000},
            "shards": shards, "processes": [], "artifacts": artifacts,
            "analysis": {"support": {"input_rows": 60000}, "plan_verified": True,
                         "scientific_gate": {"status": "INSUFFICIENT_SIGNAL"}}}}


def mirror_fixture():
    data = fixture()
    plan_hash, artifact_hash, report_hash = "a" * 64, "b" * 64, "c" * 64
    for side in ("local", "remote"):
        for shard in data[side]["shards"]:
            shard["files"]["frozen_plan.json"]["sha256"] = plan_hash
            shard["status"]["frozen_plan_sha256"] = plan_hash
        for artifact in data[side]["artifacts"].values():
            artifact["sha256"] = artifact_hash
    coord = data["local"]["coordinator"]
    coord.update(schema="graph_credit_mirror_only_v1", run_id=watch.RUN_ID,
                 terminal_verification_sha256=report_hash)
    hashes = {name: artifact_hash for name in ("analysis", "all_observations", "finalization", "allocation", "migration_receipt")}
    hashes["frozen_plan"] = plan_hash
    data["local"]["mirror_verification"] = {
        "report": {"schema": "graph_credit_mirror_terminal_verification_v1", "status": "PASS",
                   "run_id": watch.RUN_ID, "observations": 60000, "execution_references_verified": 60000,
                   "execution_files_verified": 59999, "remote_stage": watch.REVIEW_STAGE,
                   "verification_scope": "local_mirror_integrity", "scientific_gate": "INSUFFICIENT_SIGNAL",
                   "scientific_completion": False, "labels_read": False, "artifact_sha256": hashes},
        "file": {"exists": True, "bytes": 1234, "sha256": report_hash}}
    data["local"]["mirror_artifacts"] = {name: {"exists": True, "sha256": value} for name, value in hashes.items()}
    return data


class MonitorDecisions(unittest.TestCase):
    def test_complete_negative_scientific_result_is_review_ready_not_scientific_pass(self):
        result = watch.classify(fixture())
        self.assertEqual(result["state"], "REVIEW_READY")
        self.assertEqual(result["scientific_gate_status"], "INSUFFICIENT_SIGNAL")
        self.assertFalse(result["scientific_completion"])

    def test_generation_without_scoring_analysis_or_mirror_is_not_completion(self):
        data = fixture()
        data["remote"]["main"]["stage"] = "G1_REAL_GENERATION"
        data["local"]["coordinator"]["stage"] = "REMOTE_RUNNING_AND_MIRRORING"
        data["remote"]["artifacts"] = {}
        self.assertEqual(watch.classify(data)["state"], "RUNNING")
        data["remote"]["main"]["stage"] = watch.REVIEW_STAGE
        self.assertEqual(watch.classify(data)["state"], "FINALIZING_MIRROR")

    def test_connection_interruption_never_reports_completion_or_model_failure(self):
        data = fixture()
        data["remote"] = {"connection_status": "CONNECTION_RETRY", "error_type": "TimeoutError"}
        result = watch.classify(data)
        self.assertEqual(result["state"], "CONNECTION_RETRY")
        self.assertFalse(result["scientific_completion"])

    def test_actual_nonterminal_probe_null_scientific_gate_is_running(self):
        data = fixture()
        data["remote"]["main"]["stage"] = "G1_REAL_GENERATION"
        data["local"]["coordinator"]["stage"] = "REMOTE_RUNNING_AND_MIRRORING"
        data["remote"]["analysis"] = {"scientific_gate": None,
                                      "support": {"input_rows": None, "question_count": None, "incomplete_group_count": None},
                                      "plan_verified": None}
        result = watch.classify(data)
        self.assertEqual(result["state"], "RUNNING")
        self.assertIsNone(result["scientific_gate_status"])
        self.assertEqual(result["reasons"], [])

    def test_optional_null_analysis_and_support_are_not_monitor_errors(self):
        for analysis in (None, {"scientific_gate": None, "support": None, "provenance": None},
                         {"scientific_gate": {}, "support": None, "provenance": {"complete_frozen_design_coverage": None}}):
            data = fixture()
            data["remote"]["main"]["stage"] = "G1_REAL_GENERATION"
            data["local"]["coordinator"]["stage"] = "REMOTE_RUNNING_AND_MIRRORING"
            data["remote"]["analysis"] = analysis
            self.assertEqual(watch.classify(data)["state"], "RUNNING")

    def test_failed_physical_call_and_source_mismatch_require_attention(self):
        for field in ("physical", "source"):
            data = fixture()
            if field == "physical":
                data["remote"]["shards"][1]["last_physical_call"]["status"] = "FAILED"
            else:
                data["remote"]["source_mismatches"] = ["src/graph_credit/executor.py"]
            self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")

    def test_dead_or_reused_active_project_pid_requires_attention(self):
        for alive, matches in ((False, True), (True, False)):
            data = fixture()
            data["remote"]["main"]["stage"] = "G1_REAL_GENERATION"
            data["remote"]["processes"] = [{"role": "main", "pid": 1234, "alive": alive, "matches_project": matches}]
            self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")

    def test_two_stagnant_checks_only_alert_when_actual_worker_abnormal(self):
        data = fixture()
        data["remote"]["main"]["stage"] = "G1_REAL_GENERATION"
        worker = {"role": "child", "shard_index": 0, "pid": 123, "alive": True, "matches_project": True,
                  "cpu_ticks": 500, "start_ticks": 10, "state": "S"}
        data["remote"]["processes"] = [worker]
        previous = {"growth": watch.growth_key(data), "consecutive_no_growth": 0,
                    "remote": {"processes": [copy.deepcopy(worker)]}}
        self.assertEqual(watch.classify(data, previous)["state"], "RUNNING")
        previous["consecutive_no_growth"] = 1
        self.assertEqual(watch.classify(data, previous)["state"], "ATTENTION_REQUIRED")
        worker["cpu_ticks"] += 100
        self.assertEqual(watch.classify(data, previous)["state"], "RUNNING")

    def test_immutable_plan_mismatch_cannot_be_hidden_by_completed_status(self):
        data = fixture()
        data["local"]["shards"][0]["files"]["frozen_plan.json"]["sha256"] = "different"
        self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")

    def test_missing_local_or_remote_plan_cannot_be_terminal_success(self):
        for side in ("local", "remote"):
            data = fixture()
            data[side]["shards"][1]["files"]["frozen_plan.json"] = {"exists": False}
            self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")

    def test_low_disk_requires_attention_without_deleting_anything(self):
        for side in ("local", "remote"):
            data = fixture()
            data[side]["disk"] = {"free_bytes": 2 * 1024**3 - 1}
            self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")
        data["remote"]["disk"]["free_bytes"] = 2 * 1024**3
        self.assertEqual(watch.classify(data)["state"], "REVIEW_READY")

    def test_mirrored_terminal_incomplete_or_pilot_analysis_requires_attention(self):
        for status in ("INCOMPLETE", "PILOT_ONLY", "INVALID"):
            data = fixture()
            data["remote"]["analysis"]["scientific_gate"]["status"] = status
            self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")
            data["local"]["coordinator"]["stage"] = "REMOTE_RUNNING_AND_MIRRORING"
            self.assertEqual(watch.classify(data)["state"], "FINALIZING_MIRROR")

    def test_mirrored_terminal_missing_artifact_or_unfinished_local_shard_requires_attention(self):
        data = fixture()
        data["local"]["artifacts"].pop("G1_ANALYSIS.json")
        self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")
        data = fixture()
        data["local"]["shards"][2]["status"]["scored"] = False
        self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")

    def test_remote_probe_compiles_without_execution(self):
        compile(watch.REMOTE_PROBE, "<remote-read-only-probe>", "exec")

    def test_mirror_only_terminal_requires_saved_bound_verifier(self):
        data = mirror_fixture()
        self.assertEqual(watch.classify(data)["state"], "REVIEW_READY")
        data["local"].pop("mirror_verification")
        self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")

    def test_missing_verifier_during_initial_mirroring_does_not_alarm(self):
        data = mirror_fixture()
        data["remote"]["main"]["stage"] = "G1_REAL_GENERATION"
        data["local"]["coordinator"]["stage"] = "REMOTE_RUNNING_AND_MIRRORING"
        data["local"].pop("mirror_verification")
        self.assertEqual(watch.classify(data)["state"], "RUNNING")

    def test_terminal_rejects_wrong_run_sha_status_or_reference_coverage(self):
        for field, value in (("run_id", "another_run"), ("status", "WAITING"),
                             ("observations", 59999), ("execution_references_verified", 59999),
                             ("remote_stage", "G1_REAL_GENERATION")):
            data = mirror_fixture()
            data["local"]["mirror_verification"]["report"][field] = value
            self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED", field)
        data = mirror_fixture()
        data["local"]["coordinator"]["terminal_verification_sha256"] = "d" * 64
        self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")

    def test_both_report_sha_aliases_must_agree(self):
        data = mirror_fixture()
        data["local"]["coordinator"]["mirror_terminal_verification_sha256"] = "c" * 64
        self.assertEqual(watch.classify(data)["state"], "REVIEW_READY")
        data["local"]["coordinator"]["mirror_terminal_verification_sha256"] = "d" * 64
        self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")

    def test_verified_report_must_bind_current_local_and_remote_artifact_bytes(self):
        for field in ("analysis", "all_observations", "frozen_plan", "finalization", "allocation", "migration_receipt"):
            data = mirror_fixture()
            data["local"]["mirror_artifacts"][field]["sha256"] = "d" * 64
            self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED", field)
        data = mirror_fixture()
        data["local"]["mirror_verification"]["report"]["scientific_gate"] = "PASS"
        self.assertEqual(watch.classify(data)["state"], "ATTENTION_REQUIRED")


class CoordinatorIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def identity(self, script, *, schema="graph_credit_mirror_only_v1", cli_root=None, created=100,
                 expected=100, owner="DOMAIN\\research", current_owner="DOMAIN\\research", cwd=None):
        coord = {"schema": schema, "process_create_time": expected, "owner_username": current_owner}
        argv = ["python.exe", "-u", str(script), "--root", str(cli_root or self.root)]
        return watch.coordinator_identity(argv, cwd or self.root, self.root, coord, created, owner, current_owner)

    def test_accepts_exact_new_and_legacy_entry_paths(self):
        result = self.identity(self.root / "artifacts/private/monitoring/mirror_only.py")
        self.assertTrue(result["matches_project"])
        self.assertEqual(result["script_identity"], "mirror_only")
        result = self.identity(self.root / "scripts/local_coordinator.py", schema="legacy")
        self.assertTrue(result["matches_project"])

    def test_rejects_same_basename_elsewhere_or_another_task_root(self):
        for script in (self.root / "other/mirror_only.py", self.root.parent / "mirror_only.py",
                       self.root / "scripts/mirror_only.py", self.root / "artifacts/private/monitoring/unrelated.py"):
            self.assertFalse(self.identity(script)["matches_project"])
        self.assertFalse(self.identity(self.root / "artifacts/private/monitoring/mirror_only.py",
                                       cli_root=self.root / "other_task")["matches_project"])

    def test_rejects_reused_pid_creation_owner_or_cwd_mismatch(self):
        script = self.root / "artifacts/private/monitoring/mirror_only.py"
        for values in ({"created": 101}, {"expected": None}, {"owner": "DOMAIN\\someone_else"},
                       {"cwd": self.root.parent}):
            self.assertFalse(self.identity(script, **values)["matches_project"])

    def test_owner_unavailable_retains_existing_permission_behavior(self):
        self.assertTrue(self.identity(self.root / "artifacts/private/monitoring/mirror_only.py",
                                      owner=None, current_owner=None)["matches_project"])

    def test_python_eval_or_duplicate_root_cannot_fake_entry_identity(self):
        script = self.root / "artifacts/private/monitoring/mirror_only.py"
        coord = {"schema": "graph_credit_mirror_only_v1", "process_create_time": 100}
        for argv in (["python", "-c", "pass", str(script), "--root", str(self.root)],
                     ["python", str(script), "--root", str(self.root), "--root", str(self.root / "other")],
                     ["python", "different.py", str(script), "--root", str(self.root)]):
            self.assertFalse(watch.coordinator_identity(argv, self.root, self.root, coord, 100)["matches_project"])

    def test_local_snapshot_inspects_new_process_and_reads_same_byte_report_hash(self):
        coord = {"schema": "graph_credit_mirror_only_v1", "pid": 4242, "process_create_time": 100,
                 "run_id": watch.RUN_ID, "owner_username": "DOMAIN\\research", "stage": "REMOTE_RUNNING_AND_MIRRORING"}
        (self.root / "COORDINATOR_STATUS.json").write_text(__import__("json").dumps(coord), encoding="utf-8")
        report = self.root / "artifacts/audit/mirror_terminal_verification.json"
        report.parent.mkdir(parents=True)
        content = b'{"status":"PASS"}\n'
        report.write_bytes(content)
        script = self.root / "artifacts/private/monitoring/mirror_only.py"
        target = SimpleNamespace(cmdline=lambda: ["python", str(script), "--root", str(self.root)],
                                 create_time=lambda: 100, is_running=lambda: True,
                                 memory_info=lambda: SimpleNamespace(rss=123), cwd=lambda: str(self.root),
                                 username=lambda: "DOMAIN\\research")
        own = SimpleNamespace(username=lambda: "DOMAIN\\research")
        fake_psutil = SimpleNamespace(Process=lambda pid: target if pid == 4242 else own)
        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            snapshot = watch.local_snapshot(self.root)
        self.assertTrue(snapshot["coordinator_process"]["matches_project"])
        self.assertEqual(snapshot["coordinator_process"]["script_identity"], "mirror_only")
        self.assertEqual(snapshot["mirror_verification"]["file"]["sha256"], __import__("hashlib").sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
