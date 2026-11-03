"""Operational decision tests; no SSH, GPU, labels or experimental writes."""
import copy
import importlib.util
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
