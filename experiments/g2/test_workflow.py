"""Persisted synthetic regressions previously run for the CPU G2 workflow.

Fixtures use 400 synthetic IDs, synthetic labels and the label-free original
scorer source. The predictor fake isolates stage orchestration from the
mathematical predictor regressions in test_predictor.py.
"""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
STAGE = HERE.parent
SCORER_REFERENCE = STAGE / "acceptance_reference" / "scoring.py"
if not SCORER_REFERENCE.is_file():
    SCORER_REFERENCE = REPO_ROOT / "src" / "graph_credit" / "scoring.py"
SPEC = importlib.util.spec_from_file_location("g2_workflow_under_test", HERE / "workflow.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class FakePredictor:
    digest = staticmethod(workflow.digest)

    @staticmethod
    def fit_and_select(train, tune, phi, masks, contrasts):
        value = {
            "train_question_ids": [f"train{i}" for i in range(200)],
            "tune_question_ids": [f"tune{i}" for i in range(80)],
        }
        value["parameter_sha256"] = workflow.digest(value)
        return value

    @staticmethod
    def predict(parameters, phi, masks, contrasts):
        return [
            {
                "question_id": row["question_id"],
                "anchor_id": row["anchor_id"],
                "edge": row["edge"],
                "predictions": {"B0": 0.0, "C1": 0.0, "C2": 0.0},
                "parameter_sha256": parameters["parameter_sha256"],
            }
            for row in contrasts
        ]

    @staticmethod
    def evaluate(test, predictions, masks, *, prediction_freeze):
        assert prediction_freeze["state"] == "PREDICTIONS_FROZEN"
        assert prediction_freeze["predictions_sha256"] == workflow.digest(predictions)
        assert len({row["question_id"] for row in test}) == 120
        return {
            "schema": "synthetic_eval",
            "scientific_gate": "INSUFFICIENT_PREDICTIVE_GAIN",
            "primary": {"question_count": 120},
            "contrasts_per_question": 24,
        }


class WorkflowProductionContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="g2wf_", dir=STAGE)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_stage_recovery_idempotence_and_first_label_read_order(self):
        root = self.root
        (root / "out").mkdir()
        partitions = {
            f"{part}{index}": part
            for part, count in workflow.COUNTS.items()
            for index in range(count)
        }
        designs = {
            qid: {
                "contrasts": [
                    {"question_id": qid, "anchor_id": anchor, "edge": edge}
                    for anchor in ("p025", "p075")
                    for edge in range(12)
                ]
            }
            for qid in partitions
        }
        plan = {
            "run_id": "synthetic_run",
            "partitions": partitions,
            "designs": designs,
            "masks": {},
            "source_manifest": {"src/graph_credit/scoring.py": {}},
            "label_release": {
                part: f"data/{part}_labels.jsonl" for part in workflow.COUNTS
            },
        }
        scorer = root / "src/graph_credit/scoring.py"
        scorer.parent.mkdir(parents=True)
        scorer.write_bytes(SCORER_REFERENCE.read_bytes())
        context = {
            "root": root,
            "plan": plan,
            "output": "out",
            "plan_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "pipeline_sha256": "c" * 64,
        }
        global_marker = {
            "artifacts": {
                "physical_cost_audit": {"path": "unused", "sha256": "d" * 64, "bytes": 0}
            }
        }
        global_sha = workflow.atomic_json(
            root, "out/GLOBAL_OUTPUTS_FROZEN.json", global_marker
        )
        global_data = {
            "marker": global_marker,
            "marker_sha256": global_sha,
            "feature_sha256": "e" * 64,
            "feature_receipt_sha256": "f" * 64,
            "phi": {},
            "physical_cost_audit": {"formal_calls": 0},
            "rows": [
                {
                    "question_id": qid,
                    "partition": partition,
                    "service_output": '{"answer":"4"}',
                    "seed": 2701,
                    "graph_mask": [0] * 12,
                    "mask_id": "0" * 12,
                }
                for qid, partition in partitions.items()
            ],
        }
        label_hashes = {}
        for partition in workflow.COUNTS:
            content = b"".join(
                (json.dumps({"id": qid, "task": "gsm8k", "answer": "4"}) + "\n").encode()
                for qid, part in partitions.items()
                if part == partition
            )
            path = root / plan["label_release"][partition]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            label_hashes[partition] = hashlib.sha256(content).hexdigest()
        workflow.atomic_json(root, "out/TRAIN_TUNE_LABEL_RELEASE.json", {
            "state": "TRAIN_TUNE_LABELS_RELEASED",
            "plan_sha256": "a" * 64,
            "global_outputs_frozen_sha256": global_sha,
            "released_unix": 1.0,
            "train_labels_sha256": label_hashes["train"],
            "tune_labels_sha256": label_hashes["tune"],
        })
        original_open = Path.open
        reads = []
        test_case = self

        class GuardedLabelHandle:
            def __init__(self, handle, path):
                self.handle, self.path = handle, path

            def __enter__(self):
                return self

            def __exit__(self, *arguments):
                return self.handle.__exit__(*arguments)

            def __getattr__(self, name):
                return getattr(self.handle, name)

            def read(self, *arguments):
                event = (
                    "FIRST_HELDOUT_LABELS_OPENED.json"
                    if self.path.name == "test_labels.jsonl"
                    else "FIRST_TRAIN_TUNE_LABELS_OPENED.json"
                )
                test_case.assertTrue((root / "out" / event).exists())
                reads.append(self.path.name)
                return self.handle.read(*arguments)

        def guarded_open(path, *arguments, **keywords):
            handle = original_open(path, *arguments, **keywords)
            mode = arguments[0] if arguments else keywords.get("mode", "r")
            return (
                GuardedLabelHandle(handle, path)
                if path.name.endswith("_labels.jsonl") and mode == "rb"
                else handle
            )

        with mock.patch.object(workflow, "verify_context_unchanged", return_value=None), \
                mock.patch.object(Path, "open", guarded_open):
            with mock.patch.object(FakePredictor, "fit_and_select", side_effect=RuntimeError("synthetic CPU interruption")):
                with self.assertRaisesRegex(RuntimeError, "CPU interruption"):
                    workflow.predict_stage(context, global_data, FakePredictor)
            first = (root / "out/FIRST_TRAIN_TUNE_LABELS_OPENED.json").read_bytes()
            predicted = workflow.predict_stage(context, global_data, FakePredictor)
            self.assertEqual(predicted["state"], "PREDICTIONS_FROZEN_AWAITING_HELDOUT_LABELS")
            self.assertEqual((root / "out/FIRST_TRAIN_TUNE_LABELS_OPENED.json").read_bytes(), first)
            previous_reads = len(reads)
            with mock.patch.object(FakePredictor, "fit_and_select", side_effect=AssertionError("Completed predictor was refit")):
                workflow.predict_stage(context, global_data, FakePredictor)
            self.assertEqual(len(reads), previous_reads)
            self.assertNotIn("test_labels.jsonl", reads)
            prediction_sha = workflow.file_sha(root, "out/PREDICTIONS_FROZEN.json")
            workflow.atomic_json(root, "out/HELDOUT_LABEL_RELEASE.json", {
                "state": "HELDOUT_LABELS_RELEASED",
                "plan_sha256": "a" * 64,
                "predictions_frozen_sha256": prediction_sha,
                "heldout_labels_sha256": label_hashes["test"],
                "released_unix": 2.0,
            })
            completed = workflow.evaluate_stage(context, global_data, FakePredictor)
            self.assertEqual(completed["state"], "G2_EVALUATION_COMPLETE_REVIEW_REQUIRED")
            heldout_event = (root / "out/FIRST_HELDOUT_LABELS_OPENED.json").read_bytes()
            analysis = (root / "out/G2_ANALYSIS.json").read_bytes()
            replay = workflow.evaluate_stage(context, global_data, FakePredictor)
            self.assertTrue(replay["deterministic_replay_verified"])
            self.assertEqual((root / "out/FIRST_HELDOUT_LABELS_OPENED.json").read_bytes(), heldout_event)
            self.assertEqual((root / "out/G2_ANALYSIS.json").read_bytes(), analysis)
            self.assertEqual(workflow.predict_stage(context, global_data, FakePredictor)["state"],
                             "G2_EVALUATION_COMPLETE_REVIEW_REQUIRED")
            with self.assertRaises(workflow.WorkflowError):
                workflow._labels_after_event(
                    context, ("test",), {"test": "0" * 64},
                    "FIRST_HELDOUT_LABELS_OPENED.json", {"predictions_frozen_sha256": prediction_sha}
                )

    def test_raw_execution_hash_guard_and_unknown_physical_cost(self):
        root = self.root
        mask = [0] * 12
        key = ("q", 2701, tuple(mask))
        context = {
            "root": root,
            "plan_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "expected": {key: "train"},
            "plan": {
                "protocol_hash": "c" * 64,
                "questions": {"payload_sha256": {"q": "d" * 64}},
            },
        }
        base = "out/segments/gpu0"
        raw_path = base + "/units/u/raw/m.json"
        execution_path = base + "/units/u/executions/m.json"
        execution = {
            "question_id": "q", "seed": 2701, "mask": mask, "protocol_hash": "c" * 64,
            "execution_id": "x", "question_fingerprint": "d" * 64,
            "nodes": {"aggregate": {"generation": {"text": '{"answer":"4"}'}}},
        }
        execution_sha = workflow.atomic_json(root, execution_path, execution)
        raw = {
            "question_id": "q", "seed": 2701, "graph_mask": mask, "partition": "train",
            "quality": None, "labels_read": False, "plan_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64, "protocol_hash": "c" * 64,
            "mask_id": "0" * 12, "mask_sha256": workflow.digest(mask), "execution_id": "x",
            "service_output": '{"answer":"4"}', "execution_path": "units/u/executions/m.json",
            "execution_sha256": execution_sha,
        }
        raw_sha = workflow.atomic_json(root, raw_path, raw)
        row = {**raw, "raw_path": raw_path, "raw_sha256": raw_sha, "execution_path": execution_path}
        self.assertEqual(workflow._validate_graph(context, row), key)
        with self.assertRaises(workflow.WorkflowError):
            workflow._validate_graph(context, {**row, "execution_sha256": "0" * 64})
        ledger = "out/ledger.jsonl"
        content = (json.dumps({"call_id": "c1", "status": "FAILED", "input_tokens": 7,
                               "output_tokens": None, "gpu_seconds": None, "elapsed_seconds": .5}) + "\n").encode()
        (root / ledger).write_bytes(content)
        result = workflow.generation_cost_summary(context, {
            "physical_cost_audit": {
                "formal_calls": 1,
                "physical_ledgers": [{"path": ledger, "sha256": hashlib.sha256(content).hexdigest()}],
            }
        })
        self.assertIsNone(result["formal"]["metrics"]["output_tokens"]["complete_total"])
        self.assertEqual(result["formal"]["metrics"]["input_tokens"]["complete_total"], 7)


if __name__ == "__main__":
    unittest.main()
