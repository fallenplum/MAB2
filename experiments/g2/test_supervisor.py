"""CPU-only supervisor regressions using synthetic plans and mocked children."""
from collections import Counter
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))
import supervisor as module


class FakeProcess:
    def __init__(self, pid=123, code=None):
        self.pid = pid
        self.returncode = code

    def poll(self):
        return self.returncode


class SupervisorContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=HERE.parent)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(HERE.parent))
        self.addCleanup(self.temporary.cleanup)

    def bare(self, gpus):
        supervisor = module.Supervisor.__new__(module.Supervisor)
        supervisor.gpus = gpus
        supervisor.children = []
        supervisor.gpu_uuids = {}
        supervisor.output = self.root / "out"
        supervisor.private = supervisor.output / "private"
        supervisor.private.mkdir(parents=True)
        supervisor.plan = {"run_id": "synthetic", "source_manifest_sha256": "S", "protocol_hash": "P",
                           "model": {"identity": {}}, "expected_outputs": 1,
                           "source_manifest": {}, "pipeline_source_manifest": {}}
        supervisor.plan_hash = "H"
        supervisor.pipeline_hash = "Q"
        supervisor.status = mock.Mock()
        return supervisor

    def complete_unit(self):
        output = self.root / "out"
        mask, qid, seed = [0] * 12, "synthetic", 2701
        key = module.unit_id(qid, seed)
        plan = {"units": [{"question_id": qid, "seed": seed, "mask_requests": [{"mask": mask}]}],
                "expected_outputs": 1, "protocol_hash": "P", "source_manifest_sha256": "S", "partitions": {qid: "test"}}
        claim = {"schema": "g2_collection_claim_v1", "question_id": qid, "seed": seed,
                 "worker_id": "gpu0", "plan_sha256": "H", "labels_read": False}
        claim_path = output / "claims" / (key + ".json")
        module.atomic_json(claim_path, claim)
        relative = "segments/gpu0/units/" + key + "/UNIT_OUTPUTS_FROZEN.json"
        frozen_path = output / relative
        module.atomic_json(frozen_path, {"question_id": qid, "seed": seed, "plan_sha256": "H",
                           "protocol_hash": "P", "observations": 1, "labels_read": False})
        module.atomic_json(output / "claims" / (key + ".done.json"), {
            "schema": "g2_collection_done_v1", "question_id": qid, "seed": seed, "worker_id": "gpu0",
            "unit_id": key, "plan_sha256": "H", "labels_read": False,
            "claim_sha256": module.file_sha(claim_path), "unit_frozen_path": relative,
            "unit_frozen_sha256": module.file_sha(frozen_path), "observations": 1})
        raw = output / "segments/gpu0/raw_executions.jsonl"
        row = {"question_id": qid, "seed": seed, "graph_mask": mask, "mask_sha256": module.digest(mask),
               "plan_sha256": "H", "protocol_hash": "P", "source_manifest_sha256": "S",
               "partition": "test", "quality": None, "labels_read": False}
        raw.write_text(module.canonical(row) + "\n", encoding="utf-8")
        return output, plan, raw, claim_path

    def test_complete_claim_and_raw_coverage_then_duplicate_rejected(self):
        output, plan, raw, claim = self.complete_unit()
        self.assertEqual(module.verify_collection_coverage(output, plan, "H")["unique_raw_graphs"], 1)
        raw.write_bytes(raw.read_bytes() * 2)
        with self.assertRaises(ValueError):
            module.verify_collection_coverage(output, plan, "H")

    def test_missing_raw_or_done_cannot_complete(self):
        output, plan, raw, claim = self.complete_unit()
        raw.write_bytes(b"")
        with self.assertRaises(ValueError):
            module.verify_collection_coverage(output, plan, "H")
        claim.with_suffix(".done.json").unlink()
        with self.assertRaises(ValueError):
            module.verify_collection_coverage(output, plan, "H")

    def test_tampered_claim_hash_is_rejected(self):
        output, plan, raw, claim = self.complete_unit()
        claim.write_bytes(claim.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            module.verify_collection_coverage(output, plan, "H")

    def test_low_memory_waits_before_any_child_start(self):
        supervisor = self.bare([0])
        free, started, children, cycles = [9000], [], [], []
        supervisor.resources = lambda: {0: {"free_mib": free[0]}}

        def spawn(mode, gpu):
            self.assertGreaterEqual(free[0], 12000)
            started.append(gpu)
            child = module.OwnedChild(FakeProcess(), [], 1, "synthetic", io.StringIO(), mode, gpu)
            children.append(child)
            return child

        def tick(seconds):
            cycles.append(seconds)
            if not started:
                free[0] = 13000
            else:
                children[0].process.returncode = 0

        supervisor.spawn = spawn
        with mock.patch.object(module.time, "sleep", tick):
            result = supervisor.run_pool("collect")
        self.assertEqual(started, [0])
        self.assertEqual(len(cycles), 2)
        self.assertEqual(set(result), {0})

    def test_one_worker_zero_does_not_finish_other_worker(self):
        supervisor = self.bare([0, 1])
        children, ticks = [], []
        supervisor.resources = lambda: {0: {"free_mib": 13000}, 1: {"free_mib": 13000}}

        def spawn(mode, gpu):
            child = module.OwnedChild(FakeProcess(pid=100 + gpu), [], 1, "synthetic", io.StringIO(), mode, gpu)
            children.append(child)
            return child

        def tick(seconds):
            ticks.append(seconds)
            children[0].process.returncode = 0
            if len(ticks) >= 3:
                children[1].process.returncode = 0

        supervisor.spawn = spawn
        with mock.patch.object(module.time, "sleep", tick):
            result = supervisor.run_pool("collect")
        self.assertEqual(len(ticks), 3)
        self.assertEqual(set(result), {0, 1})

    def test_missing_global_coverage_prevents_freeze_subprocess(self):
        supervisor = self.bare([0])
        supervisor.lease = contextlib.nullcontext
        supervisor.run_pool = mock.Mock(side_effect=[{0: {"signature_sha256": "a" * 64}}, {0: {"exit_code": 0}}])
        supervisor.spawn = mock.Mock()
        supervisor.stop_owned_children = mock.Mock()
        with mock.patch.object(module, "verify_collection_coverage", side_effect=ValueError("incomplete")):
            self.assertEqual(supervisor.run(), 2)
        supervisor.spawn.assert_not_called()
        supervisor.stop_owned_children.assert_called_once()

    def test_pid_creation_mismatch_refuses_signal_and_preserves_unknown(self):
        supervisor = self.bare([0])
        child = module.OwnedChild(FakeProcess(), [], 1, "synthetic", io.StringIO(), "collect", 0)
        supervisor.children = [child]
        with mock.patch.object(child, "same_identity", return_value=False), \
                mock.patch.object(module, "STOP_GRACE_SECONDS", 0), \
                mock.patch.object(module, "TERM_GRACE_SECONDS", 0), mock.patch("psutil.Process") as process:
            supervisor.stop_owned_children("synthetic failure")
            process.return_value.terminate.assert_not_called()
            process.return_value.kill.assert_not_called()
        receipt = module.read_json(supervisor.private / "STOP_RECEIPT.json")
        self.assertEqual(receipt["identity_mismatch_pids_not_signalled"], [123])
        self.assertIsNone(receipt["unverified_inflight_calls_upper_bound"])
        self.assertIsNone(receipt["unverified_inflight_cost"])

    def test_spawn_keeps_physical_gpu_mask_and_child_cuda_zero(self):
        supervisor = self.bare([7])
        supervisor.python = "synthetic_venv_python"
        supervisor.collector = Path("collector.py")
        supervisor.plan_path = Path("frozen_plan.json")
        supervisor.root = self.root
        supervisor.owner = "synthetic"
        with mock.patch.object(module.subprocess, "Popen", return_value=FakeProcess()) as popen, \
                mock.patch("psutil.Process") as process:
            process.return_value.create_time.return_value = 1
            child = supervisor.spawn("collect", 7)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], "synthetic_venv_python")
        self.assertEqual(argv[-2:], ["--device", "cuda:0"])
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "7")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertEqual(environment[name], "1")
        child.logfile.close()

    def test_uuid_binding_rejects_different_actual_gpu(self):
        supervisor = self.bare([0, 1])
        supervisor.gpu_uuids = {0: "GPU0", 1: "GPU1"}
        supervisor.children = [module.OwnedChild(FakeProcess(), [], 1, "synthetic", io.StringIO(), "collect", 0)]
        devices = {0: {"gpu_index": 0, "gpu_uuid": "GPU0", "free_mib": 20000, "compute_processes": []},
                   1: {"gpu_index": 1, "gpu_uuid": "GPU1", "free_mib": 20000,
                       "compute_processes": [{"pid": 123, "used_memory_mib": 1}]}}
        with mock.patch.object(module, "query_gpus", return_value=devices), self.assertRaises(ValueError):
            supervisor.resources()

    def test_frozen_plan_marker_and_source_hash_required(self):
        pipeline = self.root / "pipeline"
        pipeline.mkdir()
        for name in ("supervisor.py", "collector.py"):
            (pipeline / name).write_text("# synthetic source\n", encoding="utf-8")
        (self.root / "original.py").write_text("# original\n", encoding="utf-8")

        def identity(path):
            return {"bytes": path.stat().st_size, "sha256": module.file_sha(path)}

        source = {"original.py": identity(self.root / "original.py")}
        pipeline_manifest = {"pipeline/" + name: identity(pipeline / name) for name in ("supervisor.py", "collector.py")}
        output = self.root / "results/g2/synthetic"
        output.mkdir(parents=True)
        feature_prefix = "artifacts/private/g2/synthetic/"
        features = self.root / feature_prefix / "question_features.npz"
        features.parent.mkdir(parents=True)
        features.write_bytes(b"synthetic features")
        receipt_path = self.root / feature_prefix / "FEATURES_FROZEN.json"
        module.atomic_json(receipt_path, {"status": "PASS_LABEL_FREE", "labels_read": False,
                           "feature_sha256": module.file_sha(features), "pca_fit_questions": 200,
                           "dimensions": 9, "unlabeled_design_rank": 711})
        partitions = {"q" + str(i): ("train" if i < 200 else "tune" if i < 280 else "test") for i in range(400)}
        self.assertEqual(Counter(partitions.values()), {"train": 200, "tune": 80, "test": 120})
        plan = {"schema": "graph_credit_g2_plan_v1", "status": "PLAN_FROZEN", "labels_read": False,
                "run_id": "synthetic", "output_dir": "results/g2/synthetic", "source_manifest": source,
                "source_manifest_sha256": module.digest(source), "pipeline_source_manifest": pipeline_manifest,
                "features": {"path": feature_prefix + "question_features.npz", "receipt": feature_prefix + "FEATURES_FROZEN.json",
                             "sha256": module.file_sha(features), "receipt_sha256": module.file_sha(receipt_path)},
                "partitions": partitions,
                "units": [{"question_id": qid, "seed": seed, "partition": partition}
                          for qid, partition in partitions.items() for seed in range(2701, 2706)],
                "expected_units": 2000, "expected_outputs": 2000, "formal_call_upper_bound": 124000, "graph": {"rounds": 2}}
        plan_path = output / "frozen_plan.json"
        module.atomic_json(plan_path, plan)
        module.atomic_json(output / "PLAN_FROZEN.json", {
            "state": "PLAN_FROZEN", "run_id": "synthetic", "plan_sha256": module.file_sha(plan_path),
            "pipeline_source_manifest_sha256": module.digest(pipeline_manifest), "source_manifest_sha256": module.digest(source),
            "features_sha256": module.file_sha(features), "labels_read": False, "created_unix": 1})
        with mock.patch.object(module, "__file__", str(pipeline / "supervisor.py")):
            self.assertEqual(module.load_frozen_plan(self.root, plan_path)[0]["run_id"], "synthetic")
            (pipeline / "collector.py").write_text("# mutation\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                module.load_frozen_plan(self.root, plan_path)
            plan["status"] = "DESIGN_DRAFT_NO_FORMAL_GENERATION"
            module.atomic_json(plan_path, plan)
            with self.assertRaises(ValueError):
                module.load_frozen_plan(self.root, plan_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
