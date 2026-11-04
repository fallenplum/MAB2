"""Data-owner barrier, verbatim split and independent prediction regressions."""
from contextlib import contextmanager
import importlib.util
import io
import itertools
import json
from pathlib import Path, PurePosixPath
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("g2_coordinator_under_test", HERE / "coordinator.py")
coordinator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinator)


def partitions():
    return {f"q{i:03}": "train" if i < 200 else "tune" if i < 280 else "test" for i in range(400)}


def source_labels():
    return b"".join(json.dumps({"id": qid, "answer": str(i)}, separators=(",", ":")).encode() + b"\r\n"
                    for i, qid in enumerate(partitions()))


def owner_fixture():
    owner = object.__new__(coordinator.Coordinator)
    owner.output = "results/g2/" + coordinator.RUN_ID
    owner.private = "artifacts/private/g2/" + coordinator.RUN_ID
    owner.plan_hash, owner.global_hash = "plan", "global"
    owner.source_hash, owner.pipeline_hash, owner.prediction_hash = "source", "pipeline", None
    owner.coordinator_sha = "owner_control_source"
    owner.remote_root = PurePosixPath("/remote")
    owner.factory = mock.Mock()
    owner.plan = {"partitions": partitions(), "label_release": {
        "train": "data/derived/g2_v1/train_labels.jsonl", "tune": "data/derived/g2_v1/tune_labels.jsonl",
        "test": "data/derived/g2_v1/heldout_labels.jsonl"}}
    files = {"data/derived/calibration_labels.jsonl": source_labels()}
    reads, writes = [], []
    class Handle(io.BytesIO):
        def fileno(self):
            return 17
        def read(self, *args):
            if owner.output + "/LOCAL_DATA_OWNER_FIRST_LABEL_READ.json" not in files:
                raise AssertionError("Source label read preceded the persisted first-access event")
            reads.append(True)
            return super().read(*args)
    class Local:
        def __init__(self, name):
            self.name = name
        def exists(self):
            return self.name in files
        def read_bytes(self):
            if self.name not in files:
                raise FileNotFoundError(self.name)
            return files[self.name]
        def open(self, mode):
            if mode != "rb":
                raise AssertionError("Original labels were opened with write access")
            return Handle(self.read_bytes())
    def publish(name, content, *, exclusive=False):
        if exclusive and name in files and files[name] != content:
            raise coordinator.IntegrityError("immutable mismatch")
        files[name] = content
        writes.append(name)
    owner.path = Local
    owner.publish = publish
    owner.save = lambda name, value, **kwargs: publish(name, coordinator.encoded(value), **kwargs)
    owner.status = mock.Mock()
    metadata = SimpleNamespace(st_size=len(source_labels()), st_dev=1, st_ino=2, st_mtime_ns=1700000000000000000)
    return owner, files, reads, writes, metadata


def prediction_fixture():
    import numpy as np
    split = partitions()
    plan = {"run_id": coordinator.RUN_ID, "source_manifest": {}, "pipeline_source_manifest": {},
            "features": {"sha256": "features", "receipt_sha256": "receipt"}, "partitions": split, "masks": {}, "designs": {}}
    designs = []
    for anchor_id, anchor in (("a", [0] * 12), ("b", [1] * 12)):
        for edge in range(12):
            left, right = anchor.copy(), anchor.copy()
            left[edge], right[edge] = 0, 1
            left_id, right_id = "".join(map(str, left)), "".join(map(str, right))
            plan["masks"].update({left_id: left, right_id: right})
            designs.append({"anchor_id": anchor_id, "edge": edge, "mask0_id": left_id, "mask1_id": right_id})
    for qid, part in split.items():
        plan["designs"][qid] = {"contrasts": [{"question_id": qid, **row} for row in designs]}
    c1, c2 = [0.0] * 117, [0.0] * 711
    pair_weights = {(a, b): (index + 1) / 1000 for index, (a, b) in enumerate(itertools.combinations(range(12), 2))}
    for edge in range(12):
        c1[8 * 13 + 1 + edge] = edge / 10
        c1[1 + edge] = (edge + 1) / 100
        c2[8 * 79 + 1 + edge] = edge / 20
        c2[1 + edge] = (edge + 1) / 30
        c2[79 + 1 + edge] = (edge + 1) / 40
    for index, pair in enumerate(pair_weights):
        c2[8 * 79 + 13 + index] = pair_weights[pair]
    parameters = {"schema": "graph_credit_g2_predictor_v1", "seeds": [2701, 2702, 2703, 2704, 2705],
                  "train_question_ids": [qid for qid, part in split.items() if part == "train"],
                  "tune_question_ids": [qid for qid, part in split.items() if part == "tune"],
                  "B0": [edge / 100 for edge in range(12)], "models": {
                      "C1": {"order": 1, "dimensions": 117, "coefficients": c1},
                      "C2": {"order": 2, "dimensions": 711, "coefficients": c2}}}
    parameters["parameter_sha256"] = coordinator.digest(parameters)
    predictions = []
    for qid, part in split.items():
        if part != "test":
            continue
        for design in designs:
            edge, anchor = design["edge"], design["anchor_id"]
            pc1, pc2 = int(qid[1:]) / 1000, -int(qid[1:]) / 2000
            pair_effect = sum(value for pair, value in pair_weights.items() if edge in pair) if anchor == "b" else 0.0
            predictions.append({"question_id": qid, "anchor_id": anchor, "edge": edge,
                                "mask0": plan["masks"][design["mask0_id"]], "mask1": plan["masks"][design["mask1_id"]],
                                "predictions": {"B0": edge / 100, "C1": edge / 10 + pc1 * (edge + 1) / 100,
                                                "C2": edge / 20 + pair_effect + pc1 * (edge + 1) / 30 + pc2 * (edge + 1) / 40},
                                "parameter_sha256": parameters["parameter_sha256"]})
    marker = {"schema": "graph_credit_g2_predictions_frozen_v1", "state": "PREDICTIONS_FROZEN", "run_id": coordinator.RUN_ID,
              "plan_sha256": "plan", "source_manifest_sha256": coordinator.digest({}), "pipeline_source_manifest_sha256": coordinator.digest({}),
              "global_outputs_frozen_sha256": "global", "feature_sha256": "features", "features_receipt_sha256": "receipt",
              "heldout_labels_read": False, "train_questions": 200, "tune_questions": 80, "heldout_questions": 120,
              "heldout_prediction_rows": 2880, "parameters_semantic_sha256": parameters["parameter_sha256"],
              "predictions_semantic_sha256": coordinator.digest(predictions)}
    phi = {qid: np.asarray([int(qid[1:]) / 1000, -int(qid[1:]) / 2000] + [0.0] * 6 + [1.0]) for qid in split}
    return plan, marker, parameters, predictions, phi


class OwnerCoordinatorRegression(unittest.TestCase):
    def test_verbatim_split_has_exact_disjoint_frozen_ids_and_keeps_crlf(self):
        original = source_labels()
        split = coordinator.split_original_labels(original, partitions())
        self.assertEqual({part: len(raw.splitlines()) for part, raw in split.items()}, {"train": 200, "tune": 80, "test": 120})
        self.assertTrue(all(raw.endswith(b"\r\n") for raw in split.values()))
        self.assertEqual(split["train"] + split["tune"] + split["test"], original)
        with self.assertRaises(coordinator.IntegrityError):
            coordinator.split_original_labels(original + original.splitlines(keepends=True)[0], partitions())

    def test_owner_first_event_is_persisted_before_read_and_recovery_keeps_original_bytes(self):
        owner, files, reads, writes, metadata = owner_fixture()
        with mock.patch.object(coordinator.os, "fstat", return_value=metadata):
            labels, receipt = owner.owner_labels()
            first_name = owner.output + "/LOCAL_DATA_OWNER_FIRST_LABEL_READ.json"
            first = files[first_name]
            repeated_labels, repeated_receipt = owner.owner_labels()
        self.assertEqual(labels, repeated_labels)
        self.assertEqual(receipt, repeated_receipt)
        self.assertEqual(files[first_name], first)
        self.assertEqual(len(reads), 1)
        self.assertLess(writes.index(first_name), writes.index(owner.output + "/LOCAL_LABEL_SPLIT_FROZEN.json"))
        self.assertEqual(files["data/derived/calibration_labels.jsonl"], source_labels())

    def test_owner_cannot_open_gold_before_verified_global_barrier(self):
        owner, files, reads, writes, metadata = owner_fixture()
        owner.global_hash = None
        with self.assertRaises(coordinator.IntegrityError):
            owner.owner_labels()
        self.assertEqual(reads, [])
        self.assertEqual(writes, [])

    def test_train_tune_release_uploads_only_two_partitions_and_reuses_identical_event_bytes(self):
        owner, files, reads, writes, metadata = owner_fixture()
        with mock.patch.object(coordinator.os, "fstat", return_value=metadata):
            labels, split = owner.owner_labels()
        sftp = mock.Mock()
        @contextmanager
        def connected(factory):
            yield None, sftp
        with mock.patch.object(coordinator, "connection", connected), mock.patch.object(coordinator, "optional_remote", return_value=None), mock.patch.object(coordinator, "upload_once") as upload:
            first_sha = owner.release("train_tune", labels, split)
            first_raw = files[owner.output + "/TRAIN_TUNE_LABEL_RELEASE.json"]
            second_sha = owner.release("train_tune", labels, split)
        self.assertEqual(first_sha, second_sha)
        self.assertEqual(first_raw, files[owner.output + "/TRAIN_TUNE_LABEL_RELEASE.json"])
        names = [call.args[2] for call in upload.call_args_list]
        self.assertNotIn(owner.plan["label_release"]["test"], names)
        self.assertEqual(names[:3], [owner.plan["label_release"]["train"], owner.plan["label_release"]["tune"], owner.output + "/TRAIN_TUNE_LABEL_RELEASE.json"])

    def test_heldout_release_requires_independently_verified_prediction_hash(self):
        owner, files, reads, writes, metadata = owner_fixture()
        with self.assertRaises(coordinator.IntegrityError):
            owner.release("heldout", {"test": b"unopened"}, {})
        self.assertEqual(reads, [])
        self.assertEqual(writes, [])

    def test_independent_kron_replay_covers_all_2880_with_pair_context_effect(self):
        plan, marker, parameters, predictions, phi = prediction_fixture()
        report = coordinator.independent_predictions(plan, "plan", "global", marker, parameters, predictions, phi)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["coordinates"], 2880)
        self.assertLessEqual(report["maximum_absolute_error"], 1e-9)
        self.assertFalse(report["heldout_labels_read"])

    def test_semantically_rehashed_wrong_prediction_still_fails_independent_formula(self):
        plan, marker, parameters, predictions, phi = prediction_fixture()
        predictions[0]["predictions"]["C2"] += .01
        marker["predictions_semantic_sha256"] = coordinator.digest(predictions)
        with self.assertRaisesRegex(coordinator.IntegrityError, "replay differs"):
            coordinator.independent_predictions(plan, "plan", "global", marker, parameters, predictions, phi)

    def test_prediction_duplicate_coordinate_or_train_test_overlap_is_rejected(self):
        for kind in ("duplicate", "overlap"):
            with self.subTest(kind=kind):
                plan, marker, parameters, predictions, phi = prediction_fixture()
                if kind == "duplicate":
                    predictions[-1] = predictions[0]
                    marker["predictions_semantic_sha256"] = coordinator.digest(predictions)
                else:
                    parameters["train_question_ids"][0] = "q280"
                    parameters["parameter_sha256"] = coordinator.digest({key: value for key, value in parameters.items() if key != "parameter_sha256"})
                    marker["parameters_semantic_sha256"] = parameters["parameter_sha256"]
                with self.assertRaises(coordinator.IntegrityError):
                    coordinator.independent_predictions(plan, "plan", "global", marker, parameters, predictions, phi)

    def test_remote_presence_preflight_never_opens_a_label_handle(self):
        sftp = mock.Mock()
        sftp.lstat.return_value = SimpleNamespace(st_size=100)
        self.assertTrue(coordinator.remote_exists(sftp, "/remote/unreleased.jsonl"))
        sftp.open.assert_not_called()

    def test_authentication_cause_is_fatal_and_local_disk_permission_is_not_network_retry(self):
        AuthenticationException = type("AuthenticationException", (Exception,), {})
        outer = RuntimeError("inventory failed")
        outer.__cause__ = AuthenticationException()
        self.assertTrue(coordinator.fatal_connection(outer))
        self.assertFalse(coordinator.retryable_connection(PermissionError("local")))
        self.assertTrue(coordinator.retryable_connection(ConnectionResetError()))

    def test_cpu_launcher_scripts_compile_and_preserve_cpu_only_environment(self):
        compile(coordinator.CPU_ENGINE, "remote_cpu_engine", "exec")
        compile(coordinator.CPU_WRAPPER, "remote_cpu_wrapper", "exec")
        self.assertIn("'CUDA_VISIBLE_DEVICES':''", coordinator.CPU_ENGINE)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertIn("'" + name + "':'4'", coordinator.CPU_ENGINE)
        self.assertIn("LAUNCH_INTENT", coordinator.CPU_ENGINE)
        self.assertIn("INTENT_WITHOUT_IDENTIFIABLE_PROCESS", coordinator.CPU_ENGINE)
        self.assertIn("fcntl.flock", coordinator.CPU_ENGINE)

    def test_incomplete_or_label_exposed_global_cannot_pass_owner_barrier(self):
        plan = {"run_id": coordinator.RUN_ID, "source_manifest": {}, "pipeline_source_manifest": {}, "protocol_hash": "p",
                "features": {"sha256": "f", "receipt_sha256": "r"}, "expected_units": 2000, "expected_outputs": 51970}
        marker = {"schema": "graph_credit_g2_outputs_frozen_v1", "state": "GLOBAL_OUTPUTS_FROZEN", "run_id": coordinator.RUN_ID,
                  "plan_sha256": "plan", "source_manifest_sha256": coordinator.digest({}), "pipeline_source_manifest_sha256": coordinator.digest({}),
                  "protocol_hash": "p", "expected_units": 2000, "completed_units": 2000, "expected_outputs": 51970, "observations": 51970,
                  "features_sha256": "f", "features_receipt_sha256": "r", "labels_read": False, "first_gold_opened_unix": None,
                  "artifact_hashes": {"observations": "o", "physical_cost_audit": "c", "features": "f", "features_receipt": "r"},
                  "artifacts": {"observations": {"path": "results/observations.jsonl", "sha256": "o", "bytes": 1},
                                "physical_cost_audit": {"path": "results/cost.json", "sha256": "c", "bytes": 1}}}
        coordinator.validate_global(marker, plan, "plan")
        for name, value in (("completed_units", 1999), ("observations", 51969), ("labels_read", True), ("first_gold_opened_unix", 1.0)):
            with self.subTest(field=name), self.assertRaises(coordinator.IntegrityError):
                coordinator.validate_global({**marker, name: value}, plan, "plan")

    def test_late_heartbeat_or_control_cannot_downgrade_final_coordinator_stage(self):
        owner = object.__new__(coordinator.Coordinator)
        owner.lock, owner.state, owner.save = threading.RLock(), {"stage": coordinator.FINAL}, mock.Mock()
        owner.status("REMOTE_RUNNING_AND_MIRRORING")
        self.assertEqual(owner.state["stage"], coordinator.FINAL)
        owner.save.assert_not_called()

    def test_supervisor_phase_failed_stops_before_owner_gold_or_cpu_launch(self):
        owner = object.__new__(coordinator.Coordinator)
        owner.plan_hash, owner.output = "plan", "results/g2/" + coordinator.RUN_ID
        owner.factory = mock.Mock()
        owner.remote_path = lambda name: "/remote/" + name
        owner.get = mock.Mock(return_value=coordinator.encoded({"phase": "FAILED", "plan_sha256": "plan"}))
        owner.owner_labels, owner.cpu = mock.Mock(), mock.Mock()
        @contextmanager
        def connected(factory):
            yield None, mock.Mock()
        with mock.patch.object(coordinator, "connection", connected), mock.patch.object(coordinator, "remote_exists", return_value=False):
            with self.assertRaisesRegex(coordinator.IntegrityError, "reported a failure"):
                owner.cycle()
        owner.owner_labels.assert_not_called()
        owner.cpu.assert_not_called()

    def test_global_proof_keeps_first_bytes_and_time_across_restart_verifications(self):
        owner, files, reads, writes, metadata = owner_fixture()
        name = owner.output + "/LOCAL_GLOBAL_BARRIER_VERIFIED.json"
        original = {"schema": "graph_credit_g2_owner_global_barrier_v1", "status": "PASS", "plan_sha256": "plan",
                    "global_outputs_frozen_sha256": "global", "source_manifest_sha256": "source",
                    "pipeline_source_manifest_sha256": "pipeline", "observations": 51970, "units": 2000,
                    "verified_unix": 1700000000.0, "labels_read": False, "coordinator_sha256": owner.coordinator_sha}
        first_result = owner.persist_verification(name, original)
        first_bytes = files[name]
        owner.coordinator_sha = "restarted_control_source"
        repeated = {**original, "verified_unix": 1700000400.0, "coordinator_sha256": owner.coordinator_sha}
        repeated_result = owner.persist_verification(name, repeated)
        self.assertEqual(files[name], first_bytes)
        self.assertEqual(first_result, repeated_result)
        self.assertEqual(repeated_result["verified_unix"], 1700000000.0)
        attempts = [coordinator.decode(raw) for path, raw in files.items() if "/verification_attempts/" in path]
        self.assertEqual(len(attempts), 2)
        self.assertEqual([row["is_first_verification"] for row in attempts], [True, False])
        self.assertEqual([row["first_proof_sha256"] for row in attempts], [coordinator.byte_sha(first_bytes)] * 2)
        self.assertEqual(attempts[-1]["new_verification"], repeated)
        self.assertEqual(reads, [])

    def test_prediction_proof_retains_first_error_observation_and_logs_new_bounded_replay(self):
        owner, files, reads, writes, metadata = owner_fixture()
        name = owner.output + "/LOCAL_PREDICTION_REPLAY.json"
        original = {"schema": "graph_credit_g2_owner_prediction_replay_v1", "status": "PASS", "plan_sha256": "plan",
                    "predictions_frozen_sha256": "prediction", "parameters_sha256": "parameters", "predictions_sha256": "values",
                    "coordinates": 2880, "questions": 120, "maximum_absolute_error": 0.0, "absolute_tolerance": 1e-9,
                    "implementation": "independent_phi_major_kron_formula", "heldout_labels_read": False,
                    "verified_unix": 1700000000.0, "coordinator_sha256": owner.coordinator_sha}
        owner.persist_verification(name, original)
        first_bytes = files[name]
        updated = {**original, "verified_unix": 1700000500.0, "maximum_absolute_error": 1e-12}
        owner.persist_verification(name, updated)
        self.assertEqual(files[name], first_bytes)
        attempts = [coordinator.decode(raw) for path, raw in files.items() if "/verification_attempts/" in path]
        self.assertEqual(attempts[-1]["new_verification"]["maximum_absolute_error"], 1e-12)
        invalid = {**updated, "maximum_absolute_error": 1e-8}
        with self.assertRaises(coordinator.IntegrityError):
            owner.persist_verification(name, invalid)
        self.assertEqual(files[name], first_bytes)
        self.assertEqual(len([path for path in files if "/verification_attempts/" in path]), 2)

    def test_reverification_conflicting_frozen_identity_or_gate_does_not_rewrite_proof(self):
        owner, files, reads, writes, metadata = owner_fixture()
        name = owner.output + "/LOCAL_PREDICTION_REPLAY.json"
        original = {"schema": "graph_credit_g2_owner_prediction_replay_v1", "status": "PASS", "plan_sha256": "plan",
                    "predictions_frozen_sha256": "prediction", "parameters_sha256": "parameters", "predictions_sha256": "values",
                    "coordinates": 2880, "absolute_tolerance": 1e-9, "maximum_absolute_error": 0.0,
                    "verified_unix": 1700000000.0, "coordinator_sha256": owner.coordinator_sha}
        owner.persist_verification(name, original)
        first_bytes = files[name]
        for field in ("plan_sha256", "predictions_frozen_sha256", "parameters_sha256", "predictions_sha256", "schema", "coordinates", "absolute_tolerance"):
            changed = {**original, field: "foreign", "verified_unix": 1700000100.0}
            with self.subTest(field=field), self.assertRaises(coordinator.IntegrityError):
                owner.persist_verification(name, changed)
            self.assertEqual(files[name], first_bytes)
        self.assertEqual(len([path for path in files if "/verification_attempts/" in path]), 1)

    def test_first_proof_commit_failure_keeps_runtime_global_barrier_closed(self):
        owner, files, reads, writes, metadata = owner_fixture()
        owner.global_hash = None
        owner.verify_sources = mock.Mock()
        owner.persist_verification = mock.Mock(side_effect=OSError("proof commit failed"))
        owner.remote_path = lambda name: "/remote/" + name
        owner.plan["features"] = {"path": "features.npz", "sha256": coordinator.byte_sha(b"feature"),
                                  "receipt": "feature_receipt.json", "receipt_sha256": coordinator.byte_sha(b"receipt")}
        files["features.npz"], files["feature_receipt.json"] = b"feature", b"receipt"
        marker = {"artifacts": {"observations": {"path": "observations", "sha256": "ignored", "bytes": 0},
                                 "physical_cost_audit": {"path": "cost", "sha256": "ignored", "bytes": 2}}}
        owner.get = lambda sftp, name, **kwargs: b"" if name == "observations" else b"{}"
        remote = {"/remote/features.npz": b"feature", "/remote/feature_receipt.json": b"receipt",
                  "/remote/" + owner.output + "/GLOBAL_OUTPUTS_FROZEN.json": b"global"}
        @contextmanager
        def connected(factory):
            yield None, mock.Mock()
        with mock.patch.object(coordinator, "connection", connected), mock.patch.object(coordinator, "validate_global"), mock.patch.object(coordinator, "validate_observations"), mock.patch.object(coordinator, "remote_bytes", side_effect=lambda sftp, path: remote[path]):
            with self.assertRaisesRegex(OSError, "proof commit"):
                owner.verify_global_barrier(marker, coordinator.byte_sha(b"global"))
        self.assertIsNone(owner.global_hash)
        with self.assertRaises(coordinator.IntegrityError):
            owner.owner_labels()
        self.assertEqual(reads, [])

    def test_attempt_commit_interruption_recovers_without_replacing_first_proof(self):
        owner, files, reads, writes, metadata = owner_fixture()
        name = owner.output + "/LOCAL_GLOBAL_BARRIER_VERIFIED.json"
        proof = {"schema": "graph_credit_g2_owner_global_barrier_v1", "status": "PASS", "plan_sha256": "plan",
                 "global_outputs_frozen_sha256": "global", "source_manifest_sha256": "source", "pipeline_source_manifest_sha256": "pipeline",
                 "verified_unix": 1700000000.0, "coordinator_sha256": owner.coordinator_sha}
        original_save = owner.save
        def fail_attempt(path, value, **options):
            if "/verification_attempts/" in path:
                raise OSError("attempt commit interrupted")
            original_save(path, value, **options)
        owner.save = fail_attempt
        with self.assertRaisesRegex(OSError, "attempt commit"):
            owner.persist_verification(name, proof)
        first = files[name]
        owner.save = original_save
        owner.persist_verification(name, {**proof, "verified_unix": 1700000100.0})
        self.assertEqual(files[name], first)
        attempts = [coordinator.decode(raw) for path, raw in files.items() if "/verification_attempts/" in path]
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0]["is_first_verification"])
        for field in ("global_outputs_frozen_sha256", "source_manifest_sha256", "pipeline_source_manifest_sha256"):
            with self.subTest(field=field), self.assertRaises(coordinator.IntegrityError):
                owner.persist_verification(name, {**proof, field: "changed", "verified_unix": 1700000200.0})
            self.assertEqual(files[name], first)

    def test_prediction_proof_commit_failure_keeps_heldout_release_closed(self):
        owner, files, reads, writes, metadata = owner_fixture()
        owner.verify_sources, owner.feature_map = mock.Mock(), mock.Mock(return_value={})
        owner.persist_verification = mock.Mock(side_effect=OSError("prediction proof commit failed"))
        owner.remote_path = lambda name: "/remote/" + name
        marker = {"parameters_sha256": "parameters", "predictions_sha256": "predictions",
                  "first_train_tune_labels_opened_sha256": "event", "train_tune_label_release_sha256": "release",
                  "training_labels_sha256": {"train": "train", "tune": "tune"}}
        release = {"global_outputs_frozen_sha256": "global", "plan_sha256": "plan",
                   "train_labels_sha256": "train", "tune_labels_sha256": "tune"}
        contents = {"parameters.json": b"{}", "heldout_predictions.json": b"[]",
                    "FIRST_TRAIN_TUNE_LABELS_OPENED.json": b"{}", "TRAIN_TUNE_LABEL_RELEASE.json": coordinator.encoded(release)}
        owner.get = lambda sftp, path, **options: contents[path.rsplit("/", 1)[-1]]
        @contextmanager
        def connected(factory):
            yield None, mock.Mock()
        with mock.patch.object(coordinator, "connection", connected), mock.patch.object(coordinator, "remote_bytes", return_value=b"prediction"), mock.patch.object(coordinator, "independent_predictions", return_value={"status": "PASS"}):
            with self.assertRaisesRegex(OSError, "prediction proof commit"):
                owner.verify_prediction_barrier(marker, coordinator.byte_sha(b"prediction"))
        self.assertIsNone(owner.prediction_hash)
        with self.assertRaises(coordinator.IntegrityError):
            owner.release("heldout", {"test": b"unopened"}, {})
        self.assertEqual(reads, [])

    def test_status_recovers_one_transient_permission_error_with_bounded_local_retry(self):
        owner = object.__new__(coordinator.Coordinator)
        owner.lock, owner.state = threading.RLock(), {"stage": "STARTING"}
        owner.save = mock.Mock(side_effect=[PermissionError("sharing violation"), None])
        with mock.patch.object(coordinator.time, "sleep") as sleep:
            owner.status("REMOTE_RUNNING_AND_MIRRORING", marker="preserved")
        self.assertEqual(owner.save.call_count, 2)
        sleep.assert_called_once_with(.02)
        self.assertEqual(owner.state["stage"], "REMOTE_RUNNING_AND_MIRRORING")
        self.assertEqual(owner.state["marker"], "preserved")
        self.assertEqual({call.args[0] for call in owner.save.call_args_list}, {"artifacts/monitoring/g2/COORDINATOR_STATUS.json"})

    def test_persistent_status_permission_error_exits_after_five_local_attempts(self):
        owner = object.__new__(coordinator.Coordinator)
        owner.lock, owner.state = threading.RLock(), {"stage": "STARTING"}
        error = PermissionError("persistent local publication failure")
        owner.save = mock.Mock(side_effect=error)
        with mock.patch.object(coordinator.time, "sleep") as sleep:
            with self.assertRaises(PermissionError) as captured:
                owner.status("REMOTE_RUNNING_AND_MIRRORING")
        self.assertIs(captured.exception, error)
        self.assertEqual(owner.save.call_count, 5)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [.02, .04, .06, .08])
        self.assertFalse(coordinator.retryable_connection(captured.exception))

    def test_callback_failure_is_not_masked_by_a_concurrent_network_failure(self):
        cases = (([], "ConnectionError", "ConnectionError", ConnectionError),
                 ([{"phase": "progress_callback", "error_type": "PermissionError"}], "ConnectionError", "ConnectionError", coordinator.IntegrityError),
                 ([], "ConnectionError", "PermissionError", coordinator.IntegrityError),
                 ([], "PermissionError", "ConnectionError", coordinator.IntegrityError))
        for callback_errors, first_error, last_error, expected_type in cases:
            with self.subTest(callback_errors=callback_errors, first_error=first_error, last_error=last_error):
                owner = object.__new__(coordinator.Coordinator)
                owner.controls, owner.status = mock.Mock(return_value={}), mock.Mock()
                owner.claims_ready, owner.root, owner.remote_root = True, Path("local"), PurePosixPath("/remote")
                owner.plan, owner.factory, owner.owner_labels = {}, mock.Mock(), mock.Mock()
                report = {"status": "ATTENTION_REQUIRED", "failures": [{"error_type": first_error, "last_error_type": last_error}],
                          "callback_errors": callback_errors}
                with mock.patch.object(coordinator.unit_mirror, "mirror_units", return_value=report):
                    with self.assertRaises(expected_type):
                        owner.cycle()
                owner.owner_labels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
