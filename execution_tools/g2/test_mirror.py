"""Pure-CPU completed-unit transport integrity and failure regressions."""
import gzip
import importlib.util
import io
from pathlib import Path, PurePosixPath
import tarfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g2_mirror_under_test", HERE / "mirror.py")
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)


def archive_bytes(contents, *, extra=None, omitted=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content in contents.items():
            if name == omitted:
                continue
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if extra is not None:
            info, content = extra
            archive.addfile(info, io.BytesIO(content) if content else None)
    return gzip.compress(stream.getvalue())


class UnitTransportRegression(unittest.TestCase):
    def setUp(self):
        self.unit = {"question_id": "calibration-zero", "seed": 2701, "partition": "train", "mask_requests": [
            {"mask": [0] * 12, "reference_mask": None, "aliases": [{"anchor_id": "a", "kind": "anchor"}]},
            {"mask": [1] + [0] * 11, "reference_mask": [0] * 12, "aliases": [{"anchor_id": "a", "edge": 0, "kind": "single_flip"}]},
            {"mask": [0] * 12, "reference_mask": [0] * 12, "aliases": [{"anchor_id": "b", "kind": "anchor"}]}]}
        self.plan = {"run_id": "g2_test", "output_dir": "results/g2/g2_test",
                     "qualification": {"worker_ids": ["gpu0"]}, "protocol_hash": "protocol"}
        self.uid = mirror.unit_id(self.unit["question_id"], self.unit["seed"])
        self.contents, rows = {}, []
        for item in mirror.schedule_for(self.unit):
            mask_id = "".join(map(str, item["mask"]))
            raw, execution = mirror.json_bytes({"mask_id": mask_id}), mirror.json_bytes({"mask": item["mask"]})
            self.contents["raw/" + mask_id + ".json"] = raw
            self.contents["executions/" + mask_id + ".json"] = execution
            rows.append({"mask_sha256": mirror.sha(mirror.canonical(item["mask"]).encode()),
                         "raw_path": f"units/{self.uid}/raw/{mask_id}.json", "raw_sha256": mirror.sha(raw),
                         "execution_path": f"units/{self.uid}/executions/{mask_id}.json", "execution_sha256": mirror.sha(execution)})
        self.marker = {"schema": "g2_collection_unit_outputs_v1", "question_id": self.unit["question_id"], "seed": 2701,
                       "plan_sha256": "plan", "protocol_hash": "protocol", "labels_read": False,
                       "first_gold_opened_unix": None, "observations": 2, "rows": rows, "frozen_unix": 1700000000.0}
        self.contents["UNIT_OUTPUTS_FROZEN.json"] = mirror.json_bytes(self.marker)
        self.contents["UNIT_PLAN.json"] = mirror.json_bytes(mirror.expected_unit_plan(self.unit, "plan", "protocol"))
        self.done = {"schema": "g2_collection_done_v1", "question_id": self.unit["question_id"], "seed": 2701,
                     "unit_id": self.uid, "worker_id": "gpu0", "plan_sha256": "plan", "labels_read": False, "observations": 2,
                     "claim_sha256": "a" * 64, "unit_frozen_sha256": mirror.sha(self.contents["UNIT_OUTPUTS_FROZEN.json"]),
                     "unit_frozen_path": f"segments/gpu0/units/{self.uid}/UNIT_OUTPUTS_FROZEN.json"}
        self.expected = mirror.validate_marker(self.contents["UNIT_OUTPUTS_FROZEN.json"], self.done, self.unit, "plan", "protocol")[1]

    def test_exact_archive_members_and_deduplicated_alias_schedule(self):
        self.assertEqual(mirror.validate_done(mirror.json_bytes(self.done), self.unit, self.plan, "plan"), self.done)
        self.assertEqual(mirror.validate_archive(archive_bytes(self.contents), self.expected), self.contents)
        self.assertEqual(len(self.expected), 6)
        self.assertEqual(mirror.schedule_for(self.unit)[0]["request_indices"], [0, 2])
        self.assertEqual(len(mirror.schedule_for(self.unit)[0]["aliases"]), 2)

    def test_archive_missing_member_and_sha_tamper_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "omits"):
            mirror.validate_archive(archive_bytes(self.contents, omitted="UNIT_PLAN.json"), self.expected)
        changed = {**self.contents, "UNIT_PLAN.json": b"{}\n"}
        with self.assertRaisesRegex(ValueError, "SHA"):
            mirror.validate_archive(archive_bytes(changed), self.expected)

    def test_duplicate_archive_member_is_rejected(self):
        info = tarfile.TarInfo("UNIT_PLAN.json")
        info.size = len(self.contents[info.name])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            mirror.validate_archive(archive_bytes(self.contents, extra=(info, self.contents[info.name])), self.expected)

    def test_traversal_absolute_backslash_and_unexpected_member_are_rejected(self):
        for name in ("../escape.json", "/escape.json", "raw\\escape.json", "raw/./escape.json", "raw//escape.json", "unplanned.json"):
            with self.subTest(name=name):
                info = tarfile.TarInfo(name)
                info.size = 0
                with self.assertRaises(ValueError):
                    mirror.validate_archive(archive_bytes(self.contents, extra=(info, b"")), self.expected)

    def test_links_and_devices_cannot_replace_a_regular_committed_member(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind):
                info = tarfile.TarInfo("UNIT_PLAN.json")
                info.type, info.linkname = kind, "/outside"
                with self.assertRaises(ValueError):
                    mirror.validate_archive(archive_bytes(self.contents, omitted="UNIT_PLAN.json", extra=(info, b"")), self.expected)

    def test_archive_and_member_caps_are_enforced(self):
        content = archive_bytes(self.contents)
        for name in ("MAX_COMPRESSED_BYTES", "MAX_ARCHIVE_BYTES", "MAX_MEMBER_BYTES"):
            with self.subTest(cap=name), mock.patch.object(mirror, name, 1):
                with self.assertRaises(ValueError):
                    mirror.validate_archive(content, self.expected)

    def test_done_and_marker_source_identity_mismatch_are_rejected(self):
        for field, value in (("question_id", "another"), ("seed", 2702), ("worker_id", "gpu7"),
                             ("unit_frozen_path", "../elsewhere"), ("plan_sha256", "another"), ("labels_read", True)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    mirror.validate_done(mirror.json_bytes({**self.done, field: value}), self.unit, self.plan, "plan")
        with self.assertRaisesRegex(ValueError, "bytes"):
            mirror.validate_marker(mirror.json_bytes({**self.marker, "seed": 2702}), self.done, self.unit, "plan", "protocol")

    def test_declared_row_path_cannot_escape_its_actual_unit(self):
        marker = {**self.marker, "rows": [dict(row) for row in self.marker["rows"]]}
        marker["rows"][0]["execution_path"] = "units/another/executions/000000000000.json"
        content = mirror.json_bytes(marker)
        with self.assertRaisesRegex(ValueError, "relative path"):
            mirror.validate_marker(content, {**self.done, "unit_frozen_sha256": mirror.sha(content)}, self.unit, "plan", "protocol")

    def test_duplicate_json_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            mirror.decode(b'{"labels_read":false,"labels_read":true}')

    def test_unknown_unit_inventory_cannot_be_mirrored(self):
        client, sftp = mock.Mock(), mock.Mock()
        sftp.listdir.return_value = ["f" * 20 + ".done.json"]
        with mock.patch.object(mirror, "connect", return_value=(client, sftp)):
            with self.assertRaisesRegex(ValueError, "unplanned"):
                mirror._mirror_pool(None, PurePosixPath("/remote"), {"run_id": "g2", "output_dir": "results/g2/g2"}, "plan",
                                    {self.uid: self.unit}, lambda: client, 4, None, None, "private", None)

    def test_no_completed_units_is_in_progress_and_opens_no_tar_channel(self):
        client, sftp = mock.Mock(), mock.Mock()
        sftp.listdir.return_value = [self.uid + ".json"]
        with mock.patch.object(mirror, "connect", return_value=(client, sftp)):
            summary = mirror._mirror_pool(None, PurePosixPath("/remote"), {"run_id": "g2", "output_dir": "results/g2/g2"}, "plan",
                                         {self.uid: self.unit}, lambda: client, 4, None, None, "private", None)
        self.assertEqual(summary["status"], "IN_PROGRESS")
        self.assertEqual(summary["pending"], 1)
        self.assertTrue(summary["final_full_verification_required"])
        client.exec_command.assert_not_called()

    def test_authentication_failure_is_not_retried_or_reported_as_complete(self):
        AuthenticationException = type("AuthenticationException", (Exception,), {})
        factory = mock.Mock(side_effect=AuthenticationException("failed"))
        with mock.patch.object(mirror.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "inventory"):
                mirror._mirror_pool(None, PurePosixPath("/remote"), {"run_id": "g2", "output_dir": "results/g2/g2"}, "plan",
                                    {self.uid: self.unit}, factory, 4, None, None, "private", None)
        self.assertEqual(factory.call_count, 1)
        sleep.assert_not_called()

    def source_fixture(self):
        claim = {"schema": "g2_collection_claim_v1", "question_id": self.unit["question_id"], "seed": 2701,
                 "worker_id": "gpu0", "plan_sha256": "plan", "labels_read": False}
        claim_content = mirror.json_bytes(claim)
        done = {**self.done, "claim_sha256": mirror.sha(claim_content)}
        remote_output = "/remote/" + self.plan["output_dir"]
        remote = {remote_output + "/claims/" + self.uid + ".json": claim_content,
                  remote_output + "/claims/" + self.uid + ".done.json": mirror.json_bytes(done),
                  remote_output + "/" + done["unit_frozen_path"]: self.contents["UNIT_OUTPUTS_FROZEN.json"]}
        return remote_output, remote

    def test_complete_unit_publishes_receipt_last_and_same_source_skips_tar(self):
        remote_output, remote = self.source_fixture()
        published, order = {}, []
        class LocalFile:
            def __init__(self, name):
                self.name = name
            def read_bytes(self):
                if self.name not in published:
                    raise FileNotFoundError(self.name)
                return published[self.name]
        def publish(name, content):
            published[name] = content
            order.append(name)
        with mock.patch.object(mirror, "remote_file", side_effect=lambda sftp, name: remote[name]) as metadata:
            with mock.patch.object(mirror, "remote_tar", return_value=archive_bytes(self.contents)) as tar:
                result = mirror._mirror_one(self.uid, self.unit, self.plan, "plan", remote_output, None, None,
                                            LocalFile, publish, "private")
                self.assertFalse(result["cached"])
                self.assertEqual(result["members_committed"], 6)
                self.assertTrue(order[-1].endswith("/mirrored_units/" + self.uid + ".json"))
                tar.assert_called_once()
            metadata.reset_mock()
            with mock.patch.object(mirror, "remote_tar") as tar:
                repeated = mirror._mirror_one(self.uid, self.unit, self.plan, "plan", remote_output, None, None,
                                              LocalFile, publish, "private")
                self.assertTrue(repeated["cached"])
                self.assertEqual(repeated["tar_bytes"], 0)
                self.assertEqual(metadata.call_count, 1)
                tar.assert_not_called()

    def test_unit_with_corrupted_tar_publishes_no_local_member_or_done(self):
        remote_output, remote = self.source_fixture()
        def local(name):
            handle = mock.Mock()
            handle.read_bytes.side_effect = FileNotFoundError(name)
            return handle
        corrupted = {**self.contents, "UNIT_PLAN.json": b"{}\n"}
        with mock.patch.object(mirror, "remote_file", side_effect=lambda sftp, name: remote[name]), mock.patch.object(mirror, "remote_tar", return_value=archive_bytes(corrupted)):
            publish = mock.Mock()
            with self.assertRaisesRegex(ValueError, "SHA"):
                mirror._mirror_one(self.uid, self.unit, self.plan, "plan", remote_output, None, None,
                                   local, publish, "private")
            publish.assert_not_called()

    def test_worker_connection_failure_is_bounded_and_keeps_completed_units_unmirrored(self):
        client, sftp = mock.Mock(), mock.Mock()
        sftp.listdir.return_value = [self.uid + ".done.json"]
        outcomes = [(client, sftp), ConnectionError("first"), ConnectionError("second"), ConnectionError("third")]
        with mock.patch.object(mirror, "connect", side_effect=outcomes) as connect, mock.patch.object(mirror.time, "sleep"):
            result = mirror._mirror_pool(None, PurePosixPath("/remote"), self.plan, "plan", {self.uid: self.unit},
                                         lambda: client, 1, None, None, "private", None)
        self.assertEqual(connect.call_count, 4)
        self.assertEqual(result["status"], "ATTENTION_REQUIRED")
        self.assertEqual(result["unmirrored_completed_units"], 1)
        self.assertEqual(result["failures"][0]["error_type"], "ConnectionError")
        self.assertEqual(result["failures"][0]["attempts"], 3)

    def test_callback_io_error_does_not_retry_or_settle_the_committed_unit_twice(self):
        for error in (OSError("status publication failed"), PermissionError("status sharing violation")):
            with self.subTest(error_type=type(error).__name__):
                client, sftp = mock.Mock(), mock.Mock()
                sftp.listdir.return_value = [self.uid + ".done.json"]
                callback = mock.Mock(side_effect=[error, None])
                result = {"cached": True, "tar_bytes": 0, "members_committed": 0}
                with mock.patch.object(mirror, "connect", return_value=(client, sftp)), mock.patch.object(mirror, "_mirror_one", return_value=result) as transfer:
                    summary = mirror._mirror_pool(None, PurePosixPath("/remote"), self.plan, "plan", {self.uid: self.unit},
                                                  lambda: client, 4, None, None, "private", callback)
                transfer.assert_called_once()
                callback.assert_called_once()
                self.assertEqual(summary["remote_completed_units"], 1)
                self.assertEqual(summary["skipped_cached"], 1)
                self.assertEqual(summary["settled_units"], 1)
                self.assertEqual(summary["unmirrored_completed_units"], 0)
                self.assertEqual(summary["failures"], [])
                self.assertEqual(summary["status"], "ATTENTION_REQUIRED")
                self.assertEqual(summary["callback_errors"][0]["phase"], "progress_callback")
                self.assertEqual(summary["callback_errors"][0]["error_type"], type(error).__name__)
                self.assertTrue(summary["callback_errors"][0]["unit_settled"])
                self.assertTrue(summary["callback_errors"][0]["traceback_frames"])

    def test_completed_inventory_growth_during_transfer_remains_a_partial_snapshot(self):
        client, sftp = mock.Mock(), mock.Mock()
        another = {**self.unit, "seed": 2702}
        another_id = mirror.unit_id(another["question_id"], another["seed"])
        names = [self.uid + ".done.json"]
        sftp.listdir.return_value = names
        def transfer(*args):
            names.append(another_id + ".done.json")
            return {"cached": True, "tar_bytes": 0, "members_committed": 0}
        with mock.patch.object(mirror, "connect", return_value=(client, sftp)), mock.patch.object(mirror, "_mirror_one", side_effect=transfer) as copy:
            summary = mirror._mirror_pool(None, PurePosixPath("/remote"), self.plan, "plan",
                                          {self.uid: self.unit, another_id: another}, lambda: client, 4,
                                          None, None, "private", None)
        copy.assert_called_once()
        self.assertEqual(len(names), 2)
        self.assertEqual(summary["remote_completed_units"], 1)
        self.assertEqual(summary["skipped_cached"], 1)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["unmirrored_completed_units"], 0)
        self.assertEqual(summary["status"], "IN_PROGRESS")

    def test_duplicate_inventory_unit_names_are_rejected_before_transfer(self):
        client, sftp = mock.Mock(), mock.Mock()
        sftp.listdir.return_value = [self.uid + ".done.json"] * 2
        with mock.patch.object(mirror, "connect", return_value=(client, sftp)), mock.patch.object(mirror, "_mirror_one") as transfer:
            with self.assertRaisesRegex(ValueError, "repeats"):
                mirror._mirror_pool(None, PurePosixPath("/remote"), self.plan, "plan", {self.uid: self.unit},
                                    lambda: client, 4, None, None, "private", None)
        transfer.assert_not_called()

    def test_multithreaded_callback_failure_preserves_distinct_unit_accounting(self):
        units = {}
        for index in range(20):
            unit = {**self.unit, "seed": 2701 + index}
            units[mirror.unit_id(unit["question_id"], unit["seed"])] = unit
        client, sftp = mock.Mock(), mock.Mock()
        sftp.listdir.return_value = [uid + ".done.json" for uid in units]
        callback = mock.Mock(side_effect=PermissionError("persistent status error"))
        with mock.patch.object(mirror, "connect", return_value=(client, sftp)), mock.patch.object(mirror, "_mirror_one", return_value={"cached": True, "tar_bytes": 0, "members_committed": 0}) as transfer:
            summary = mirror._mirror_pool(None, PurePosixPath("/remote"), self.plan, "plan", units,
                                          lambda: client, 4, None, None, "private", callback)
        called_ids = [call.args[0] for call in transfer.call_args_list]
        self.assertEqual(len(called_ids), len(set(called_ids)))
        self.assertEqual(summary["skipped_cached"], len(called_ids))
        self.assertEqual(summary["settled_units"], len(called_ids))
        self.assertEqual(summary["unmirrored_completed_units"], 20 - len(called_ids))
        self.assertGreaterEqual(summary["unmirrored_completed_units"], 0)
        self.assertEqual(summary["status"], "ATTENTION_REQUIRED")


if __name__ == "__main__":
    unittest.main()
