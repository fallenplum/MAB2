"""CPU-only mirror transport contracts; all SSH/SFTP objects are fakes."""
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mirror_only", HERE / "mirror_only.py")
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)


class Source(io.BytesIO):
    def __init__(self, data, opened_size):
        super().__init__(data)
        self.opened_size = opened_size
        self.prefetch_limit = None
        self.read_sizes = []
    def stat(self):
        return SimpleNamespace(st_size=self.opened_size, st_mtime=100)
    def prefetch(self, file_size, max_concurrent_requests):
        self.prefetch_limit = max_concurrent_requests
        assert file_size == self.opened_size
    def read(self, count=-1):
        self.read_sizes.append(count)
        return super().read(count)


class SFTP:
    def __init__(self, data, opened_size=None):
        self.data = data
        self.source = Source(data, len(data) if opened_size is None else opened_size)
    def open(self, path, mode):
        assert mode == "rb"  # Remote write modes are forbidden by the fake.
        return self.source
    def stat(self, path):
        return SimpleNamespace(st_size=len(self.data), st_mtime=100)


def instance(root):
    return mirror.Mirror(SimpleNamespace(root=root, host="unused", port=1, user="unused",
                       remote_root="/unused", remote_python="python3", known_hosts=root / "unused"))


class MirrorOnlyContracts(unittest.TestCase):
    def test_growing_file_copies_only_open_size_without_chasing_eof(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "snapshot"
            sftp = SFTP(b'{"a":1}\n{"b":2}\n', opened_size=8)
            progress = []
            result = mirror.fixed_copy(sftp, "/remote/raw.jsonl", path,
                                       lambda delta, received, total: progress.append((delta, received, total)), is_jsonl=True)
            self.assertEqual(path.read_bytes(), b'{"a":1}\n')
            self.assertEqual(result["opened_bytes"], 8)
            self.assertTrue(result["source_changed"])
            self.assertEqual(sum(x[0] for x in progress), 8)
            self.assertEqual(sftp.source.prefetch_limit, 16)
            self.assertEqual(sftp.source.read_sizes[-1], 1)

    def test_shrinking_or_changed_immutable_file_never_reports_success(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "snapshot"
            for sftp, immutable in ((SFTP(b"short", opened_size=10), False), (SFTP(b"old-new", opened_size=3), True)):
                with self.assertRaises(EOFError):
                    mirror.fixed_copy(sftp, "/remote/file", path, lambda *x: None, is_immutable=immutable)

    def test_incomplete_final_jsonl_record_is_not_published_as_committed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "snapshot"
            sftp = SFTP(b'{"a":1}\n{"partial":')
            result = mirror.fixed_copy(sftp, "/remote/raw.jsonl", path, lambda *x: None, is_jsonl=True)
            self.assertEqual(path.read_bytes(), b'{"a":1}\n')
            self.assertEqual(result["committed_bytes"], 8)
            self.assertTrue(result["source_changed"])

    def test_immutable_cache_avoids_repeat_local_stat_but_mutable_does_not(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            obj = instance(root)
            relative = "results/pilot/current/executions/one.json"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"{}")
            os.utime(path, (100, 100))
            item = [relative, 2, 100_000_000_000, 1, 1]
            self.assertFalse(obj.needs_transfer(item))
            with mock.patch.object(Path, "stat", side_effect=AssertionError("unexpected repeated stat")):
                self.assertFalse(obj.needs_transfer(item))
            obj.force_paths.add(relative)
            self.assertTrue(obj.needs_transfer(item))
            mutable = root / "report.json"
            mutable.write_bytes(b"{}")
            os.utime(mutable, (100, 100))
            with mock.patch.object(Path, "stat", side_effect=AssertionError("mutable really statted")):
                with self.assertRaises(AssertionError):
                    obj.needs_transfer(["report.json", 2, 100_000_000_000, 1, 1])

    def test_all_temporary_suffixes_and_path_traversal_are_rejected(self):
        for suffix in mirror.TEMP_SUFFIXES:
            self.assertTrue(mirror.is_temporary("status" + suffix))
        self.assertTrue(mirror.is_temporary("file.abc012.tmp"))
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            for name in ("../escape", "/absolute", "C:/absolute", "dir\\file"):
                with self.assertRaises(ValueError):
                    mirror.local_path(Path(temp), name)

    def test_probe_only_writes_report_never_data_status_upload_or_launch(self):
        class Client:
            def close(self): pass
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            obj = instance(root)
            obj.connect = lambda: Client()
            obj.execute = lambda client, code: ({"files": {"RUN_STATUS.json": {"value": {"stage": "G1_REAL_GENERATION"}}}}
                                                if code == mirror.CONTROL else {"files": [["file.json", 2, 100, 1, 1]]})
            with mock.patch.object(mirror, "fixed_copy", side_effect=AssertionError("probe copied data")):
                report = obj.probe()
            self.assertEqual(report["files_copied"], 0)
            self.assertEqual(report["experiment_launches"], 0)
            self.assertFalse((root / "COORDINATOR_STATUS.json").exists())
            self.assertTrue((root / "artifacts/private/mirroring/probe_latest.json").is_file())

    def test_live_legacy_publisher_is_refused_before_network_or_threads(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            obj = instance(Path(temp))
            with mock.patch.object(mirror, "legacy_process", return_value={"legacy_alive": True}), \
                    mock.patch.object(obj, "connect", side_effect=AssertionError("unexpected connect")):
                with self.assertRaisesRegex(RuntimeError, "LEGACY_COORDINATOR_ALIVE"):
                    obj.run()

    def test_read_only_remote_programs_compile(self):
        compile(mirror.CONTROL, "<control-read-only>", "exec")
        compile(mirror.INVENTORY, "<inventory-read-only>", "exec")
        self.assertNotIn("subprocess", mirror.CONTROL + mirror.INVENTORY)
        self.assertNotIn(".write", mirror.CONTROL + mirror.INVENTORY)

    def test_terminal_and_failed_stages_are_sticky_against_late_control_update(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            obj = instance(Path(temp))
            for terminal in ("REMOTE_TERMINAL_MIRRORED", "FAILED"):
                obj.state["stage"] = terminal
                obj.update(stage="REMOTE_RUNNING_AND_MIRRORING", control_poll_at=123)
                self.assertEqual(obj.state["stage"], terminal)
                obj.update(stage="MIRROR_CONNECTION_RETRY")
                self.assertEqual(obj.state["stage"], terminal)

    def test_actual_late_control_loop_cannot_lower_terminal_stage(self):
        class Client:
            def close(self): pass
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            obj = instance(Path(temp))
            obj.state["stage"] = "REMOTE_TERMINAL_MIRRORED"
            obj.connect = lambda: Client()
            obj.execute = lambda *args: {"files": {"RUN_STATUS.json": {"value": {"stage": mirror.REVIEW}}}}
            with mock.patch.object(obj.stop, "wait", side_effect=lambda seconds: obj.stop.set()):
                obj.control_loop()
            self.assertEqual(obj.state["stage"], "REMOTE_TERMINAL_MIRRORED")
            self.assertTrue(obj.state["control_poll_at"])

    def test_authentication_and_hostkey_errors_are_fatal_but_timeout_is_transient(self):
        import paramiko
        self.assertTrue(mirror.fatal_connection_error(paramiko.AuthenticationException("auth")))
        self.assertTrue(mirror.fatal_connection_error(paramiko.SSHException("Server x not found in known_hosts")))
        self.assertFalse(mirror.fatal_connection_error(TimeoutError("short interruption")))

    def test_fatal_control_auth_records_failure_instead_of_retrying(self):
        import paramiko
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            obj = instance(Path(temp))
            obj.connect = mock.Mock(side_effect=paramiko.AuthenticationException("auth"))
            obj.control_loop()
            self.assertEqual(obj.connect.call_count, 1)
            self.assertEqual(obj.state["stage"], "FAILED")
            self.assertTrue(obj.thread_errors)

    def test_terminal_requires_60000_references_and_binds_actual_report_bytes(self):
        class Client:
            def close(self): pass
        class Loader:
            def exec_module(self, module): pass
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            obj = instance(root)
            obj.connect = lambda: Client()
            obj.execute = lambda *args: {"files": {"RUN_STATUS.json": {"value": {"stage": mirror.REVIEW}}}}
            result = {"status": "PASS", "run_id": mirror.RUN_ID, "remote_stage": mirror.REVIEW,
                      "execution_references_verified": 60000, "execution_files_verified": 50000,
                      "scientific_gate": "INSUFFICIENT_SIGNAL"}
            fake_module = SimpleNamespace(verify_terminal=lambda *a, **k: result)
            with mock.patch.object(mirror.importlib.util, "spec_from_file_location", return_value=SimpleNamespace(loader=Loader())), \
                 mock.patch.object(mirror.importlib.util, "module_from_spec", return_value=fake_module):
                obj.terminal_verify()
                self.assertEqual(obj.state["stage"], "REMOTE_TERMINAL_MIRRORED")
                report = root / "artifacts/audit/mirror_terminal_verification.json"
                self.assertEqual(obj.state["terminal_verification_sha256"], mirror.sha_file(report))
                self.assertEqual(obj.state["mirror_terminal_verification_sha256"], mirror.sha_file(report))
                obj.state["stage"] = "FINALIZING_MIRROR"
                result["execution_references_verified"] = 10
                with self.assertRaises(RuntimeError):
                    obj.terminal_verify()
                self.assertNotEqual(obj.state["stage"], "REMOTE_TERMINAL_MIRRORED")


if __name__ == "__main__":
    unittest.main()
