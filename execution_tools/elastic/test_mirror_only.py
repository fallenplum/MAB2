"""CPU-only mirror transport contracts; all SSH/SFTP objects are fakes."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import stat
import subprocess
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
TEST_MTIME = 1_700_000_000
TEST_MTIME_NS = TEST_MTIME * 1_000_000_000
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
        return SimpleNamespace(st_size=self.opened_size, st_mtime=TEST_MTIME)
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
        return SimpleNamespace(st_size=len(self.data), st_mtime=TEST_MTIME)


def instance(root):
    return mirror.Mirror(SimpleNamespace(root=root, host="unused", port=1, user="unused",
                       remote_root="/unused", remote_python="python3", known_hosts=root / "unused"))


def make_directory_link(link, target):
    if os.name == "nt":
        result = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        if result.returncode:
            reason = (result.stderr or result.stdout).decode(errors="replace")[:400].strip()
            secret = os.environ.get("GRAPH_CREDIT_SSH_PASSWORD")
            if secret:
                reason = reason.replace(secret, "<redacted>")
            raise unittest.SkipTest(f"Windows junction creation rc={result.returncode}: {reason}")
    else:
        os.symlink(target, link, target_is_directory=True)


def remove_directory_link(link):
    if os.name == "nt": os.rmdir(link)
    else: link.unlink()


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
            for index, (sftp, immutable) in enumerate(((SFTP(b"short", opened_size=10), False), (SFTP(b"old-new", opened_size=3), True))):
                path = Path(temp) / f"snapshot_{index}"
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
            os.utime(path, (TEST_MTIME, TEST_MTIME))
            item = [relative, 2, TEST_MTIME_NS, 1, 1]
            self.assertFalse(obj.needs_transfer(item))
            with mock.patch.object(Path, "stat", side_effect=AssertionError("unexpected repeated stat")):
                self.assertFalse(obj.needs_transfer(item))
            obj.force_paths.add(relative)
            self.assertTrue(obj.needs_transfer(item))
            mutable = root / "report.json"
            mutable.write_bytes(b"{}")
            os.utime(mutable, (TEST_MTIME, TEST_MTIME))
            with mock.patch.object(Path, "stat", side_effect=AssertionError("mutable really statted")):
                with self.assertRaises(AssertionError):
                    obj.needs_transfer(["report.json", 2, TEST_MTIME_NS, 1, 1])

    def test_all_temporary_suffixes_and_path_traversal_are_rejected(self):
        for suffix in mirror.TEMP_SUFFIXES:
            self.assertTrue(mirror.is_temporary("status" + suffix))
        self.assertTrue(mirror.is_temporary("file.abc012.tmp"))
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            for name in ("../escape", "/absolute", "C:/absolute", "dir\\file"):
                with self.assertRaises(ValueError):
                    mirror.local_path(Path(temp), name)

    def test_missing_deep_suffix_is_lexical_and_never_resolves_missing_leaf(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            original = Path.resolve
            def guarded(path, *args, **kwargs):
                if path.suffix == ".json" and not path.exists():
                    raise AssertionError("Resolved missing destination leaf")
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, "resolve", guarded):
                target = mirror.local_path(root, "results/new/deep/raw_executions.jsonl")
                self.assertEqual(target, root / "results/new/deep/raw_executions.jsonl")
                target = mirror.local_path(root, "results/new/deep/run_manifest.json")
                mirror.atomic_bytes(target, b"{}", root=root)
                self.assertEqual(target.read_bytes(), b"{}")

    def test_eight_threads_can_create_missing_shared_parents_and_publish_240_files(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            identity = (root.lstat().st_dev, root.lstat().st_ino)
            def publish(index):
                name = f"results/pilot/motivation_v3_20261103/shard_002/elastic_segments/gpu7/executions/{index}.json"
                path = mirror.local_path(root, name, root_identity=identity)
                mirror.atomic_bytes(path, b"{}", root=root, root_identity=identity)
                return path.read_bytes()
            with ThreadPoolExecutor(max_workers=8) as pool:
                values = list(pool.map(publish, range(240)))
            self.assertEqual(values, [b"{}"] * 240)

    def test_unsafe_raw_components_are_rejected_before_posix_normalization(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            for name in ("a//b", "a/./b", "a/../b", "a/", "/a", "CON.txt", "NUL", "com1.json", "LPT9", "a.", "a ", "q<bad>", "a\x00b"):
                with self.subTest(name=name), self.assertRaises(mirror.MirrorPathError):
                    mirror.local_path(Path(temp), name)

    def test_permission_errors_are_not_treated_as_missing_suffixes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            original = Path.lstat
            def guarded(path):
                if path == root / "restricted":
                    raise PermissionError("Denied existing ancestor")
                return original(path)
            with mock.patch.object(Path, "lstat", guarded):
                with self.assertRaises(PermissionError):
                    mirror.local_path(root, "restricted/file.json")

    def test_windows_reparse_flag_is_rejected_even_if_target_would_stay_inside(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            flagged = root / "flagged"
            flagged.mkdir()
            original = Path.lstat
            def guarded(path):
                if path == flagged:
                    return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_file_attributes=0x400)
                return original(path)
            with mock.patch.object(Path, "lstat", guarded):
                with self.assertRaises(mirror.MirrorPathError) as caught:
                    mirror.local_path(root, "flagged/file.json")
            self.assertEqual(caught.exception.path_diagnostics["root"], str(root))
            self.assertEqual(caught.exception.path_diagnostics["candidate"], str(flagged / "file.json"))
            self.assertIn("resolved_existing_parent", caught.exception.path_diagnostics)

    def test_actual_directory_link_or_junction_to_outside_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            base = Path(temp)
            root, outside = base / "mirror", base / "outside"
            root.mkdir(); outside.mkdir()
            marker = outside / "marker"
            marker.write_bytes(b"preserve")
            link = root / "junction"
            make_directory_link(link, outside)
            try:
                with self.assertRaises(mirror.MirrorPathError):
                    mirror.local_path(root, "junction/escape.json")
                with self.assertRaises(mirror.MirrorPathError):
                    mirror.atomic_bytes(link / "escape.json", b"bad", root=root)
                self.assertFalse((outside / "escape.json").exists())
                self.assertEqual(marker.read_bytes(), b"preserve")
            finally:
                # Nonrecursive removal of this test-owned link, never the target.
                remove_directory_link(link)

    def test_junction_injected_after_mkdir_is_rejected_before_outside_write(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            base = Path(temp)
            root, outside = base / "mirror", base / "outside"
            root.mkdir(); outside.mkdir()
            (outside / "marker").write_bytes(b"preserve")
            link = root / "created"
            original = Path.mkdir
            injected = False
            def mkdir_then_swap(path, *args, **kwargs):
                nonlocal injected
                value = original(path, *args, **kwargs)
                if path == link and not injected:
                    os.rmdir(link)  # Only this just-created, empty test directory.
                    make_directory_link(link, outside)
                    injected = True
                return value
            try:
                with mock.patch.object(Path, "mkdir", mkdir_then_swap):
                    with self.assertRaises(mirror.MirrorPathError):
                        mirror.atomic_bytes(link / "escaped.json", b"bad", root=root)
                self.assertFalse((outside / "escaped.json").exists())
                self.assertEqual((outside / "marker").read_bytes(), b"preserve")
            finally:
                if injected:
                    remove_directory_link(link)

    def test_root_directory_replacement_identity_and_sibling_escape_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            base = Path(temp)
            root = base / "mirror"
            root.mkdir()
            original_identity = (root.lstat().st_dev, root.lstat().st_ino)
            root.rename(base / "old_mirror")
            root.mkdir()
            with self.assertRaises(mirror.MirrorPathError):
                mirror.local_path(root, "normal.json", root_identity=original_identity)
            with self.assertRaises(mirror.MirrorPathError):
                mirror.local_path(root, "../mirror_sibling/escaped.json")

    def test_dangling_reparse_point_is_not_treated_as_missing_suffix(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            dangling = root / "dangling"
            dangling.mkdir()
            real_lstat, real_resolve = Path.lstat, Path.resolve
            def lstat(path):
                if path == dangling:
                    return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_file_attributes=0x400)
                return real_lstat(path)
            def resolve(path, *args, **kwargs):
                if path == dangling:
                    raise FileNotFoundError("Dangling reparse target")
                return real_resolve(path, *args, **kwargs)
            with mock.patch.object(Path, "lstat", lstat), mock.patch.object(Path, "resolve", resolve):
                with self.assertRaises(mirror.MirrorPathError) as caught:
                    mirror.local_path(root, "dangling/escaped.json")
            self.assertEqual(caught.exception.path_diagnostics["resolved_existing_parent"], "unresolved:FileNotFoundError")

    def test_probe_only_writes_report_never_data_status_upload_or_launch(self):
        class Client:
            def close(self): pass
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            obj = instance(root)
            obj.connect = lambda: Client()
            obj.execute = lambda client, code: ({"files": {"RUN_STATUS.json": {"value": {"stage": "G1_REAL_GENERATION"}}}}
                                                if code == mirror.CONTROL else {"files": [["file.json", 2, TEST_MTIME_NS, 1, 1]]})
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

    def test_control_primary_and_cleanup_valueerrors_are_recorded_and_connection_recovers(self):
        class Client:
            def __init__(self, bad=False): self.bad, self.closes = bad, 0
            def close(self):
                self.closes += 1
                if self.bad: raise ValueError("control cleanup closed file")
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            obj = instance(Path(temp))
            first, second = Client(True), Client()
            obj.connect = mock.Mock(side_effect=[first, second])
            obj.execute = mock.Mock(side_effect=[ValueError("control primary failure"),
                                   {"files": {"RUN_STATUS.json": {"value": {"stage": "G1_REAL_GENERATION"}}}}])
            waits = 0
            def wait(_):
                nonlocal waits
                waits += 1
                if waits >= 2: obj.stop.set()
            with mock.patch.object(obj.stop, "wait", side_effect=wait):
                obj.control_loop()
            self.assertEqual(obj.thread_errors, [])
            self.assertEqual(obj.connect.call_count, 2)
            self.assertEqual(first.closes, 1)
            self.assertEqual(second.closes, 1)
            self.assertEqual(obj.control["stage"], "G1_REAL_GENERATION")
            self.assertIn("control primary failure", obj.state["last_control_error"]["traceback"])
            self.assertIn("control cleanup closed file", obj.state["last_cleanup_error"]["traceback"])

    def test_failed_attempts_clean_only_exclusive_own_temps_and_preserve_destination(self):
        class Client:
            def open_sftp(self): return SimpleNamespace(get_channel=lambda: SimpleNamespace(settimeout=lambda value: None))
            def close(self): pass
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            obj = instance(root)
            destination = root / "result.json"
            destination.write_bytes(b"{}")
            obj.pending.put(["result.json", 2, TEST_MTIME_NS, 1, 1])
            obj.state["queue_pending"] = 1
            obj.connect = lambda: Client()
            attempts = 0
            def fail_copy(sftp, remote, temporary, progress, **kwargs):
                nonlocal attempts
                attempts += 1
                with Path(temporary).open("xb") as handle:
                    kwargs["on_temp_created"]()
                    handle.write(b"unfinished owned snapshot")
                if attempts == 3: obj.stop.set()
                raise EOFError("Source changed during this attempt")
            with mock.patch.object(mirror, "fixed_copy", side_effect=fail_copy), mock.patch.object(obj.stop, "wait", return_value=False):
                obj.data_worker(0)
            self.assertEqual(attempts, 3)
            self.assertEqual(list(root.glob("*.mirrorpartial")), [])
            self.assertEqual(destination.read_bytes(), b"{}")
            self.assertEqual(obj.thread_errors, [])
            foreign = root / ("foreign.attach0." + "0" * 32 + ".mirrorpartial")
            foreign.write_bytes(b"foreign")
            info = foreign.lstat()
            obj.cleanup_owned_temp(foreign, (info.st_dev, info.st_ino + 1), worker=0)
            self.assertEqual(foreign.read_bytes(), b"foreign")

    def test_fixed_copy_never_truncates_an_existing_temporary_entry(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "existing.mirrorpartial"
            path.write_bytes(b"foreign")
            with self.assertRaises(FileExistsError):
                mirror.fixed_copy(SFTP(b"new"), "/remote/file", path, lambda *x: None)
            self.assertEqual(path.read_bytes(), b"foreign")

    def test_closed_file_error_and_cleanup_valueerror_reconnect_without_losing_primary_trace(self):
        class SFTPClient:
            def get_channel(self): return SimpleNamespace(settimeout=lambda value: None)
        class Client:
            def __init__(self, bad_close=False): self.bad_close = bad_close
            def open_sftp(self): return SFTPClient()
            def close(self):
                if self.bad_close: raise ValueError("cleanup used an already closed file")
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            obj = instance(root)
            obj.pending.put(["results/one.json", 2, TEST_MTIME_NS, 1, 1])
            obj.pending.put(["results/two.json", 2, TEST_MTIME_NS, 1, 2])
            obj.state["queue_pending"] = 2
            original_task_done = obj.pending.task_done
            def acknowledge():
                original_task_done()
                if obj.pending.unfinished_tasks == 0:
                    obj.stop.set()
            clients = iter((Client(True), Client(False)))
            obj.connect = lambda: next(clients)
            attempts = []
            def copy(sftp, remote, temporary, progress, **kwargs):
                attempts.append(remote)
                if len(attempts) == 1:
                    raise ValueError("I/O operation on closed file")
                Path(temporary).write_bytes(b"{}")
                progress(2, 2, 2)
                if len(attempts) == 3: obj.stop.set()
                return {"mtime": TEST_MTIME, "committed_bytes": 2, "source_changed": False}
            with mock.patch.object(mirror, "fixed_copy", side_effect=copy), mock.patch.object(obj.stop, "wait", return_value=False), \
                    mock.patch.object(obj.pending, "task_done", side_effect=acknowledge):
                obj.data_worker(3)
            self.assertEqual(obj.thread_errors, [])
            self.assertEqual(obj.pending.unfinished_tasks, 0)
            self.assertEqual(obj.state["files_committed"], 2, msg=mirror.dumps(obj.state))
            self.assertEqual((root / "results/two.json").read_bytes(), b"{}")
            self.assertIn("I/O operation on closed file", obj.state["last_file_error"]["traceback"])
            self.assertEqual(obj.state["last_file_error"]["path"], "results/one.json")
            self.assertEqual(obj.state["last_file_error"]["phase"], "copy_open_snapshot")
            self.assertIn("already closed file", obj.state["last_cleanup_error"]["traceback"])

    def test_outer_path_failure_keeps_full_trace_and_filename_after_current_file_cleanup(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            obj = instance(Path(temp))
            obj.pending.put(["unsafe:name.json", 2, TEST_MTIME_NS, 1, 1])
            obj.state["queue_pending"] = 1
            obj.data_worker(3)
            self.assertTrue(obj.thread_errors)
            self.assertEqual(obj.state["last_failed_file"], "unsafe:name.json")
            self.assertEqual(obj.state["last_thread_error"]["phase"], "prepare_local_path")
            self.assertIn("Unsafe mirror path", obj.state["last_thread_error"]["traceback"])
            self.assertEqual(obj.state["current_files"], {})
            self.assertEqual(obj.pending.unfinished_tasks, 0)

    def test_exception_diagnostics_redact_runtime_ssh_password(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp, mock.patch.dict(os.environ, {"GRAPH_CREDIT_SSH_PASSWORD": "DO_NOT_LOG"}):
            obj = instance(Path(temp))
            try:
                raise ValueError("sensitive=DO_NOT_LOG")
            except ValueError as exc:
                detail = obj.exception_detail(exc, worker=3, path="file.json", phase="copy_open_snapshot")
            self.assertNotIn("DO_NOT_LOG", detail["error_message"])
            self.assertNotIn("DO_NOT_LOG", detail["traceback"])
            self.assertIn("<redacted>", detail["traceback"])

    def test_queue_accounting_valueerror_remains_fatal_with_specific_filename_and_stack(self):
        class Client:
            def open_sftp(self): return SimpleNamespace(get_channel=lambda: SimpleNamespace(settimeout=lambda value: None))
            def close(self): pass
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            obj = instance(Path(temp))
            obj.pending.put(["results/committed.json", 2, TEST_MTIME_NS, 1, 1])
            obj.state["queue_pending"] = 1
            obj.connect = lambda: Client()
            def copy(sftp, remote, path, progress, **kwargs):
                Path(path).write_bytes(b"{}")
                return {"mtime": TEST_MTIME, "committed_bytes": 2, "source_changed": False}
            with mock.patch.object(mirror, "fixed_copy", side_effect=copy), \
                    mock.patch.object(obj.pending, "task_done", side_effect=ValueError("task_done() called too many times")):
                obj.data_worker(3)
            detail = obj.state["last_thread_error"]
            self.assertEqual(detail["phase"], "acknowledge_queue")
            self.assertEqual(detail["path"], "results/committed.json")
            self.assertIn("task_done() called too many times", detail["traceback"])
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
