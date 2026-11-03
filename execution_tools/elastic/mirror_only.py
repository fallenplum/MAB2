"""Attach-only result mirror: no remote writes, uploads or experiment launches."""
from __future__ import annotations
import argparse
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shlex
import threading
import time
import uuid

RUN_ID = "motivation_v3_20261103"
REVIEW = "G1_GENERATION_AND_ANALYSIS_COMPLETE_REVIEW_REQUIRED"
FAILED_STAGES = {"FAILED", "G1_PARTIAL_FAILED", "FAILED_CHILDREN_SETTLING"}
CONTROL_FILES = ["RUN_STATUS.json"] + [f"results/pilot/{RUN_ID}/shard_{i:03d}/STATUS.json" for i in range(3)]
TEMP_SUFFIXES = (".tmp", ".workertmp", ".pooltmp", ".migrationtmp", ".remote_partial", ".uploading", ".mirrorpartial")

INVENTORY = r'''
import json,os,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();run="motivation_v3_20261103"; files=[]; roots=[]
pilot=r/"results/pilot"
if pilot.exists():
 for p in pilot.iterdir():
  if p.is_dir() and (p.name in [run,"s0_v3_20261103","message_sanity_v3_20261103"] or p.name.startswith("elastic_")):roots.append(p)
roots.append(r/"artifacts/audit")
private=r/"artifacts/private/performance"
if private.exists():
 for p in private.iterdir():
  n=p.name
  if p.is_file() and (n.endswith(".py") or n=="allocation.json" or n.startswith(("tasks_gpu","worker_gpu")) and n.endswith((".json",".claim"))):roots.append(p)
logs=r/"logs"
if logs.exists():
 for p in logs.iterdir():
  if p.is_file() and p.name.startswith(("elastic_","g1_elastic_","s0_v3","message_sanity_v3")):roots.append(p)
seen=set()
for source in roots:
 for p in ([source] if source.is_file() else source.rglob("*")):
  if p.name.endswith((".tmp",".workertmp",".pooltmp",".migrationtmp",".remote_partial",".uploading",".mirrorpartial")):continue
  try:
   if p.is_symlink() or not p.is_file():continue
   s=p.stat();relative=p.relative_to(r).as_posix()
   if relative in seen:continue
   seen.add(relative);files.append([relative,s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino])
  except FileNotFoundError:continue
print(json.dumps({"files":files,"observed_unix":__import__("time").time()}))
'''

CONTROL = r'''
import base64,json,sys,time
from pathlib import Path
r=Path(sys.argv[1]).resolve();names=["RUN_STATUS.json"]+["results/pilot/motivation_v3_20261103/shard_%03d/STATUS.json"%i for i in range(3)];out={}
for n in names:
 try:
  raw=(r/n).read_bytes(); value=json.loads(raw);out[n]={"base64":base64.b64encode(raw).decode(),"value":value}
 except FileNotFoundError:continue
 except (ValueError,OSError) as e:out[n]={"error_type":type(e).__name__}
print(json.dumps({"files":out,"observed_unix":time.time()}))
'''


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_json(path, value):
    atomic_bytes(path, (dumps(value) + "\n").encode("utf-8"))


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def local_path(root, name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Unsafe mirror path")
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("Unsafe mirror path")
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Mirror path escapes the root")
    return path


def is_temporary(name):
    return Path(name).name.endswith(TEMP_SUFFIXES)


def immutable(name):
    return ("executions" in PurePosixPath(name).parts or "qualification_executions" in PurePosixPath(name).parts) and name.endswith(".json")


def fatal_connection_error(exc):
    import paramiko
    return (isinstance(exc, (paramiko.AuthenticationException, paramiko.BadHostKeyException, paramiko.PasswordRequiredException))
            or isinstance(exc, paramiko.SSHException) and "not found in known_hosts" in str(exc))


def signature(item):
    return tuple(item[1:])


def legacy_process(root):
    """Refuse an actual live legacy publisher; never signal it."""
    import psutil
    status = read_json(root / "COORDINATOR_STATUS.json")
    pid = status.get("pid")
    result = {"recorded_pid": pid, "legacy_alive": False}
    if not pid or int(pid) == os.getpid():
        return result
    try:
        process = psutil.Process(int(pid))
        argv = process.cmdline()
        result.update(recorded_process_alive=process.is_running(), process_create_time=process.create_time())
        try:
            cli_root = Path(argv[argv.index("--root") + 1]).resolve()
        except (ValueError, IndexError):
            cli_root = None
        result["legacy_alive"] = process.is_running() and cli_root == root and any(Path(x).name == "local_coordinator.py" for x in argv)
    except psutil.NoSuchProcess:
        result["recorded_process_alive"] = False
    return result


class PublisherLock:
    def __init__(self, path):
        self.path = path
        self.handle = None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.handle.seek(0, 2) == 0:
            self.handle.write(b"0"); self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Another mirror-only publisher holds the local lease")
        return self
    def __exit__(self, *_):
        if self.handle:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


def fixed_copy(sftp, remote, temporary, progress, *, is_immutable=False, is_jsonl=False):
    """Copy exactly the open-handle size; append-only growth is a new version."""
    started = time.monotonic()
    with sftp.open(remote, "rb") as source:
        attrs = source.stat()
        length = int(attrs.st_size)
        deadline = started + min(900, max(120, length / 131072))
        if length:
            source.prefetch(file_size=length, max_concurrent_requests=16)
        with Path(temporary).open("wb") as target:
            received = 0
            while received < length:
                if time.monotonic() > deadline:
                    raise TimeoutError("File transfer exceeded bounded total duration")
                block = source.read(min(131072, length - received))
                if not block:
                    raise EOFError("Source shrank before the opened snapshot length")
                target.write(block)
                received += len(block)
                progress(len(block), received, length)
            extra = source.read(1)  # Never follow appended data beyond this single-byte test.
            target.flush()
            os.fsync(target.fileno())
        final = sftp.stat(remote)
        changed = bool(extra) or final.st_size != length or int(final.st_mtime) != int(attrs.st_mtime)
        if is_immutable and changed:
            raise EOFError("Immutable source changed during transfer; retry a new snapshot")
        if Path(temporary).stat().st_size != length:
            raise IOError("Local snapshot length mismatch")
        committed = length
        if is_jsonl and length:
            with Path(temporary).open("r+b") as target:
                target.seek(length - 1)
                if target.read(1) != b"\n":
                    end, boundary = length, 0
                    while end > 0:
                        start = max(0, end - 65536)
                        target.seek(start); data = target.read(end - start)
                        found = data.rfind(b"\n")
                        if found >= 0:
                            boundary = start + found + 1; break
                        end = start
                    target.truncate(boundary)
                    target.flush(); os.fsync(target.fileno())
                    committed = boundary
                    changed = True
        return {"opened_bytes": length, "committed_bytes": committed,
                "mtime": int(attrs.st_mtime), "source_changed": changed}


class Mirror:
    def __init__(self, args):
        import psutil
        self.args, self.root = args, args.root.resolve()
        self.private = self.root / "artifacts/private/mirroring"
        self.private.mkdir(parents=True, exist_ok=True)
        current = psutil.Process()
        self.state = {"schema": "graph_credit_mirror_only_v1", "pid": os.getpid(), "run_id": RUN_ID,
                      "process_create_time": current.create_time(), "owner_username": current.username(),
                      "stage": "REMOTE_RUNNING_AND_MIRRORING", "scientific_completion": False,
                      "mode": "read_only_attach_no_upload_no_launch", "bytes_transferred": 0,
                      "bytes_committed": 0, "files_committed": 0, "queue_pending": 0,
                      "current_files": {}, "file_errors": 0}
        self.guard = threading.RLock()
        self.stop = threading.Event()
        self.pending = queue.Queue()
        self.cache = {}
        self.force_paths = set()
        self.force_all = False
        self.control = {}
        self.thread_errors = []

    def connect(self):
        import paramiko
        client = paramiko.SSHClient()
        try:
            client.load_host_keys(str(self.args.known_hosts))
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.connect(self.args.host, port=self.args.port, username=self.args.user,
                           password=os.environ["GRAPH_CREDIT_SSH_PASSWORD"], look_for_keys=False,
                           allow_agent=False, timeout=20, auth_timeout=20, banner_timeout=20)
            client.get_transport().set_keepalive(15)
            return client
        except BaseException:
            client.close()
            raise

    def execute(self, client, code, timeout=30):
        command = " ".join((shlex.quote(self.args.remote_python), "-c", shlex.quote(code), shlex.quote(self.args.remote_root)))
        _, out, err = client.exec_command(command, timeout=timeout)
        data = out.read(); err.read()
        if out.channel.recv_exit_status():
            raise RuntimeError("Read-only remote inspection failed")
        return json.loads(data)

    def update(self, **values):
        with self.guard:
            if self.state.get("stage") in {"REMOTE_TERMINAL_MIRRORED", "FAILED"}:
                values.pop("stage", None)
            self.state.update(values)

    def publish_status(self):
        with self.guard:
            record = {**self.state, "updated_at_unix": time.time()}
        atomic_json(self.root / "COORDINATOR_STATUS.json", record)

    def heartbeat(self):
        while not self.stop.is_set():
            try:
                self.publish_status()
            except Exception as exc:
                self.thread_errors.append(type(exc).__name__)
            self.stop.wait(5)

    def control_loop(self):
        client = None
        try:
            while not self.stop.is_set():
                try:
                    client = client or self.connect()
                    data = self.execute(client, CONTROL)
                    if self.stop.is_set():
                        break
                    for name, value in data["files"].items():
                        if "base64" in value:
                            atomic_bytes(local_path(self.root, name), base64.b64decode(value["base64"]))
                    main = data.get("files", {}).get("RUN_STATUS.json", {}).get("value", {})
                    with self.guard:
                        self.control = main
                        self.state.update(control_poll_at=time.time(), remote_stage=main.get("stage"), control_error_type=None)
                        if self.state.get("stage") not in {"FINALIZING_MIRROR", "REMOTE_TERMINAL_MIRRORED", "FAILED"}:
                            self.state["stage"] = "REMOTE_RUNNING_AND_MIRRORING"
                except Exception as exc:
                    if fatal_connection_error(exc):
                        self.thread_errors.append("control_fatal:" + type(exc).__name__)
                        self.update(stage="FAILED", control_error_type=type(exc).__name__)
                        break
                    with self.guard:
                        self.state["control_error_type"] = type(exc).__name__
                        if self.state.get("stage") not in {"FINALIZING_MIRROR", "REMOTE_TERMINAL_MIRRORED", "FAILED"}:
                            self.state["stage"] = "MIRROR_CONNECTION_RETRY"
                    if client:
                        client.close(); client = None
                self.stop.wait(30)
        finally:
            if client:
                client.close()

    def needs_transfer(self, item, forced=False):
        name, size, mtime_ns, *_ = item
        if name in CONTROL_FILES or is_temporary(name):
            return False
        if forced or name in self.force_paths:
            return True
        if immutable(name) and self.cache.get(name) == signature(item):
            return False
        path = local_path(self.root, name)
        try:
            stat = path.stat()
            matched = stat.st_size == size and int(stat.st_mtime) == mtime_ns // 1_000_000_000
        except FileNotFoundError:
            matched = False
        if matched and immutable(name):
            self.cache[name] = signature(item)
        return not matched

    def data_worker(self, index):
        client = sftp = None
        try:
            while not self.stop.is_set():
                try:
                    item = self.pending.get(timeout=.5)
                except queue.Empty:
                    continue
                name = item[0]
                try:
                    with self.guard:
                        self.state["current_files"][str(index)] = {"path": name, "bytes": 0, "last_progress_at": time.time()}
                    local = local_path(self.root, name)
                    local.parent.mkdir(parents=True, exist_ok=True)
                    temp = local.with_name(local.name + f".attach{index}.mirrorpartial")
                    success = False
                    for attempt in range(3):
                        if self.stop.is_set():
                            break
                        try:
                            if client is None:
                                client = self.connect(); sftp = client.open_sftp()
                                sftp.get_channel().settimeout(25)
                            def progress(delta, received, total):
                                with self.guard:
                                    self.state["bytes_transferred"] += delta
                                    self.state["current_files"][str(index)] = {"path": name, "bytes": received,
                                         "opened_bytes": total, "last_progress_at": time.time(), "attempt": attempt + 1}
                            result = fixed_copy(sftp, self.args.remote_root.rstrip("/") + "/" + name, temp, progress,
                                                is_immutable=immutable(name), is_jsonl=name.endswith(".jsonl"))
                            if name.endswith(".json") and not immutable(name):
                                json.loads(temp.read_bytes())
                            os.replace(temp, local)
                            os.utime(local, (result["mtime"], result["mtime"]))
                            with self.guard:
                                self.state["files_committed"] += 1
                                self.state["bytes_committed"] += result["committed_bytes"]
                                self.force_paths.discard(name)
                                if immutable(name) and not result["source_changed"]:
                                    self.cache[name] = signature(item)
                                else:
                                    self.cache.pop(name, None)
                            success = True
                            break
                        except Exception as exc:
                            if fatal_connection_error(exc):
                                self.thread_errors.append("data_fatal:" + type(exc).__name__)
                                raise
                            self.update(last_file_error={"path": name, "error_type": type(exc).__name__, "attempt": attempt + 1})
                            if client:
                                client.close(); client = sftp = None
                            if isinstance(exc, FileNotFoundError) or getattr(exc, "errno", None) == 2:
                                break  # A disappearing file never restarts the entire mirror cycle.
                            self.stop.wait(min(2 ** attempt, 4))
                    if not success:
                        with self.guard:
                            self.state["file_errors"] += 1
                            self.cache.pop(name, None)
                finally:
                    with self.guard:
                        self.state["current_files"].pop(str(index), None)
                        self.state["queue_pending"] = max(0, self.state["queue_pending"] - 1)
                    self.pending.task_done()
        except Exception as exc:
            self.thread_errors.append("data_worker_" + str(index) + ":" + type(exc).__name__)
        finally:
            if client:
                client.close()

    def probe(self):
        client = self.connect()
        try:
            control = self.execute(client, CONTROL)
            inventory = self.execute(client, INVENTORY)
        finally:
            client.close()
        report = {"schema": "graph_credit_mirror_attach_probe_v1", "run_id": RUN_ID,
                  "observed_unix": time.time(), "legacy": legacy_process(self.root),
                  "remote_stage": control.get("files", {}).get("RUN_STATUS.json", {}).get("value", {}).get("stage"),
                  "remote_files": len(inventory["files"]), "remote_bytes": sum(item[1] for item in inventory["files"]),
                  "remote_access": "read_only", "files_copied": 0, "experiment_launches": 0}
        atomic_json(self.private / "probe_latest.json", report)
        return report

    def terminal_verify(self):
        module_path = Path(__file__).with_name("mirror_terminal_verify.py")
        if not module_path.exists():
            raise FileNotFoundError("Terminal verifier module is not deployed")
        spec = importlib.util.spec_from_file_location("mirror_terminal_verify", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def progress(value):
            self.update(terminal_verification_progress=value, terminal_verification_progress_at=time.time())
        result = module.verify_terminal(self.root, run_id=RUN_ID, progress_callback=progress)
        if (result.get("status") != "PASS" or result.get("run_id") != RUN_ID
                or result.get("execution_references_verified") != 60000 or result.get("remote_stage") != REVIEW):
            raise RuntimeError("Terminal verifier did not establish the requested completed run")
        report_path = self.root / "artifacts/audit/mirror_terminal_verification.json"
        atomic_json(report_path, result)
        client = self.connect()
        try:
            remote = self.execute(client, CONTROL)
        finally:
            client.close()
        stage = remote.get("files", {}).get("RUN_STATUS.json", {}).get("value", {}).get("stage")
        if stage != REVIEW:
            raise RuntimeError("Remote experiment changed during final local verification")
        report_sha = sha_file(report_path)
        self.update(stage="REMOTE_TERMINAL_MIRRORED", remote_stage=stage,
                    terminal_verification_sha256=report_sha, mirror_terminal_verification_sha256=report_sha,
                    mirrored_files_verified=result.get("execution_files_verified"), workflow_complete=False)

    def run(self):
        if legacy_process(self.root)["legacy_alive"]:
            raise RuntimeError("LEGACY_COORDINATOR_ALIVE: settle the old local publisher before attach")
        os.chdir(self.root)
        with PublisherLock(self.private / "mirror_only.lock"):
            if legacy_process(self.root)["legacy_alive"]:
                raise RuntimeError("LEGACY_COORDINATOR_ALIVE")
            threads = [threading.Thread(target=self.heartbeat, daemon=True),
                       threading.Thread(target=self.control_loop, daemon=True)]
            threads += [threading.Thread(target=self.data_worker, args=(i,), daemon=True) for i in range(4)]
            for thread in threads:
                thread.start()
            terminal_error, terminal_repeats = None, 0
            try:
                while not self.stop.is_set():
                    if self.thread_errors:
                        raise RuntimeError("Local heartbeat persistence failed: " + self.thread_errors[-1])
                    client = None
                    try:
                        client = self.connect()
                        inventory = self.execute(client, INVENTORY)
                        self.update(cycle_started_at=time.time(), scanned_files=len(inventory["files"]), inventory_error_type=None)
                    except Exception as exc:
                        if fatal_connection_error(exc):
                            raise
                        self.update(stage="MIRROR_CONNECTION_RETRY", inventory_error_type=type(exc).__name__)
                        self.stop.wait(10)
                        continue
                    finally:
                        if client:
                            client.close()
                    forced = self.force_all
                    self.force_all = False
                    for item in inventory["files"]:
                        if self.needs_transfer(item, forced):
                            with self.guard:
                                self.state["queue_pending"] += 1
                            self.pending.put(item)
                    while self.pending.unfinished_tasks and not self.stop.is_set():
                        if self.thread_errors:
                            raise RuntimeError("A mirror data thread failed: " + self.thread_errors[-1])
                        self.stop.wait(1)
                    if self.thread_errors:
                        raise RuntimeError("A mirror background thread failed: " + self.thread_errors[-1])
                    if self.control.get("stage") in FAILED_STAGES:
                        raise RuntimeError("Remote experiment reported failure; available evidence mirrored")
                    if self.control.get("stage") == REVIEW:
                        self.update(stage="FINALIZING_MIRROR")
                        try:
                            self.terminal_verify()
                            self.stop.set()
                            break
                        except Exception as exc:
                            relative = getattr(exc, "relative_path", None)
                            token = (type(exc).__name__, str(relative))
                            terminal_repeats = terminal_repeats + 1 if token == terminal_error else 1
                            terminal_error = token
                            self.cache.clear()
                            if relative:
                                local_path(self.root, relative)
                                self.force_paths.add(relative)
                            else:
                                self.force_all = True
                            self.update(stage="FINALIZING_MIRROR", terminal_error_type=type(exc).__name__,
                                        terminal_error_relative_path=relative, terminal_attempt=terminal_repeats)
                            if terminal_repeats >= 3:
                                raise RuntimeError("Repeated terminal verification failure: " + type(exc).__name__)
                    self.stop.wait(5 if self.control.get("stage") == REVIEW else 30)
            except BaseException as exc:
                self.update(stage="FAILED", error_type=type(exc).__name__)
                raise
            finally:
                self.stop.set()
                for thread in threads:
                    thread.join(timeout=30)
                self.publish_status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--known-hosts", required=True, type=Path)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("GRAPH_CREDIT_SSH_PASSWORD"):
        parser.error("GRAPH_CREDIT_SSH_PASSWORD runtime environment is required")
    args.root = args.root.resolve(strict=True)
    mirror = Mirror(args)
    if args.probe_only:
        print(dumps(mirror.probe()), flush=True)
        return 0
    try:
        mirror.run()
        print(dumps({"stage": mirror.state["stage"], "run_id": RUN_ID}), flush=True)
        return 0
    except Exception as exc:
        print(dumps({"stage": "FAILED", "error_type": type(exc).__name__, "run_id": RUN_ID}), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
