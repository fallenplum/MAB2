"""Verified handoff of only this project's owned v3 processes; no label reads."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import platform
import sys
import time

RUN_ID = "motivation_v3_20261103"


def checked_syscall(number, *arguments):
    # Linux x86-64 ABI numbers verified against the Linux syscall table. This
    # path preserves pidfd identity protection when Python was built without
    # pidfd wrappers; unsupported kernels fail before any legacy process stops.
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("Native pidfd fallback is restricted to Linux x86-64")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return checked_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_signal(handle, signum):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(handle, signum)
    return checked_syscall(424, ctypes.c_int(handle), ctypes.c_int(signum),
                           ctypes.c_void_p(), ctypes.c_uint(0))


def pidfd_preflight():
    handle = pidfd_open(os.getpid())
    try:
        pidfd_signal(handle, 0)  # Query this preflight process only, no state change.
    finally:
        os.close(handle)
    return "python_or_verified_linux_pidfd_syscalls"


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".migrationtmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def valid_prefix(data):
    """Keep all complete JSON records, preserving an uncommitted tail separately."""
    lines = data.splitlines(keepends=True)
    records, complete = [], []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise ValueError("Invalid JSON before final append record")
            return records, b"".join(complete), line
        if not isinstance(record, dict):
            raise ValueError("Non-object ledger row")
        records.append(record)
        complete.append(line.rstrip(b"\r\n") + b"\n")
    return records, b"".join(complete), b""


def design_steps(plan):
    return [(context, pair, cell) for context in plan["base_masks"]
            for pair in plan["edge_pairs"] for cell in ("11", "01", "00", "10")]


def row_key(row):
    return (row["question_id"], row["seed"], row["context_family"],
            row["edge_e"], row["edge_f"], row["cell"])


def make_tasks(root, plan, shard_index, raw_rows):
    steps = design_steps(plan)
    if len(steps) != 60:
        raise ValueError("Unexpected unit size")
    assigned = [qid for index, qid in enumerate(plan["question_ids"]) if index % 3 == shard_index]
    expected = [(qid, seed, context["name"], pair[0], pair[1], cell)
                for qid in assigned for seed in plan["seeds"] for context, pair, cell in steps]
    if [row_key(row) for row in raw_rows] != expected[:len(raw_rows)] or len(raw_rows) > len(expected):
        raise ValueError("Legacy outputs are not the exact frozen-design prefix")
    tasks = []
    offset = 0
    for qid in assigned:
        for seed in plan["seeds"]:
            done = min(60, max(0, len(raw_rows) - offset))
            if done < 60:
                task = {"logical_shard": shard_index, "question_id": qid, "seed": seed,
                        "start_index": done, "expected_new_rows": 60 - done}
                if done:
                    last = raw_rows[offset + done - 1]
                    shard = root / "results/pilot" / RUN_ID / f"shard_{shard_index:03d}"
                    path = (shard / last["execution_path"]).resolve()
                    if not path.is_relative_to(shard) or sha_bytes(path.read_bytes()) != last["execution_sha256"]:
                        raise ValueError("Partial-unit resume execution is missing or corrupt")
                    task.update(resume_execution_path=last["execution_path"],
                                resume_execution_sha256=last["execution_sha256"])
                tasks.append(task)
            offset += 60
    return tasks


def owned_process(root, pid, role, shard_index=None):
    p = Path("/proc") / str(pid)
    argv = [x.decode() for x in (p / "cmdline").read_bytes().split(b"\0") if x]
    cwd = Path(os.readlink(p / "cwd")).resolve()
    fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
    status = (p / "status").read_text()
    uid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("Uid:")))
    name = "server_supervisor.py" if role == "main" else "run_pilot.py"
    scripts = [x for x in argv if x.endswith("/" + name) or x == name]
    def flag(key):
        return argv[argv.index(key) + 1]
    if uid != os.getuid() or cwd != root or not any((cwd / x).resolve() == root / "scripts" / name for x in scripts):
        raise ValueError("Refuse to signal unowned process")
    if Path(flag("--root")).resolve() != root:
        raise ValueError("Unexpected process root")
    if role != "main" and (flag("--run-id") != RUN_ID or flag("--num-shards") != "3" or flag("--shard-index") != str(shard_index)):
        raise ValueError("Unexpected worker identity")
    return {"pid": pid, "role": role, "shard_index": shard_index,
            "start_ticks": int(fields[19]), "command_sha256": sha_bytes(json.dumps(argv).encode()),
            "uid": uid, "cwd": str(cwd)}


@contextmanager
def frozen_owned(root, identities, receipt_path, receipt):
    """Record first, await actual stops, and never leave paused jobs on failure."""
    atomic(receipt_path, receipt)
    handles = []
    stopped = []
    try:
        for item in identities:
            handle = pidfd_open(item["pid"])
            handles.append((item, handle))
            if owned_process(root, item["pid"], item["role"], item["shard_index"]) != item:
                raise ValueError("Process identity changed during handoff")
            pidfd_signal(handle, signal.SIGSTOP)
            stopped.append((item, handle))
        deadline = time.monotonic() + 10
        while True:
            states = [(Path("/proc") / str(item["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()[0]
                      for item, _ in handles]
            if all(state in {"T", "t"} for state in states):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Owned workers did not actually stop")
            time.sleep(0.05)
        receipt["phase"] = "ALL_OWNED_PROCESSES_CONFIRMED_STOPPED"
        atomic(receipt_path, receipt)
        yield handles
    except BaseException as exc:
        # A stopped process must not resume into files changed during snapshot
        # normalization. Before that point, rollback the pause without loss.
        terminate = receipt.get("snapshot_started", False) or receipt.get("legacy_killed", False)
        for item, handle in reversed(stopped):
            try:
                pidfd_signal(handle, signal.SIGKILL if terminate else signal.SIGCONT)
            except ProcessLookupError:
                pass
        receipt.update(state="FAILED", failed_unix=time.time(), error_type=type(exc).__name__,
                       rollback="legacy_terminated_artifacts_retained" if terminate else "pauses_reverted")
        atomic(receipt_path, receipt)
        raise
    finally:
        for _, handle in handles:
            os.close(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--deferred-gpus", default="")
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    pidfd_preflight()  # Capability gate precedes receipt creation and every pause.
    base = root / "results/pilot" / RUN_ID
    receipt_path = root / "artifacts/audit/elastic_migration_20261103.json"
    if receipt_path.exists():
        raise FileExistsError("Migration receipt already exists; no silent second handoff")
    plans = [(base / f"shard_{i:03d}" / "frozen_plan.json").read_bytes() for i in range(3)]
    if len(set(plans)) != 1:
        raise ValueError("Logical shards do not share their original plan")
    plan = json.loads(plans[0])
    for relative, item in plan["source_manifest"].items():
        if sha_bytes((root / relative).read_bytes()) != item["sha256"]:
            raise ValueError("Frozen source changed")
    state = read(root / "RUN_STATUS.json")
    if state["schema"] != "graph_credit_supervisor_v1" or state["stage"] != "G1_REAL_GENERATION":
        raise ValueError("Unexpected legacy supervisor state")
    identities = [owned_process(root, state["pid"], "main")]
    identities += [owned_process(root, pid, "child", i) for i, pid in enumerate(state["active_pids"])]
    if len(identities) != 4:
        raise ValueError("Expected exactly three legacy workers")
    for i in range(3):
        shard = base / f"shard_{i:03d}"
        if (shard / "OUTPUTS_FROZEN.json").exists() or (shard / "SCORING.json").exists():
            raise ValueError("Labels may already have been opened")
        raw_rows, _, _ = valid_prefix((shard / "raw_executions.jsonl").read_bytes())
        make_tasks(root, plan, i, raw_rows)
    if not args.commit:
        print(json.dumps({"status": "DRY_RUN_PASS", "owned_processes": identities,
                          "labels_read": False, "frozen_plan_sha256": sha_bytes(plans[0])}))
        return 0
    gpus = [int(x) for x in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)) or any(x not in range(8) for x in gpus):
        raise ValueError("Invalid GPU allocation")
    deferred = {int(x) for x in args.deferred_gpus.split(",") if x.strip()}
    if not deferred.issubset(set(gpus)):
        raise ValueError("Deferred GPU is not a pool candidate")
    folder = root / "artifacts/private/performance"
    scheduler_sources = {str((folder / name).relative_to(root).as_posix()): sha_bytes((folder / name).read_bytes())
                         for name in ("elastic_worker.py", "elastic_finalize.py", "elastic_supervisor.py", "elastic_migrate.py",
                                      "cross_gpu_audit.py", "elastic_resume_qualification.py")}
    if (folder / "allocation.json").exists() or any((folder / f"tasks_gpu{gpu}.json").exists() for gpu in gpus):
        raise FileExistsError("Allocation already exists; attach instead of repeating handoff")
    qualification = read(root / "artifacts/audit/elastic_cross_gpu_qualification.json")
    if qualification.get("status") != "PASS" or not (set(gpus) - deferred).issubset(set(qualification["qualified_gpus"])):
        raise ValueError("Cross-GPU execution equivalence has not passed")
    resume_gate = read(root / "artifacts/audit/elastic_resume_qualification.json")
    if resume_gate.get("status") != "PASS" or resume_gate.get("worker_source_sha256") != scheduler_sources["artifacts/private/performance/elastic_worker.py"]:
        raise ValueError("Real deserialize/cache resume qualification has not passed for this worker code")
    snapshots = []
    tasks = []
    receipt = {"schema": "graph_credit_elastic_migration_v1", "state": "IN_PROGRESS",
               "run_id": RUN_ID, "started_unix": time.time(), "labels_read": False,
               "logical_shards": 3, "frozen_plan_sha256": sha_bytes(plans[0]),
               "source_manifest_sha256": plan["source_manifest_sha256"], "owned_processes": identities,
               "migration_inflight_upper_bound": 3,
               "unknown_cost_policy": "Up to one unfinished model call per stopped worker; cost unknown, never imputed zero",
               "files": snapshots}
    receipt["phase"] = "BEFORE_ANY_PAUSE"
    with frozen_owned(root, identities, receipt_path, receipt) as handles:
        receipt["snapshot_started"] = True
        atomic(receipt_path, receipt)
        for i in range(3):
            shard = base / f"shard_{i:03d}"
            backup = shard / "migration_20261103"
            backup.mkdir(exist_ok=False)
            for name in ("STATUS.json", "run_manifest.json", "frozen_plan.json"):
                (backup / name).write_bytes((shard / name).read_bytes())
            raw_rows = None
            for name in ("raw_executions.jsonl", "physical_calls.jsonl"):
                original = (shard / name).read_bytes()
                records, normalized, tail = valid_prefix(original)
                target = backup / (name + ".original_bytes")
                target.write_bytes(original)
                if normalized != original:
                    (shard / name).write_bytes(normalized)
                snapshots.append({"logical_shard": i, "path": (shard / name).relative_to(root).as_posix(),
                                  "backup_path": target.relative_to(root).as_posix(),
                                  "backup_sha256": sha_bytes(original), "normalized_sha256": sha_bytes(normalized),
                                  "valid_rows": len(records), "partial_tail_bytes": len(tail),
                                  "partial_tail_sha256": sha_bytes(tail) if tail else None})
                if name == "raw_executions.jsonl":
                    raw_rows = records
            tasks.extend(make_tasks(root, plan, i, raw_rows))
        # Every paid call and already-complete execution remains on disk.
        for item, handle in reversed(handles):
            pidfd_signal(handle, signal.SIGKILL)
        receipt["legacy_killed"] = True
        # Identical immutable candidate queues with atomic per-unit ownership.
        # Faster GPUs claim more independent units; no unit is regenerated.
        remaining = sum(task["expected_new_rows"] for task in tasks)
        schedules = [tasks for _ in gpus]
        loads = [remaining for _ in gpus]
        receipt.update(state="COMPLETE", completed_unix=time.time(),
                       remaining_rows=remaining, completed_rows=60000 - remaining,
                       claim_mode="shared_atomic_units")
        final_receipt_sha = sha_bytes((json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        workers = []
        for index, gpu in enumerate(gpus):
            worker_id = f"gpu{gpu}"
            path = folder / f"tasks_{worker_id}.json"
            if path.exists():
                raise FileExistsError("Task allocation already exists")
            manifest = {"schema": "graph_credit_elastic_tasks_v1", "run_id": RUN_ID,
                        "worker_id": worker_id, "logical_shards": 3,
                        "claim_mode": "shared_atomic_units",
                        "frozen_plan_sha256": sha_bytes(plans[0]), "migration_receipt": str(receipt_path.relative_to(root).as_posix()),
                        "migration_receipt_sha256": final_receipt_sha,
                        "scheduler_sources": scheduler_sources, "tasks": schedules[index],
                        "expected_new_rows": loads[index], "labels_read": False}
            atomic(path, manifest)
            workers.append({"worker_id": worker_id, "gpu": gpu, "task_manifest": path.relative_to(root).as_posix(),
                            "task_manifest_sha256": sha_bytes(path.read_bytes()), "expected_new_rows": loads[index],
                            "deferred_qualification": gpu in deferred})
        allocation = {"schema": "graph_credit_elastic_allocation_v1", "run_id": RUN_ID,
                      "logical_shards": 3, "workers": workers, "scheduler_sources": scheduler_sources,
                      "claim_mode": "shared_atomic_units",
                      "remaining_rows": remaining, "completed_rows": 60000 - remaining,
                      "frozen_plan_sha256": sha_bytes(plans[0]), "migration_receipt_sha256": final_receipt_sha,
                      "labels_read": False, "scientific_completion": False}
        atomic(folder / "allocation.json", allocation)
        atomic(receipt_path, receipt)
        atomic(root / "RUN_STATUS.json", {**state, "stage": "G1_ELASTIC_SCHEDULER_HANDOFF_COMPLETE",
                                         "active_pids": [], "legacy_supervisor_pid": state["pid"],
                                         "scientific_completion": False, "updated_at_unix": time.time()})
        print(json.dumps({"status": "HANDOFF_COMPLETE", "completed_rows": 60000 - remaining,
                          "remaining_rows": remaining, "gpus": gpus, "deferred_gpus": sorted(deferred),
                          "claim_mode": "shared_atomic_units", "labels_read": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
