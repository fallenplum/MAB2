"""Execution-only GPU pool for the immutable three-logical-shard G1 design.

This file lives outside src/scripts/configs.  Its SHA is locked in a separate
scheduler allocation before the first elastic observation is generated.
"""
from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:  # Pure scheduling helpers are testable on the Windows owner.
    fcntl = None
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN_ID = "motivation_v3_20261103"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".pooltmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def row_count(path):
    path = Path(path)
    if not path.exists():
        return 0
    # Append-only logs commit one newline-terminated record at a time. Full JSON
    # and lineage validation is performed by the independent finalizer.
    with path.open("rb") as handle:
        return sum(block.count(b"\n") for block in iter(lambda: handle.read(1048576), b""))


def progress(root):
    base = root / "results/pilot" / RUN_ID
    summaries = []
    for i in range(3):
        shard = base / f"shard_{i:03d}"
        count = row_count(shard / "raw_executions.jsonl")
        calls = row_count(shard / "physical_calls.jsonl")
        for segment in sorted((shard / "elastic_segments").glob("*")):
            if segment.is_dir():
                count += row_count(segment / "raw_executions.jsonl")
                calls += row_count(segment / "physical_calls.jsonl")
        prior = read(shard / "STATUS.json")
        prior.update(phase="GENERATING", outputs_frozen=count,
                     successful_physical_calls=calls, scheduler_revision="elastic_v1",
                     updated_at_unix=time.time(), generation_complete=False,
                     scored=False, scientific_completion=False)
        atomic(shard / "STATUS.json", prior)
        summaries.append({"logical_shard": i, "outputs_frozen": count,
                          "successful_physical_calls": calls})
    return summaries


def verify_scheduler(root, allocation):
    for relative, expected in allocation["scheduler_sources"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("Elastic scheduler source changed: " + relative)
    for worker in allocation["workers"]:
        if sha(root / worker["task_manifest"]) != worker["task_manifest_sha256"]:
            raise ValueError("Worker task manifest changed")


def query_gpu_free():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                   "--format=csv,noheader,nounits"], text=True)
    return {int(line.split(",")[0]): int(line.split(",")[1])
            for line in raw.splitlines() if line.strip()}


def has_unclaimed_units(root, allocation):
    """Mirror elastic_worker's immutable O_EXCL key; never steal active claims."""
    if allocation.get("claim_mode") != "shared_atomic_units":
        return False
    workers = allocation.get("workers", [])
    if not workers:
        return False
    manifest = read(root / workers[0]["task_manifest"])
    if manifest.get("claim_mode") != "shared_atomic_units":
        raise ValueError("Deferred admission requires the shared atomic unit protocol")
    folder = root / "results/pilot" / RUN_ID / "elastic_claims"
    for task in manifest["tasks"]:
        key = [task["logical_shard"], task["question_id"], task["seed"]]
        encoded = json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if not (folder / (hashlib.sha256(encoded).hexdigest() + ".json")).exists():
            return True
    return False


def _load_private(root, name):
    spec = importlib.util.spec_from_file_location("elastic_deferred_" + name,
                                                root / "artifacts/private/performance" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _successful_call_count(path, expected):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected or any(row.get("status") != "SUCCESS" for row in rows):
        raise ValueError(f"Expected {expected} successful real qualification calls")
    ids = [row["call_id"] for row in rows]
    if len(set(ids)) != expected:
        raise ValueError("Qualification physical call IDs are not unique")
    return len(rows)


def verify_deferred_qualification(root, gpu, allocation):
    """Nonempty 10-file / 110-node S0 signature comparison, no label access."""
    auditor = _load_private(root, "cross_gpu_audit")
    reference = root / "results/pilot/s0_v3_20261103"
    folder = root / "results/pilot" / f"elastic_qualification_gpu{gpu}_20261103"
    gate = read(folder / "QUALIFICATION.json")
    if (gate.get("status") != "PASS" or gate.get("eligible_for_G1") is not True
            or gate.get("execution_gate", {}).get("status") != "PASS"
            or gate.get("message_gate", {}).get("status") != "PASS"):
        raise ValueError("Deferred GPU execution/message qualification did not pass")
    reference_gate = read(reference / "QUALIFICATION.json")
    if reference_gate.get("status") != "PASS" or reference_gate.get("eligible_for_G1") is not True:
        raise ValueError("Original S0 reference gate is not PASS")
    before, actual = auditor.executions(reference), auditor.executions(folder)
    if (len(before) != 10 or len(actual) != 10
            or sum(len(value["nodes"]) for value in before.values()) != 110
            or sum(len(value["nodes"]) for value in actual.values()) != 110
            or before != actual):
        raise ValueError("Deferred GPU differs from nonempty original 10-file/110-node S0 signature")
    calls = _successful_call_count(folder / "physical_calls.jsonl", 96)
    if read(folder / "run_manifest.json")["protocol_hash"] != read(reference / "run_manifest.json")["protocol_hash"]:
        raise ValueError("Deferred GPU two-round protocol differs from original S0")
    return {"status": "PASS", "physical_calls": calls, "execution_files": 10,
            "node_occurrences": 110, "exact_signature_equal": True, "labels_read": False,
            "run_id": folder.name, "auditor_source_sha256": sha(root / "artifacts/private/performance/cross_gpu_audit.py")}


def verify_deferred_resume(root, gpu, allocation):
    folder = root / "results/pilot" / f"elastic_resume_gpu{gpu}_20261103"
    value = read(folder / "QUALIFICATION.json")
    worker_path = root / "artifacts/private/performance/elastic_worker.py"
    expected_worker_sha = allocation["scheduler_sources"].get("artifacts/private/performance/elastic_worker.py")
    plan_path = root / "results/pilot" / RUN_ID / "shard_000/frozen_plan.json"
    plan = read(plan_path)
    if (value.get("status") != "PASS" or value.get("labels_read") is not False
            or value.get("exact_node_signature_equal") is not True
            or value.get("physical_calls") != 2 or value.get("expected_closure_calls") != 2
            or value.get("gpu") != gpu or value.get("protocol_hash") != plan["protocol_hash"]
            or value.get("original_frozen_plan_sha256") != allocation["frozen_plan_sha256"]
            or sha(plan_path) != allocation["frozen_plan_sha256"]
            or not expected_worker_sha or sha(worker_path) != expected_worker_sha
            or value.get("worker_source_sha256") != expected_worker_sha
            or value.get("probe_source_sha256") != sha(root / "artifacts/private/performance/elastic_resume_qualification.py")):
        raise ValueError("Deferred GPU resume qualification/provenance did not pass")
    calls = _successful_call_count(folder / "physical_calls.jsonl", 2)
    auditor = _load_private(root, "cross_gpu_audit")
    expected = read(root / "results/pilot/s0_v3_20261103/qualification_executions/r2_switch_closure.json")
    if auditor.signature(read(folder / "resumed_execution.json")) != auditor.signature(expected):
        raise ValueError("Deferred deserialized resume signature differs from original closure")
    return {"status": "PASS", "physical_calls": calls, "worker_source_sha256": expected_worker_sha,
            "protocol_hash": plan["protocol_hash"], "labels_read": False, "run_id": folder.name}


class DeferredAdmission:
    """GPU3 admission state machine; injected callbacks permit CPU-only tests."""
    def __init__(self, root, allocation, worker, spawn, start_worker, *,
                 unclaimed=None, memory=None, verify=None, qualify=None, resume=None):
        self.root, self.allocation, self.worker = root, allocation, worker
        self.spawn, self.start_worker = spawn, start_worker
        self.unclaimed = unclaimed or (lambda: has_unclaimed_units(root, allocation))
        self.memory = memory or query_gpu_free
        self.verify = verify or (lambda: verify_scheduler(root, allocation))
        self.qualify = qualify or (lambda: verify_deferred_qualification(root, worker["gpu"], allocation))
        self.resume = resume or (lambda: verify_deferred_resume(root, worker["gpu"], allocation))
        self.process = None
        self.last_memory_check = float("-inf")
        self.info = {"gpu": worker["gpu"], "worker_id": worker["worker_id"],
                     "phase": "WAITING_FOR_FREE_MEMORY", "qualification_attempts": 0,
                     "resume_attempts": 0, "worker_started": False, "labels_read": False,
                     "history": [], "pending_qualification_pid": None}

    def mark(self, phase, **values):
        self.info.update(phase=phase, **values)
        self.info["history"].append({"phase": phase, "observed_unix": time.time(), **values})

    def _launch(self, phase):
        gpu = self.worker["gpu"]
        self.verify()
        if phase == "QUALIFYING":
            run_id = f"elastic_qualification_gpu{gpu}_20261103"
            if (self.root / "results/pilot" / run_id).exists():
                raise ValueError("Deferred qualification run already exists; no inference retry")
            command = [sys.executable, "-u", "scripts/run_pilot.py", "--root", str(self.root),
                       "--data", str(self.root / "data/derived/qualification_questions.jsonl"),
                       "--model", str(self.root / "models/Qwen2.5-3B-Instruct"), "--devices", "0",
                       "--config", str(self.root / "configs/qualification_v3.json"), "--run-id", run_id,
                       "--qualification-only", "--limit", "1"]
            self.info["qualification_attempts"] += 1
        else:
            run_id = f"elastic_resume_gpu{gpu}_20261103"
            if (self.root / "results/pilot" / run_id).exists():
                raise ValueError("Deferred resume run already exists; no inference retry")
            command = [sys.executable, "-u", str(self.root / "artifacts/private/performance/elastic_resume_qualification.py"),
                       "--root", str(self.root), "--gpu-label", str(gpu)]
            self.info["resume_attempts"] += 1
        self.process = self.spawn(command, gpu, f"elastic_deferred_gpu{gpu}_{phase.lower()}")
        self.mark(phase, pending_qualification_pid=self.process.pid, pending_run_id=run_id)

    def tick(self, *, formal_running, now=None):
        now = time.monotonic() if now is None else now
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                return
            old_phase = self.info["phase"]
            self.process = None
            self.info["pending_qualification_pid"] = None
            if code != 0:
                self.mark("EXCLUDED", failed_phase=old_phase, returncode=code, reason="Qualification process failed; no retry")
                return
            try:
                if old_phase == "QUALIFYING":
                    self.info["execution_qualification"] = self.qualify()
                    if not formal_running or not self.unclaimed():
                        self.mark("QUALIFIED_BUT_NO_UNCLAIMED_WORK")
                        return
                    self._launch("RESUME_QUALIFYING")
                else:
                    self.info["resume_qualification"] = self.resume()
                    if not formal_running or not self.unclaimed():
                        self.mark("QUALIFIED_BUT_NO_UNCLAIMED_WORK")
                        return
                    self.verify()  # Includes exact allocation-bound worker SHA.
                    self.start_worker(self.worker)
                    self.mark("FORMAL_WORKER_STARTED", worker_started=True)
            except Exception as exc:
                self.mark("EXCLUDED", failed_phase=old_phase, reason=str(exc), error_type=type(exc).__name__)
            return
        if self.info["phase"] != "WAITING_FOR_FREE_MEMORY":
            return
        try:
            if not formal_running or not self.unclaimed():
                self.mark("SKIPPED_NO_UNCLAIMED_WORK")
                return
            if now - self.last_memory_check < 60:
                return
            self.last_memory_check = now
            try:
                free = self.memory()
                self.info["last_memory_query_error"] = None
            except Exception as exc:
                # Read-only telemetry errors never stop the paid seven-card work.
                self.info["last_memory_query_error"] = str(exc)
                return
            self.info.update(last_free_mib=free.get(self.worker["gpu"], 0), last_memory_check_unix=time.time())
            if self.info["last_free_mib"] >= 12000:
                self._launch("QUALIFYING")
        except Exception as exc:
            self.mark("EXCLUDED", reason=str(exc), error_type=type(exc).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--allocation", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if fcntl is None:
        raise RuntimeError("The compute supervisor requires Linux flock")
    os.chdir(root)
    allocation = read(args.allocation)
    if allocation["run_id"] != RUN_ID or allocation["logical_shards"] != 3:
        raise ValueError("Unexpected frozen experiment")
    verify_scheduler(root, allocation)
    lock = (root / "artifacts/supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / "src"), CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    status_path = root / "RUN_STATUS.json"
    state = {"schema": "graph_credit_elastic_supervisor_v1", "pid": os.getpid(),
             "scientific_completion": False, "shared_gpu_authorized": True,
             "inference_retries": 0, "scope": "S0_then_G1_only", "s0": "PASS",
             "s0_execution": "PASS", "expected_questions": 200,
             "expected_graph_observations": 60000, "expected_seeds": 5,
             "logical_shards": 3, "scheduler_revision": "elastic_v1",
             "allocation_sha256": sha(args.allocation), "gpus": [], "active_pids": []}
    children = []
    handles = []
    deferred = []
    qa_children = []

    def update(stage, **kwargs):
        state.update(stage=stage, updated_at_unix=time.time(), **kwargs)
        atomic(status_path, state)

    def spawn(command, gpu, label):
        run_env = env.copy()
        run_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        stdout = (root / "logs" / (label + ".stdout.log")).open("ab")
        stderr = (root / "logs" / (label + ".stderr.log")).open("ab")
        handles.extend([stdout, stderr])
        process = subprocess.Popen(command, env=run_env, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr)
        return process

    def spawn_qualification(command, gpu, label):
        process = spawn(command, gpu, label)
        qa_children.append(process)
        return process

    def start_worker(worker):
        if any(prior["worker_id"] == worker["worker_id"] for prior, _ in children):
            raise ValueError("Formal worker was already started; no inference retry")
        command = [sys.executable, "-u", str(root / "artifacts/private/performance/elastic_worker.py"),
                   "--root", str(root), "--task-manifest", str(root / worker["task_manifest"]),
                   "--worker-id", worker["worker_id"], "--device", "0"]
        process = spawn(command, worker["gpu"], "elastic_" + worker["worker_id"])
        children.append((worker, process))
        state["gpus"].append(worker["gpu"])
        return process

    def publish_pool(stage, **values):
        running = [(w, p) for w, p in children if p.poll() is None]
        pending = [{"pid": entry.process.pid, "gpu": entry.worker["gpu"],
                    "worker_id": entry.worker["worker_id"], "phase": entry.info["phase"],
                    "run_id": entry.info.get("pending_run_id"), "role": "deferred_qualification"}
                   for entry in deferred if entry.process is not None and entry.process.poll() is None]
        admissions = [dict(entry.info) for entry in deferred]
        update(stage, active_pids=[p.pid for _, p in running],
               active_workers=[{"pid": p.pid, "gpu": w["gpu"], "worker_id": w["worker_id"],
                                "task_manifest": w["task_manifest"]} for w, p in running],
               pending_qualifications=pending, deferred_gpu_admission=admissions, **values)
        if deferred:
            atomic(root / "artifacts/audit/elastic_deferred_gpu_admission.json", {
                "schema": "elastic_deferred_gpu_admission_v1", "observed_unix": time.time(),
                "allocation_sha256": state["allocation_sha256"], "labels_read": False,
                "admissions": admissions, "pending_qualifications": pending,
                "qualification_cost_sources": [
                    {"gpu": entry.worker["gpu"], "run_id": run_id,
                     "physical_ledger": str((root / "results/pilot" / run_id / "physical_calls.jsonl").relative_to(root)),
                     "exists": (root / "results/pilot" / run_id / "physical_calls.jsonl").exists(),
                     "cost_not_in_formal_G1_ledger": True}
                    for entry in deferred for run_id in (
                        f"elastic_qualification_gpu{entry.worker['gpu']}_20261103",
                        f"elastic_resume_gpu{entry.worker['gpu']}_20261103")],
                "scientific_completion": False})

    try:
        update("G1_ELASTIC_GPU_ADMISSION")
        for worker in allocation["workers"]:
            flag = worker.get("deferred_qualification", False)
            if type(flag) is not bool:
                raise ValueError("deferred_qualification must be boolean")
            if flag:
                if worker["gpu"] != 3 or allocation.get("claim_mode") != "shared_atomic_units":
                    raise ValueError("Only shared-unit GPU3 deferred qualification is authorized")
                deferred.append(DeferredAdmission(root, allocation, worker, spawn_qualification, start_worker))
        for worker in allocation["workers"]:
            if worker.get("deferred_qualification", False):
                continue  # Never block the seven already-qualified cards on GPU3.
            gpu = worker["gpu"]
            while True:
                free = query_gpu_free()
                if free.get(gpu, 0) >= 12000:
                    break
                publish_pool("G1_ELASTIC_GPU_ADMISSION", waiting_gpu=gpu, free_mib=free)
                time.sleep(20)
            start_worker(worker)
            publish_pool("G1_REAL_GENERATION")
        while True:
            codes = [p.poll() for _, p in children]
            if any(code not in (None, 0) for code in codes):
                # Settle only owned workers. Do not restart failures or inspect labels.
                publish_pool("FAILED_CHILDREN_SETTLING", child_returncodes=codes)
                while any(p.poll() is None for _, p in children) or any(p.poll() is None for p in qa_children):
                    for entry in deferred:
                        entry.tick(formal_running=False)  # Settle paid QA, never add a card after failure.
                    publish_pool("FAILED_CHILDREN_SETTLING")
                    time.sleep(20)
                raise RuntimeError("Elastic worker failed; paid artifacts retained")
            formal_running = any(code is None for code in codes)
            for entry in deferred:
                entry.tick(formal_running=formal_running)
            formal_running = any(p.poll() is None for _, p in children)
            pending = any(p.poll() is None for p in qa_children)
            if not formal_running and not pending:
                publish_pool("G1_REAL_GENERATION", formal_generation_finished=True,
                             child_returncodes=[p.poll() for _, p in children])
                break
            summaries = progress(root)
            publish_pool("G1_REAL_GENERATION", child_returncodes=[p.poll() for _, p in children],
                         shard_progress=summaries, formal_generation_finished=not formal_running,
                         waiting_for_owned_qualification=not formal_running and pending)
            time.sleep(20)
        if any(p.returncode for _, p in children):
            raise RuntimeError("Elastic worker failed")
        verify_scheduler(root, allocation)
        update("G1_ELASTIC_FINALIZING", active_pids=[])
        spec = importlib.util.spec_from_file_location("elastic_finalize", root / "artifacts/private/performance/elastic_finalize.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.finalize(root, RUN_ID, commit=False, migration_inflight_upper_bound=3,
                        lineage_manifest_path=args.allocation)
        module.finalize(root, RUN_ID, commit=True, migration_inflight_upper_bound=3,
                        lineage_manifest_path=args.allocation)
        base = root / "results/pilot" / RUN_ID
        result = subprocess.run([sys.executable, "scripts/analyze_pilot.py", "--observations",
                                 str(base / "all_observations.jsonl"), "--plan",
                                 str(base / "shard_000/frozen_plan.json"), "--config",
                                 str(root / "configs/gate_v1.json"), "--output", str(base / "G1_ANALYSIS.json")],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        (root / "logs/g1_elastic_analysis.stdout.log").write_text(result.stdout)
        (root / "logs/g1_elastic_analysis.stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError("Frozen independent analysis failed")
        update("G1_GENERATION_AND_ANALYSIS_COMPLETE_REVIEW_REQUIRED", active_pids=[],
               workflow_complete=False, result=str(base / "G1_ANALYSIS.json"))
        return 0
    except Exception as exc:
        if any(p.poll() is None for _, p in children) or any(p.poll() is None for p in qa_children):
            publish_pool("FAILED_CHILDREN_SETTLING", error_type=type(exc).__name__, error=str(exc))
            while any(p.poll() is None for _, p in children) or any(p.poll() is None for p in qa_children):
                for entry in deferred:
                    entry.tick(formal_running=False)
                publish_pool("FAILED_CHILDREN_SETTLING")
                time.sleep(20)
        update("FAILED", error_type=type(exc).__name__, error=str(exc), workflow_complete=False)
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
