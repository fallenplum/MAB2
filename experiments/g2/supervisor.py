"""Qualify admitted GPUs, collect one frozen G2 pool, then freeze all outputs."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from collector import unit_id


MIN_FREE_MIB = 12000
POLL_SECONDS = 10
STOP_GRACE_SECONDS = 120
TERM_GRACE_SECONDS = 20


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON field " + key)
        value[key] = item
    return value


def read_json(path):
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=_unique_object)
    require(isinstance(value, dict), "Expected a JSON object")
    return value


def safe_path(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
            "Expected a relative POSIX project path")
    parts = relative.split("/")
    require(all(part not in ("", ".", "..") for part in parts), "Unsafe project path component")
    path = root.joinpath(*parts)
    require(path.resolve().is_relative_to(root), "Project path escapes its root")
    return path


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".supervisortmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((canonical(value) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_manifest(root, manifest):
    require(isinstance(manifest, dict) and manifest, "An explicit source manifest is required")
    for relative, identity in manifest.items():
        path = safe_path(root, relative)
        require(path.suffix in (".py", ".json") and "label" not in path.name.lower(),
                "Source manifests cannot request label/data file access")
        require(path.is_file() and path.stat().st_size == identity["bytes"]
                and file_sha(path) == identity["sha256"], "Frozen source mismatch " + relative)


def load_frozen_plan(root, plan_path):
    """Validate bytes and sources without importing model code or reading labels."""
    root = Path(root).resolve(strict=True)
    path = Path(plan_path).resolve(strict=True)
    plan = read_json(path)
    require(plan.get("schema") == "graph_credit_g2_plan_v1" and plan.get("status") == "PLAN_FROZEN",
            "Only the final PLAN_FROZEN G2 schema can run")
    run_id = plan.get("run_id")
    require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", run_id), "Unsafe G2 run ID")
    output = safe_path(root, "results/g2/" + run_id)
    require(plan.get("output_dir") == output.relative_to(root).as_posix()
            and path == output / "frozen_plan.json", "Plan must use its canonical G2 output path")
    plan_hash = file_sha(path)
    marker = read_json(output / "PLAN_FROZEN.json")
    source_hash = digest(plan["source_manifest"])
    pipeline_hash = digest(plan["pipeline_source_manifest"])
    require(marker.get("state") == "PLAN_FROZEN" and marker.get("run_id") == run_id
            and marker.get("plan_sha256") == plan_hash and marker.get("labels_read") is False
            and plan.get("labels_read") is False
            and marker.get("source_manifest_sha256") == plan.get("source_manifest_sha256") == source_hash
            and marker.get("pipeline_source_manifest_sha256") == pipeline_hash,
            "Plan/source/pipeline freeze marker binding failed")
    require(type(marker.get("created_unix")) in (int, float) and math.isfinite(marker["created_unix"])
            and marker["created_unix"] > 0, "Plan freeze timestamp is missing or invalid")
    validate_manifest(root, plan["source_manifest"])
    validate_manifest(root, plan["pipeline_source_manifest"])
    for script in (Path(__file__).resolve(), Path(__file__).resolve().with_name("collector.py")):
        require(script.is_relative_to(root) and script.relative_to(root).as_posix() in plan["pipeline_source_manifest"],
                "Running supervisor and collector must be in the frozen pipeline manifest")
    feature_prefix = "artifacts/private/g2/" + run_id + "/"
    require(plan["features"]["path"] == feature_prefix + "question_features.npz"
            and plan["features"]["receipt"] == feature_prefix + "FEATURES_FROZEN.json",
            "Only canonical label-free feature artifacts may be opened")
    feature_path = safe_path(root, plan["features"]["path"])
    feature_receipt = read_json(safe_path(root, plan["features"]["receipt"]))
    require(file_sha(feature_path) == marker.get("features_sha256") == feature_receipt.get("feature_sha256") == plan["features"]["sha256"]
            and file_sha(safe_path(root, plan["features"]["receipt"])) == plan["features"]["receipt_sha256"]
            and feature_receipt.get("status") == "PASS_LABEL_FREE"
            and feature_receipt.get("pca_fit_questions") == 200 and feature_receipt.get("dimensions") == 9
            and feature_receipt.get("unlabeled_design_rank") == 711
            and feature_receipt.get("labels_read") is False, "Frozen label-free features do not match")
    partitions = plan["partitions"]
    require(Counter(partitions.values()) == {"train": 200, "tune": 80, "test": 120}, "G2 partition sizes changed")
    units = plan["units"]
    actual = {(unit["question_id"], unit["seed"]) for unit in units}
    expected = {(qid, seed) for qid in partitions for seed in (2701, 2702, 2703, 2704, 2705)}
    require(len(units) == plan.get("expected_units") == 2000 and len(actual) == 2000 and actual == expected,
            "The exact 400-question, five-seed pool is required")
    require(all(unit["partition"] == partitions[unit["question_id"]] for unit in units), "Unit partition mismatch")
    require(plan.get("formal_call_upper_bound") == 124000 and plan.get("expected_outputs", 52001) <= 52000,
            "Formal generation budget differs from the approved design")
    require(plan["graph"].get("rounds") == 2, "The fresh-call bound requires two rounds")
    return plan, plan_hash, pipeline_hash, output


def expected_graph_keys(plan):
    keys = set()
    for unit in plan["units"]:
        for request in unit["mask_requests"]:
            mask = request["mask"]
            require(isinstance(mask, list) and len(mask) == 12
                    and all(type(bit) is int and bit in (0, 1) for bit in mask), "Invalid planned graph mask")
            keys.add((unit["question_id"], unit["seed"], digest(mask)))
    require(len(keys) == plan["expected_outputs"], "Frozen unique graph count does not match its units")
    return keys


def verify_collection_coverage(output, plan, plan_hash):
    """Independent coverage gate before the collector's deeper byte audit."""
    expected = expected_graph_keys(plan)
    expected_units = {unit_id(unit["question_id"], unit["seed"]): unit for unit in plan["units"]}
    require(len(expected_units) == len(plan["units"]), "Collection unit IDs collide")
    folder = output / "claims"
    require(folder.is_dir(), "Formal claims are missing")
    actual_claims = {path.stem for path in folder.glob("*.json") if not path.name.endswith(".done.json")}
    actual_done = {path.name[:-len(".done.json")] for path in folder.glob("*.done.json")}
    require(actual_claims == actual_done == set(expected_units), "All 2000 claims must have exactly one done receipt")
    for key, unit in expected_units.items():
        claim_path = folder / (key + ".json")
        claim, done = read_json(claim_path), read_json(folder / (key + ".done.json"))
        require(claim.get("plan_sha256") == done.get("plan_sha256") == plan_hash
                and claim.get("schema") == "g2_collection_claim_v1"
                and claim.get("question_id") == done.get("question_id") == unit["question_id"]
                and claim.get("seed") == done.get("seed") == unit["seed"]
                and claim.get("worker_id") == done.get("worker_id")
                and done.get("unit_id") == key and done.get("schema") == "g2_collection_done_v1"
                and done.get("claim_sha256") == file_sha(claim_path)
                and claim.get("labels_read") is False and done.get("labels_read") is False,
                "A claim/done binding is invalid")
        expected_path = "segments/" + done["worker_id"] + "/units/" + key + "/UNIT_OUTPUTS_FROZEN.json"
        require(done.get("unit_frozen_path") == expected_path, "Done receipt points to a different unit owner")
        frozen_path = safe_path(output, expected_path)
        frozen = read_json(frozen_path)
        count = len({digest(request["mask"]) for request in unit["mask_requests"]})
        require(file_sha(frozen_path) == done.get("unit_frozen_sha256")
                and frozen.get("plan_sha256") == plan_hash and frozen.get("protocol_hash") == plan["protocol_hash"]
                and frozen.get("question_id") == unit["question_id"] and frozen.get("seed") == unit["seed"]
                and frozen.get("observations") == done.get("observations") == count
                and frozen.get("labels_read") is False, "A completed unit is not bound to its frozen outputs")
    observed = set()
    for raw in sorted((output / "segments").glob("*/raw_executions.jsonl")):
        with raw.open("rb") as handle:
            for line in handle:
                require(line.endswith(b"\n") and line.strip(), "Uncommitted or blank raw row")
                row = json.loads(line, object_pairs_hook=_unique_object)
                require(type(row.get("seed")) is int, "Raw graph seed must be an integer")
                mask = row["graph_mask"]
                key = (row["question_id"], row["seed"], digest(mask))
                require(key in expected and key not in observed, "Unknown or duplicate raw graph key")
                require(row.get("plan_sha256") == plan_hash and row.get("protocol_hash") == plan["protocol_hash"]
                        and row.get("source_manifest_sha256") == plan["source_manifest_sha256"]
                        and row.get("partition") == plan["partitions"][row["question_id"]]
                        and row.get("mask_sha256") == key[2] and row.get("quality") is None
                        and row.get("labels_read") is False, "Raw row identity/label isolation mismatch")
                observed.add(key)
    require(observed == expected, "Formal raw rows do not exactly cover all frozen graph keys")
    return {"units": len(expected_units), "unique_raw_graphs": len(observed), "status": "PASS_LABEL_FREE_COVERAGE"}


def query_gpus():
    """Record device UUID, free memory and compute PIDs, never foreign commands."""
    def query(fields, entity):
        result = subprocess.run(["nvidia-smi", "--query-" + entity + "=" + fields,
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        require(result.returncode == 0, "nvidia-smi resource query failed")
        return [[part.strip() for part in line.split(",")] for line in result.stdout.splitlines() if line.strip()]
    devices = {}
    for index, gpu_uuid, free in query("index,uuid,memory.free", "gpu"):
        devices[int(index)] = {"gpu_index": int(index), "gpu_uuid": gpu_uuid, "free_mib": int(free), "compute_processes": []}
    by_uuid = {row["gpu_uuid"]: row for row in devices.values()}
    for gpu_uuid, pid, used in query("gpu_uuid,pid,used_gpu_memory", "compute-apps"):
        if gpu_uuid in by_uuid:
            by_uuid[gpu_uuid]["compute_processes"].append({"pid": int(pid), "used_memory_mib": int(used) if used.isdigit() else None})
    return devices


@dataclass
class OwnedChild:
    process: subprocess.Popen
    argv: list
    create_time: float | None
    owner: str
    logfile: object
    mode: str
    gpu: int | None

    def same_identity(self):
        import psutil
        if self.process.poll() is not None or self.create_time is None:
            return False
        try:
            process = psutil.Process(self.process.pid)
            return (process.create_time() == self.create_time and process.username() == self.owner
                    and process.cmdline() == self.argv)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False


class Supervisor:
    def __init__(self, root, plan_path, gpus, python):
        import psutil
        self.root = Path(root).resolve(strict=True)
        self.plan_path = Path(plan_path).resolve(strict=True)
        self.plan, self.plan_hash, self.pipeline_hash, self.output = load_frozen_plan(self.root, self.plan_path)
        self.gpus = gpus
        workers = self.plan["qualification"]["worker_ids"]
        require({"gpu" + str(gpu) for gpu in gpus} == set(workers) and len(workers) == len(gpus),
                "Physical GPU candidates must exactly match the frozen qualification pool")
        # Preserve a virtualenv interpreter symlink. Resolving it can silently
        # select the base interpreter and its different package environment.
        self.python = os.path.abspath(os.fspath(python))
        require(Path(self.python).is_file(), "Project Python must be an existing executable path")
        self.collector = Path(__file__).resolve().with_name("collector.py")
        self.owner = psutil.Process().username()
        self.process_create_time = psutil.Process().create_time()
        self.children, self.gpu_uuids = [], {}
        self.private = self.output / "private"
        self.private.mkdir(parents=True, exist_ok=True)
        self.state = {"schema": "graph_credit_g2_supervisor_v1", "run_id": self.plan["run_id"],
                      "pid": os.getpid(), "process_create_time": self.process_create_time, "owner_username": self.owner,
                      "plan_sha256": self.plan_hash, "source_manifest_sha256": self.plan["source_manifest_sha256"],
                      "pipeline_source_manifest_sha256": self.pipeline_hash, "labels_read": False,
                      "generation_complete": False, "scientific_completion": False, "workflow_complete": False,
                      "expected_units": 2000, "expected_outputs": self.plan["expected_outputs"]}
        require(not (self.output / "STOP_REQUESTED.json").exists(), "This run has a persistent stop request")
        require(not (self.output / "GLOBAL_OUTPUTS_FROZEN.json").exists(), "This G2 collection is already frozen")

    def status(self, phase, **fields):
        self.state.update(phase=phase, updated_unix=time.time(), **fields)
        self.state["active_children"] = [{"pid": child.process.pid, "mode": child.mode, "gpu": child.gpu,
                                           "create_time": child.create_time} for child in self.children if child.process.poll() is None]
        atomic_json(self.output / "STATUS.json", self.state)

    @contextmanager
    def lease(self):
        path = self.private / "SUPERVISOR_LEASE.json"
        record = {"run_id": self.plan["run_id"], "pid": os.getpid(), "create_time": self.process_create_time,
                  "owner": self.owner, "plan_sha256": self.plan_hash}
        content = (canonical(record) + "\n").encode("utf-8")
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            yield
        finally:
            if path.exists() and path.read_bytes() == content:
                path.unlink()

    def spawn(self, mode, gpu=None):
        import psutil
        argv = [self.python, "-u", str(self.collector), "--root", str(self.root), "--plan", str(self.plan_path), "--mode", mode]
        environment = os.environ.copy()
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[name] = "1"
        if gpu is not None:
            argv += ["--worker-id", "gpu" + str(gpu), "--device", "cuda:0"]
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        log = (self.private / (mode + ("_gpu" + str(gpu) if gpu is not None else "") + ".log")).open("xb")
        try:
            process = subprocess.Popen(argv, cwd=self.root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        except BaseException:
            log.close()
            raise
        try:
            created = psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess:
            created = None
        child = OwnedChild(process, argv, created, self.owner, log, mode, gpu)
        self.children.append(child)
        append_json(self.private / "CHILDREN.jsonl", {"pid": process.pid, "create_time": created,
                    "owner": self.owner, "argv": argv, "mode": mode, "gpu": gpu,
                    "plan_sha256": self.plan_hash, "started_unix": time.time()})
        return child

    def resources(self):
        devices = query_gpus()
        require(set(self.gpus) <= set(devices), "A requested physical GPU is absent")
        for gpu in self.gpus:
            current = devices[gpu]["gpu_uuid"]
            require(self.gpu_uuids.get(gpu, current) == current, "Physical GPU UUID changed")
            self.gpu_uuids[gpu] = current
        for child in self.children:
            if child.gpu is None or child.process.poll() is not None:
                continue
            actual = [device["gpu_uuid"] for device in devices.values()
                      if any(row["pid"] == child.process.pid for row in device["compute_processes"])]
            require(not actual or actual == [self.gpu_uuids[child.gpu]], "A collector process used a different physical GPU")
        append_json(self.private / "RESOURCE_SNAPSHOTS.jsonl", {"observed_unix": time.time(),
                    "plan_sha256": self.plan_hash, "devices": [devices[gpu] for gpu in self.gpus]})
        return devices

    def qualification_report(self, gpu):
        path = self.output / "qualification" / ("gpu" + str(gpu)) / "QUALIFICATION.json"
        report = read_json(path)
        require(report.get("status") == "PASS" and report.get("exact_replay") is True
                and report.get("run_id") == self.plan["run_id"] and report.get("worker_id") == "gpu" + str(gpu)
                and report.get("plan_sha256") == self.plan_hash and report.get("protocol_hash") == self.plan["protocol_hash"]
                and report.get("source_manifest_sha256") == self.plan["source_manifest_sha256"]
                and report.get("model_identity") == self.plan["model"]["identity"]
                and report.get("pipeline_source_manifest_sha256") == self.pipeline_hash
                and report.get("labels_read") is False
                and isinstance(report.get("signature_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", report["signature_sha256"]), "GPU qualification is not a bound PASS")
        return {"report_sha256": file_sha(path), "signature_sha256": report["signature_sha256"], "gpu_uuid": self.gpu_uuids[gpu]}

    def run_pool(self, mode):
        pending, active, completed = set(self.gpus), {}, {}
        while pending or active:
            for gpu, child in list(active.items()):
                code = child.process.poll()
                if code is None:
                    continue
                child.logfile.close()
                require(code == 0, "Collector child failed in " + mode + " on GPU " + str(gpu) + " exit " + str(code))
                completed[gpu] = self.qualification_report(gpu) if mode == "qualify" else {"exit_code": code}
                del active[gpu]
            if pending or active:
                devices = self.resources()
            if pending:
                for gpu in sorted(list(pending)):
                    if devices[gpu]["free_mib"] >= MIN_FREE_MIB:
                        active[gpu] = self.spawn(mode, gpu)
                        pending.remove(gpu)
            self.status("QUALIFYING" if mode == "qualify" else "COLLECTING", waiting_for_resource_gpus=sorted(pending),
                        successful_gpus=sorted(completed),
                        admission_minimum_free_mib=MIN_FREE_MIB)
            if pending or active:
                time.sleep(POLL_SECONDS)
        return completed

    def stop_owned_children(self, reason):
        """Ask for a graph-boundary stop, then act only on verified own children."""
        write_errors = []
        try:
            atomic_json(self.output / "STOP_REQUESTED.json", {"schema": "graph_credit_g2_stop_v1",
                        "run_id": self.plan["run_id"], "plan_sha256": self.plan_hash,
                        "reason": reason, "created_unix": time.time(), "labels_read": False})
        except OSError as error:
            write_errors.append(str(error))
        deadline = time.monotonic() + (0 if write_errors else STOP_GRACE_SECONDS)
        while any(child.process.poll() is None for child in self.children) and time.monotonic() < deadline:
            time.sleep(1)
        signalled, refused = [], []
        import psutil
        for child in self.children:
            if child.process.poll() is not None:
                continue
            if child.same_identity():
                try:
                    psutil.Process(child.process.pid).terminate()
                    signalled.append(child.process.pid)
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied:
                    refused.append(child.process.pid)
            else:
                refused.append(child.process.pid)
        deadline = time.monotonic() + TERM_GRACE_SECONDS
        while any(child.process.poll() is None for child in self.children) and time.monotonic() < deadline:
            time.sleep(1)
        for child in self.children:
            if child.process.poll() is None and child.same_identity():
                try:
                    psutil.Process(child.process.pid).kill()
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied:
                    refused.append(child.process.pid)
        failed = [child for child in self.children if child.gpu is not None and child.process.poll() not in (None, 0)]
        unsettled = [child.process.pid for child in self.children if child.process.poll() is None]
        unsettled_gpu = any(child.gpu is not None and child.process.pid in unsettled for child in self.children)
        receipt = {"plan_sha256": self.plan_hash,
                    "terminated_owned_pids": signalled, "identity_mismatch_pids_not_signalled": refused,
                    "unsettled_child_pids": unsettled,
                    "unverified_inflight_calls_upper_bound": None if unsettled_gpu else len({child.process.pid for child in failed} | set(signalled)),
                    "unverified_inflight_exact_count": None, "unverified_inflight_cost": None,
                    "persistence_errors": write_errors, "labels_read": False, "completed_unix": time.time()}
        try:
            atomic_json(self.private / "STOP_RECEIPT.json", receipt)
        except OSError as error:
            print(canonical({"status": "STOP_RECEIPT_WRITE_FAILED", "error": str(error), "stop_receipt": receipt}), file=sys.stderr)

    def run(self):
        with self.lease():
            try:
                self.status("QUALIFICATION_PENDING")
                reports = self.run_pool("qualify")
                require(len(reports) == len(self.gpus) and len({row["signature_sha256"] for row in reports.values()}) == 1,
                        "Every candidate GPU must pass the identical qualification signature")
                atomic_json(self.output / "GLOBAL_QUALIFICATION.json", {"schema": "graph_credit_g2_gpu_qualification_v1",
                            "status": "PASS", "run_id": self.plan["run_id"], "plan_sha256": self.plan_hash,
                            "source_manifest_sha256": self.plan["source_manifest_sha256"],
                            "pipeline_source_manifest_sha256": self.pipeline_hash,
                            "gpu_reports": {str(gpu): report for gpu, report in reports.items()}, "labels_read": False})
                self.run_pool("collect")
                coverage = verify_collection_coverage(self.output, self.plan, self.plan_hash)
                validate_manifest(self.root, self.plan["source_manifest"])
                validate_manifest(self.root, self.plan["pipeline_source_manifest"])
                self.status("FREEZING_OUTPUTS", coverage=coverage)
                child = self.spawn("freeze-outputs")
                while child.process.poll() is None:
                    self.status("FREEZING_OUTPUTS", coverage=coverage)
                    time.sleep(POLL_SECONDS)
                child.logfile.close()
                require(child.process.returncode == 0, "Collector global output freeze failed")
                frozen_path = self.output / "GLOBAL_OUTPUTS_FROZEN.json"
                frozen = read_json(frozen_path)
                require(frozen.get("plan_sha256") == self.plan_hash and frozen.get("labels_read") is False
                        and frozen.get("run_id") == self.plan["run_id"] and frozen.get("state") == "GLOBAL_OUTPUTS_FROZEN"
                        and frozen.get("completed_units") == frozen.get("expected_units") == 2000
                        and frozen.get("source_manifest_sha256") == self.plan["source_manifest_sha256"]
                        and frozen.get("pipeline_source_manifest_sha256") == self.pipeline_hash
                        and frozen.get("protocol_hash") == self.plan["protocol_hash"]
                        and frozen.get("observations") == self.plan["expected_outputs"], "Global output freeze is not bound and complete")
                self.status("WAITING_TRAIN_TUNE_LABELS", generation_complete=True,
                            global_outputs_frozen_sha256=file_sha(frozen_path), coverage=coverage)
                return 0
            except BaseException as error:
                try:
                    self.status("STOPPING_AFTER_FAILURE", error_type=type(error).__name__, error=str(error))
                except OSError:
                    pass
                try:
                    self.stop_owned_children(str(error))
                finally:
                    try:
                        self.status("FAILED", error_type=type(error).__name__, error=str(error))
                    except OSError:
                        pass
                return 2
            finally:
                for child in self.children:
                    if not child.logfile.closed:
                        child.logfile.close()


def _stop_signal(signum, frame):
    raise KeyboardInterrupt("Supervisor stop signal " + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU indices")
    parser.add_argument("--python", default=sys.executable, help="Project Python used by collector children")
    args = parser.parse_args()
    try:
        gpus = [int(value) for value in args.gpus.split(",")]
        require(gpus and min(gpus) >= 0 and len(gpus) == len(set(gpus)), "Physical GPU indices must be unique and nonnegative")
        for name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), _stop_signal)
        return Supervisor(args.root, args.plan, gpus, args.python).run()
    except Exception as error:
        print(canonical({"status": "FAILED_PREFLIGHT", "error_type": type(error).__name__, "error": str(error), "labels_read": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
