#!/usr/bin/env python3
"""Draft elastic G1 finalization; imports the immutable original scorer/executor.

Default is validation only: no gold access and no artifact mutations. The caller
must quiesce every original and elastic writer before commit=True. This program
does not stop, start, or signal processes and never performs model inference.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import uuid


class FinalizationError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (canonical(value) + "\n").encode("utf-8")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FinalizationError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def decode_json(content):
    return json.loads(content, object_pairs_hook=_object)


def read_json(path):
    return decode_json(Path(path).read_bytes())


def safe_relative(base, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise FinalizationError(f"Invalid relative artifact path: {relative!r}")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(part in (".", "..", "") for part in rel.parts):
        raise FinalizationError(f"Artifact path traversal: {relative!r}")
    base = Path(base).resolve(strict=True)
    candidate = base.joinpath(*rel.parts)
    for part in (candidate, *candidate.parents):
        if part == base:
            break
        if part.is_symlink():
            raise FinalizationError(f"Symlink is not an immutable artifact: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise FinalizationError(f"Artifact escapes source directory: {relative}")
    return resolved


def read_jsonl(path):
    content = Path(path).read_bytes()
    if content and not content.endswith(b"\n"):
        raise FinalizationError(f"JSONL lacks complete trailing newline: {path}")
    result = []
    for number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            raise FinalizationError(f"Blank JSONL record: {path}:{number}")
        value = decode_json(line)
        if not isinstance(value, dict):
            raise FinalizationError(f"JSONL record is not an object: {path}:{number}")
        result.append(value)
    return content, result


def _atomic(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _exclusive_backup(path, content):
    """Never overwrite a historical backup, including when bytes differ."""
    if path.exists():
        if path.read_bytes() != content:
            raise FinalizationError(f"Existing immutable backup differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _normalize_migration_backup(content):
    """Verify exactly the migration helper's complete-prefix normalization."""
    lines = content.splitlines(keepends=True)
    normalized = []
    for index, line in enumerate(lines):
        try:
            row = decode_json(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise FinalizationError("Invalid migration backup before the final partial append")
            return len(normalized), b"".join(normalized), line
        if not isinstance(row, dict):
            raise FinalizationError("Non-object in immutable migration backup")
        normalized.append(line.rstrip(b"\r\n") + b"\n")
    return len(normalized), b"".join(normalized), b""


def _verify_migration(root, runid, plan_hash, source_hash, originals, inflight_upper_bound):
    path = safe_relative(root, "artifacts/audit/elastic_migration_20261103.json")
    content = path.read_bytes()
    receipt = decode_json(content)
    if (receipt.get("schema") != "graph_credit_elastic_migration_v1" or receipt.get("state") != "COMPLETE"
            or receipt.get("run_id") != runid or receipt.get("labels_read") is not False
            or receipt.get("logical_shards") != 3 or receipt.get("frozen_plan_sha256") != plan_hash
            or receipt.get("source_manifest_sha256") != source_hash):
        raise FinalizationError("Migration receipt does not bind a complete label-free original-plan handoff")
    if receipt.get("migration_inflight_upper_bound") != inflight_upper_bound:
        raise FinalizationError("Unknown inflight-cost bound differs from migration receipt")
    expected = {f"results/pilot/{runid}/shard_{i:03d}/{name}": (i, name)
                for i in range(3) for name in ("raw_executions.jsonl", "physical_calls.jsonl")}
    entries = receipt.get("files", [])
    if not isinstance(entries, list) or len(entries) != 6 or {item.get("path") for item in entries} != set(expected):
        raise FinalizationError("Migration receipt must cover six distinct original raw/physical canonical files")
    verified = []
    for item in entries:
        index, name = expected[item["path"]]
        if item.get("logical_shard") != index:
            raise FinalizationError("Migration logical-shard identity mismatch")
        backup = safe_relative(root, item["backup_path"])
        allowed = root / "results/pilot" / runid / f"shard_{index:03d}" / "migration_20261103"
        if backup.parent != allowed.resolve(strict=True) or backup.name != name + ".original_bytes":
            raise FinalizationError("Migration full-bytes backup path does not match its original artifact")
        backup_bytes = backup.read_bytes()
        if digest(backup_bytes) != item.get("backup_sha256"):
            raise FinalizationError("Immutable migration full-bytes backup hash mismatch")
        rows, normalized, tail = _normalize_migration_backup(backup_bytes)
        if (digest(normalized) != item.get("normalized_sha256") or normalized != originals[(index, name)]
                or rows != item.get("valid_rows") or len(tail) != item.get("partial_tail_bytes")
                or (digest(tail) if tail else None) != item.get("partial_tail_sha256")):
            raise FinalizationError("Migration original/normalized/partial-tail binding mismatch")
        verified.append({**item, "verified": True})
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(content),
            "files": verified, "inflight_upper_bound": inflight_upper_bound}


def _design(plan):
    qids, seeds, pairs, contexts = (plan[key] for key in ("question_ids", "seeds", "edge_pairs", "base_masks"))
    if len(qids) != 200 or len(set(qids)) != 200:
        raise FinalizationError("This finalizer requires exactly 200 unique frozen questions")
    if seeds != [1701, 1702, 1703, 1704, 1705] or pairs != [[0, 4], [2, 8], [6, 10]]:
        raise FinalizationError("Unexpected frozen G1 seeds or ordered edge pairs")
    if plan.get("num_shards") != 3 or plan.get("intended_shards") != 3:
        raise FinalizationError("Original plan must preserve three logical shards")
    if {context["name"] for context in contexts} != {"full", "chain", "star", "fixed_random", "sequential_deletion"} or len(contexts) != 5:
        raise FinalizationError("Expected the original five graph backgrounds")
    context_masks = {}
    for context in contexts:
        mask = context["mask"]
        if len(mask) != 12 or any(type(v) is not int or v not in (0, 1) for v in mask):
            raise FinalizationError("Invalid frozen context mask")
        context_id = context["name"] + "_" + digest(canonical(mask).encode())[:12]
        context_masks[context_id] = (context["name"], mask)
    expected = {(qid, context_id, e, f, seed, cell)
                for qid in qids for context_id in context_masks for e, f in pairs
                for seed in seeds for cell in ("00", "01", "10", "11")}
    if len(expected) != 60000:
        raise FinalizationError("Frozen Cartesian design is not 60000 unique observations")
    return qids, context_masks, expected


def _row_key(row):
    try:
        if any(type(row[field]) is not int for field in ("edge_e", "edge_f", "seed")):
            raise FinalizationError("Non-integer intervention coordinates")
        return tuple(row[field] for field in ("question_id", "context_id", "edge_e", "edge_f", "seed", "cell"))
    except KeyError as exc:
        raise FinalizationError(f"Missing observation coordinate: {exc}") from exc


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise FinalizationError(f"Invalid nonnegative ledger metric {label}: {value!r}")
    return value


def _load_original_modules(root):
    sys.path.insert(0, str(root / "src"))
    name = "elastic_frozen_run_pilot_" + digest(str(root).encode())[:12]
    spec = importlib.util.spec_from_file_location(name, root / "scripts/run_pilot.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_guard(module, root, plan):
    actual = module.source_manifest(root)
    if actual != plan["source_manifest"] or digest(canonical(actual).encode()) != plan["source_manifest_sha256"]:
        raise FinalizationError("Original source/config tree differs from frozen plan")
    spec = plan.get("scorer") or "graph_credit.scoring:score_item"
    if spec != "graph_credit.scoring:score_item":
        raise FinalizationError("This draft supports only the original frozen GSM score_item")
    if module.scorer_identity(spec) != plan["scorer_identity"]:
        raise FinalizationError("Original scorer identity differs from frozen plan")


def _lineage_file(root, path, snapshots):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise FinalizationError("Scheduler lineage file escapes experiment root")
    path = safe_relative(root, path.relative_to(root).as_posix())
    content = path.read_bytes()
    snapshots[path] = digest(content)
    return decode_json(content), digest(content)


def _verify_allocation(root, runid, plan, plan_hash, allocation_path, original_rows):
    """Lock actual scheduler bytes and exact remaining original-plan units."""
    if allocation_path is None:
        raise FinalizationError("An immutable scheduler allocation is required")
    snapshots = {}
    allocation, allocation_hash = _lineage_file(root, allocation_path, snapshots)
    if (allocation.get("schema") != "graph_credit_elastic_allocation_v1" or allocation.get("run_id") != runid
            or allocation.get("logical_shards") != 3 or allocation.get("frozen_plan_sha256") != plan_hash
            or allocation.get("labels_read") is not False or allocation.get("claim_mode") != "shared_atomic_units"):
        raise FinalizationError("Unexpected allocation identity or ownership mode")
    receipt, receipt_hash = _lineage_file(root, root / "artifacts/audit/elastic_migration_20261103.json", snapshots)
    if (receipt_hash != allocation.get("migration_receipt_sha256") or receipt.get("state") != "COMPLETE"
            or receipt.get("frozen_plan_sha256") != plan_hash or receipt.get("labels_read") is not False):
        raise FinalizationError("Allocation migration receipt does not match actual bytes")
    scheduler = allocation.get("scheduler_sources")
    if not isinstance(scheduler, dict) or not scheduler:
        raise FinalizationError("Missing scheduler source manifest")
    for relative, expected in scheduler.items():
        path = safe_relative(root, relative)
        actual = digest(path.read_bytes())
        if actual != expected:
            raise FinalizationError("Actual scheduler source differs from allocation: " + relative)
        snapshots[path] = actual
    self_path = Path(__file__).resolve()
    if not self_path.is_relative_to(root) or scheduler.get(self_path.relative_to(root).as_posix()) != digest(self_path.read_bytes()):
        raise FinalizationError("This finalizer is not bound by the scheduler allocation")
    steps = [(context, e, f, cell) for context in plan["base_masks"] for e, f in plan["edge_pairs"]
             for cell in ("11", "01", "00", "10")]
    qindex = {qid: i for i, qid in enumerate(plan["question_ids"])}
    completed, last_legacy = collections.Counter(), {}
    for index in range(3):
        expected = [(qid, seed, context["name"], e, f, cell)
                    for qid in plan["question_ids"] if qindex[qid] % 3 == index
                    for seed in plan["seeds"] for context, e, f, cell in steps]
        rows = original_rows.get(index, [])
        coordinates = [(r["question_id"], r["seed"], r["context_family"], r["edge_e"], r["edge_f"], r["cell"]) for r in rows]
        if coordinates != expected[:len(rows)] or len(rows) > len(expected):
            raise FinalizationError("Original rows are not the exact migration prefix")
        for row in rows:
            key = (index, row["question_id"], row["seed"])
            completed[key] += 1
            last_legacy[key] = row
    owners, tasks = {}, None
    workers = allocation.get("workers")
    if not isinstance(workers, list) or not workers:
        raise FinalizationError("Allocation contains no physical workers")
    for worker in workers:
        worker_id = worker.get("worker_id")
        if not isinstance(worker_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", worker_id) or worker_id in owners:
            raise FinalizationError("Invalid or duplicate scheduler worker identity")
        path = safe_relative(root, worker["task_manifest"])
        manifest, manifest_hash = _lineage_file(root, path, snapshots)
        if (manifest_hash != worker.get("task_manifest_sha256") or manifest.get("schema") != "graph_credit_elastic_tasks_v1"
                or manifest.get("worker_id") != worker_id or manifest.get("run_id") != runid or manifest.get("logical_shards") != 3
                or manifest.get("frozen_plan_sha256") != plan_hash or manifest.get("scheduler_sources") != scheduler
                or manifest.get("migration_receipt_sha256") != receipt_hash or manifest.get("labels_read") is not False
                or manifest.get("claim_mode") != "shared_atomic_units"):
            raise FinalizationError("Worker task bytes/provenance differ from allocation")
        if tasks is None:
            tasks = manifest.get("tasks")
        if not isinstance(tasks, list) or manifest.get("tasks") != tasks:
            raise FinalizationError("Shared workers do not bind an identical global task list")
        if manifest.get("expected_new_rows") != sum(t["expected_new_rows"] for t in tasks):
            raise FinalizationError("Worker global coverage count mismatch")
        owners[worker_id] = {"manifest": manifest, "hash": manifest_hash, "path": path}
    task_map = {}
    for task in tasks:
        key = (task.get("logical_shard"), task.get("question_id"), task.get("seed"))
        index, qid, seed = key
        if (type(index) is not int or index not in range(3) or qid not in qindex or qindex[qid] % 3 != index
                or seed not in plan["seeds"] or key in task_map or type(task.get("start_index")) is not int
                or task["start_index"] != completed[key] or not 0 <= task["start_index"] < len(steps)
                or task.get("expected_new_rows") != len(steps) - task["start_index"]):
            raise FinalizationError("Invalid/duplicate/incorrectly resumed unit in global allocation")
        if task["start_index"]:
            prior = last_legacy[key]
            if task.get("resume_execution_path") != prior["execution_path"] or task.get("resume_execution_sha256") != prior["execution_sha256"]:
                raise FinalizationError("Task resumes from a different legacy execution")
        task_map[key] = task
    remaining = {(qindex[qid] % 3, qid, seed) for qid in plan["question_ids"] for seed in plan["seeds"]
                 if completed[(qindex[qid] % 3, qid, seed)] < len(steps)}
    if set(task_map) != remaining or allocation.get("remaining_rows") != sum(t["expected_new_rows"] for t in task_map.values()):
        raise FinalizationError("Allocation omits or adds remaining immutable units")
    return {"allocation": allocation, "allocation_hash": allocation_hash, "allocation_path": Path(allocation_path).resolve(),
            "owners": owners, "tasks": task_map, "steps": steps, "snapshots": snapshots,
            "unit_rows": collections.defaultdict(list), "claims": {}, "segments": {}}


def _verify_segment(root, runid, plan, plan_hash, index, directory, gate):
    worker_id = directory.name
    owner = gate["owners"].get(worker_id)
    if owner is None or directory.parent != root / "results/pilot" / runid / f"shard_{index:03d}/elastic_segments":
        raise FinalizationError("Segment has no immutable allocated owner")
    manifest, _ = _lineage_file(root, directory / "task_manifest.json", gate["snapshots"])
    runtime, _ = _lineage_file(root, directory / "run_manifest.json", gate["snapshots"])
    status, _ = _lineage_file(root, directory / "STATUS.json", gate["snapshots"])
    if manifest != owner["manifest"]:
        raise FinalizationError("Segment task object differs from its original task-file bytes")
    identities = {"worker_id": worker_id, "task_manifest_sha256": owner["hash"], "frozen_plan_sha256": plan_hash,
                  "protocol_hash": plan["protocol_hash"], "model_identity": plan["model_identity"],
                  "source_manifest": plan["source_manifest"], "source_manifest_sha256": plan["source_manifest_sha256"],
                  "scorer_identity": plan["scorer_identity"], "scheduler_sources": gate["allocation"]["scheduler_sources"],
                  "migration_receipt_sha256": gate["allocation"]["migration_receipt_sha256"], "labels_read": False}
    if any(runtime.get(key) != value for key, value in identities.items()):
        raise FinalizationError("Segment runtime model/protocol/source/scorer/task identity mismatch")
    if (status.get("worker_id") != worker_id or status.get("task_manifest_sha256") != owner["hash"]
            or status.get("phase") != "GENERATION_COMPLETE_UNSCORED" or status.get("generation_complete") is not True
            or status.get("scored") is not False or status.get("labels_read") is not False):
        raise FinalizationError("Segment writer is not complete and unscored")
    gate["segments"][(index, worker_id)] = status


def _verify_elastic_row(root, runid, index, directory, row, gate):
    worker_id = directory.name
    owner = gate["owners"][worker_id]
    key = (index, row["question_id"], row["seed"])
    task = gate["tasks"].get(key)
    cell_index = row.get("elastic_cell_index")
    if (task is None or row.get("elastic_task_manifest_sha256") != owner["hash"] or row.get("logical_shard") != index
            or row.get("elastic_unit_start_index") != task["start_index"] or type(cell_index) is not int
            or not task["start_index"] <= cell_index < len(gate["steps"])):
        raise FinalizationError("Elastic raw row has no correctly assigned task coordinate")
    context, e, f, cell = gate["steps"][cell_index]
    if (row.get("context_family"), row.get("edge_e"), row.get("edge_f"), row.get("cell")) != (context["name"], e, f, cell):
        raise FinalizationError("Elastic cell index differs from the original execution order")
    if key not in gate["claims"]:
        claim_id = digest(canonical(list(key)).encode())
        folder = root / "results/pilot" / runid / "elastic_claims"
        claim, claim_hash = _lineage_file(root, folder / (claim_id + ".json"), gate["snapshots"])
        done, _ = _lineage_file(root, folder / (claim_id + ".done.json"), gate["snapshots"])
        for value in (claim, done):
            if (value.get("worker_id") != worker_id or value.get("task_manifest_sha256") != owner["hash"]
                    or value.get("task") != task or value.get("labels_read") is not False):
                raise FinalizationError("Unit claim/completion owner or task identity mismatch")
        if (claim.get("schema") != "graph_credit_elastic_unit_claim_v1"
                or done.get("schema") != "graph_credit_elastic_unit_completion_v1" or done.get("claim_sha256") != claim_hash
                or done.get("new_rows") != task["expected_new_rows"] or done.get("segment") != directory.relative_to(root).as_posix()):
            raise FinalizationError("Unit completion is not bound to its exact exclusive claim")
        gate["claims"][key] = {"worker_id": worker_id, "done": done}
    if gate["claims"][key]["worker_id"] != worker_id:
        raise FinalizationError("A unit is split across multiple physical owners")
    gate["unit_rows"][key].append(row)


def _finish_lineage(gate):
    if set(gate["unit_rows"]) != set(gate["tasks"]):
        raise FinalizationError("Some allocated remaining units have no elastic observations")
    segment_rows = collections.Counter()
    for key, rows in gate["unit_rows"].items():
        task, claim = gate["tasks"][key], gate["claims"][key]
        ordered = sorted(rows, key=lambda row: row["elastic_cell_index"])
        if ([row["elastic_cell_index"] for row in ordered] != list(range(task["start_index"], len(gate["steps"])))
                or claim["done"].get("last_execution_sha256") != ordered[-1]["execution_sha256"]):
            raise FinalizationError("Unit observations or final completion execution are incomplete")
        segment_rows[(key[0], claim["worker_id"])] += len(rows)
    for key, status in gate["segments"].items():
        if status.get("segment_rows") != segment_rows[key]:
            raise FinalizationError("Segment completion support differs from actual raw rows")


def _unchanged_lineage(gate):
    for path, expected in gate["snapshots"].items():
        if digest(path.read_bytes()) != expected:
            raise FinalizationError("Scheduler/segment/claim provenance changed during finalization")


def _verify_execution(execution, row, expected_payload_hash, all_calls, used_fresh):
    for field, expected in (("execution_id", row["execution_id"]), ("question_id", row["question_id"]),
                            ("question_fingerprint", expected_payload_hash), ("mask", row["graph_mask"]),
                            ("seed", row["seed"]), ("protocol_hash", row["protocol_hash"]), ("answer", row["answer"])):
        if execution.get(field) != expected:
            raise FinalizationError(f"Execution/observation mismatch: {field}")
    nodes = execution.get("nodes", {})
    if set(nodes) != {"aggregate", *(f"r{r}:w{w}" for r in (1, 2) for w in range(4))}:
        raise FinalizationError("Expected four workers, two rounds and fixed aggregator")
    if nodes["aggregate"]["generation"]["text"] != row["service_output"]:
        raise FinalizationError("Raw service output was changed")
    physical = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                "gpu_seconds": 0.0, "model_elapsed_seconds": 0.0}
    for name, node in nodes.items():
        generation = node["generation"]
        call_id = generation["call_id"]
        if call_id not in all_calls or all_calls[call_id]["status"] != "SUCCESS":
            raise FinalizationError(f"Node refers to absent or unsuccessful paid call: {call_id}")
        call = all_calls[call_id]
        expected_seed = int(digest(canonical([row["question_id"], row["seed"], name]).encode())[:16], 16) % (2**31 - 1)
        if node["seed"] != expected_seed or call["seed"] != node["seed"]:
            raise FinalizationError("Node/paid-ledger seed mismatch")
        if call["prompt_sha256"] != digest(canonical(node["messages"]).encode()):
            raise FinalizationError("Node/paid-ledger prompt hash mismatch")
        for field in ("input_tokens", "output_tokens", "finish_reason"):
            if call[field] != generation[field]:
                raise FinalizationError(f"Node/paid-ledger {field} mismatch")
        if type(node.get("reused")) is not bool:
            raise FinalizationError("Node must explicitly declare cache reuse")
        if not node["reused"]:
            if call_id in used_fresh:
                raise FinalizationError("A physical call is claimed as newly executed in multiple observations")
            used_fresh.add(call_id)
            physical["calls"] += 1
            for field in ("input_tokens", "output_tokens", "gpu_seconds"):
                physical[field] += _number(generation[field], field)
            physical["model_elapsed_seconds"] += _number(generation["elapsed_seconds"], "elapsed_seconds")
    for field, value in physical.items():
        for accounting in (execution.get("physical_cost", {}), row.get("physical_cost", {})):
            observed = accounting.get(field)
            if type(observed) not in (int, float) or not math.isclose(observed, value, rel_tol=1e-9, abs_tol=1e-7):
                raise FinalizationError(f"Observation/execution physical cost mismatch: {field}")


def _paid_summary(calls, used_fresh, migration_inflight_upper_bound):
    success = [row for row in calls if row["status"] == "SUCCESS"]
    failed = [row for row in calls if row["status"] == "FAILED"]
    orphan = [row for row in success if row["call_id"] not in used_fresh]
    def totals(records):
        return {"calls": len(records),
                "input_tokens": sum(_number(row["input_tokens"], "input_tokens") for row in records),
                "output_tokens": sum(_number(row["output_tokens"], "output_tokens") for row in records),
                "gpu_seconds": sum(_number(row["gpu_seconds"], "gpu_seconds") for row in records),
                "elapsed_seconds": sum(_number(row["elapsed_seconds"], "elapsed_seconds") for row in records)}
    return {
        "recorded_success_all_paid": totals(success),
        "recorded_success_associated_with_fresh_nodes": totals([r for r in success if r["call_id"] in used_fresh]),
        "recorded_success_orphaned_from_completed_graphs": totals(orphan),
        "orphaned_completed_call_ids": [row["call_id"] for row in orphan],
        "recorded_failed_calls": len(failed),
        "failed_call_costs": [{"call_id": row["call_id"], "input_tokens": row.get("input_tokens"),
                               "output_tokens": row.get("output_tokens"), "gpu_seconds": row.get("gpu_seconds"),
                               "elapsed_seconds": row.get("elapsed_seconds"), "cost_incomplete": True} for row in failed],
        "migration_unrecorded_inflight": {"count_upper_bound": migration_inflight_upper_bound,
                                          "exact_count": None, "input_tokens": None, "output_tokens": None,
                                          "gpu_seconds": None, "elapsed_seconds": None,
                                          "cost_status": "UNKNOWN_NOT_ZERO_NOT_IMPUTED"},
        "scope": "G1 original and elastic physical ledgers; S0 and message-sanity costs remain separate",
    }


def finalize(root, runid, *, commit=False, migration_inflight_upper_bound=3, lineage_manifest_path=None):
    """Validate, then explicitly commit one frozen G1 run; never infer or signal.

    Caller precondition for commit=True: every original/elastic writer is
    quiescent and scheduler ownership is settled. Complete receipts are
    idempotent; partial commits refuse an automatic second merge or rescoring.
    """
    root = Path(root).resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", runid) or runid in (".", ".."):
        raise FinalizationError("Invalid run ID")
    if type(migration_inflight_upper_bound) is not int or not 0 <= migration_inflight_upper_bound <= 3:
        raise FinalizationError("Unrecorded migration inflight bound must be an explicit integer in [0,3]")
    parent = root / "results/pilot" / runid
    if not parent.resolve(strict=True).is_relative_to(root):
        raise FinalizationError("Run directory escapes root")
    receipt_path = parent / "ELASTIC_FINALIZATION.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("phase") != "COMPLETE":
            raise FinalizationError("Partial finalization exists; manual audit is required, no automatic retry")
        required_artifacts = {"all_observations.jsonl", "ELASTIC_PAID_COSTS.json", "ELASTIC_ALL_OUTPUTS_FROZEN.json"}
        for i in range(3):
            required_artifacts.update(f"shard_{i:03d}/{name}" for name in (
                "raw_executions.jsonl", "physical_calls.jsonl", "observations.jsonl", "OUTPUTS_FROZEN.json",
                "SCORING.json", "STATUS.json", "original_raw_executions.jsonl", "original_physical_calls.jsonl"))
        if (receipt.get("schema") != "elastic_g1_finalization_v1" or receipt.get("run_id") != runid
                or receipt.get("observations") != 60000 or receipt.get("scientific_completion") is not False
                or not required_artifacts <= set(receipt.get("committed_artifact_hashes", {}))):
            raise FinalizationError("Incomplete/invalid COMPLETE finalization receipt")
        for rel, expected_hash in receipt["committed_artifact_hashes"].items():
            if digest(safe_relative(parent, rel).read_bytes()) != expected_hash:
                raise FinalizationError(f"Completed finalization artifact changed: {rel}")
        return {**receipt, "idempotent_already_complete": True}
    shards = [parent / f"shard_{i:03d}" for i in range(3)]
    plan_bytes = [(shard / "frozen_plan.json").read_bytes() for shard in shards]
    if any(content != plan_bytes[0] for content in plan_bytes):
        raise FinalizationError("Logical shards do not share identical original plan bytes")
    plan = decode_json(plan_bytes[0])
    plan_hash = digest(plan_bytes[0])
    protocol = plan["protocol_hash"]
    qids, contexts, expected = _design(plan)
    qindex = {qid: index for index, qid in enumerate(qids)}
    payloads = dict(zip(qids, plan["question_payload_sha256"]))
    if len(payloads) != 200:
        raise FinalizationError("Frozen question-payload hashes are incomplete")
    module = _load_original_modules(root)
    _source_guard(module, root, plan)
    question_bytes, questions = read_jsonl(root / "data/derived/motivation_questions.jsonl")
    if digest(question_bytes) != plan["data_sha256"] or [q["id"] for q in questions] != qids:
        raise FinalizationError("Actual frozen question file/ordering differs from original plan")
    for question in questions:
        if digest(canonical(module.Executor.question_payload(question)).encode()) != payloads[question["id"]]:
            raise FinalizationError("Actual question payload differs from frozen plan")
    snapshots = {}
    originals = {}
    shard_rows = collections.defaultdict(list)
    shard_calls = collections.defaultdict(list)
    all_calls = {}
    raw_sources = []
    elastic_directories = []
    # Read ledgers once, from canonical original paths and actual segment paths;
    # historical backups are never counted as additional paid calls.
    for index, shard in enumerate(shards):
        if (shard / "SCORING.json").exists() or (shard / "observations.jsonl").exists():
            raise FinalizationError("Original scoring already exists; do not overwrite or reopen this lineage")
        sources = [(shard, False)]
        segments = shard / "elastic_segments"
        if segments.exists():
            sources += [(p, True) for p in sorted(segments.iterdir()) if p.is_dir()]
        for directory, elastic in sources:
            if elastic:
                elastic_directories.append((index, directory))
            for filename in ("raw_executions.jsonl", "physical_calls.jsonl"):
                path = directory / filename
                if not path.exists():
                    if elastic:
                        continue
                    raise FinalizationError(f"Missing original artifact: {path}")
                safe_relative(shard, path.relative_to(shard).as_posix())
                content, records = read_jsonl(path)
                snapshots[path] = digest(content)
                if not elastic:
                    originals[(index, filename)] = content
                if filename == "raw_executions.jsonl":
                    raw_sources.append((index, directory, path, records))
                else:
                    for call in records:
                        call_id = call.get("call_id")
                        if not isinstance(call_id, str) or not call_id or call_id in all_calls:
                            raise FinalizationError("Missing/duplicate physical call ID across paid ledgers")
                        if call.get("status") not in ("SUCCESS", "FAILED"):
                            raise FinalizationError("Unknown physical ledger call status")
                        all_calls[call_id] = call
                        shard_calls[index].append(call)
    legacy_rows = {i: next(rows for index, directory, _, rows in raw_sources if index == i and directory == shards[i]) for i in range(3)}
    lineage_gate = _verify_allocation(root, runid, plan, plan_hash, lineage_manifest_path, legacy_rows)
    for index, directory in elastic_directories:
        _verify_segment(root, runid, plan, plan_hash, index, directory, lineage_gate)
    seen = set()
    used_fresh = set()
    mappings = []
    for index, directory, source_path, rows in raw_sources:
        for row in rows:
            if directory != shards[index]:
                _verify_elastic_row(root, runid, index, directory, row, lineage_gate)
            key = _row_key(row)
            if key not in expected:
                raise FinalizationError(f"Unknown/unplanned observation key: {key}")
            if key in seen:
                raise FinalizationError(f"Duplicate observation key (never silently deduplicated): {key}")
            seen.add(key)
            qid, context_id, e, f, seed, cell = key
            if qindex[qid] % 3 != index:
                raise FinalizationError("Observation belongs to a different original logical shard")
            if row.get("protocol_hash") != protocol or row.get("frozen_plan_sha256") != plan_hash:
                raise FinalizationError("Observation plan/protocol differs from original frozen lineage")
            if row.get("task") != "gsm8k" or row.get("split") not in ("dev", "motivation") or row.get("quality") is not None:
                raise FinalizationError("Only unscored GSM development raw outputs may enter finalization")
            family, base_mask = contexts[context_id]
            mask = list(base_mask)
            mask[e], mask[f] = int(cell[0]), int(cell[1])
            if row.get("graph_mask") != mask or row.get("context_family") != family:
                raise FinalizationError("Actual graph/context differs from frozen intervention")
            if row.get("mask_e") != int(cell[0]) or row.get("mask_f") != int(cell[1]):
                raise FinalizationError("Cell coordinates disagree with raw mask flags")
            original_relative = row["execution_path"]
            execution_path = safe_relative(directory, original_relative)
            execution_bytes = execution_path.read_bytes()
            if digest(execution_bytes) != row.get("execution_sha256"):
                raise FinalizationError("Execution-file SHA256 differs from frozen raw observation")
            execution = decode_json(execution_bytes)
            _verify_execution(execution, row, payloads[qid], all_calls, used_fresh)
            relative = execution_path.relative_to(shards[index]).as_posix()
            mapped = {**row, "execution_path": relative}
            if original_relative != relative:
                mappings.append({"logical_shard": index, "key": list(key),
                                 "source_raw": source_path.relative_to(parent).as_posix(),
                                 "old_execution_path": original_relative, "new_execution_path": relative,
                                 "execution_sha256": row["execution_sha256"]})
            shard_rows[index].append(mapped)
    if seen != expected:
        missing = sorted(expected - seen)
        raise FinalizationError(f"Incomplete frozen coverage: {len(seen)}/60000; missing={len(missing)}, first={missing[:5]}")
    _finish_lineage(lineage_gate)
    for index in range(3):
        if len(shard_rows[index]) != (67 if index < 2 else 66) * 300:
            raise FinalizationError("Logical-shard observation support is incomplete")
        shard_rows[index].sort(key=lambda row: (qindex[row["question_id"]], *_row_key(row)[1:]))
    migration = _verify_migration(root, runid, plan_hash, plan["source_manifest_sha256"], originals,
                                  migration_inflight_upper_bound)
    paid = _paid_summary(list(all_calls.values()), used_fresh, migration_inflight_upper_bound)
    paid["by_logical_shard"] = {f"shard_{i:03d}": _paid_summary(shard_calls[i], used_fresh, 0)
                                 for i in range(3)}
    for summary in paid["by_logical_shard"].values():
        summary["migration_unrecorded_inflight"] = {"exact_count": None, "cost_status": "UNKNOWN; global upper bound is not allocated to shards"}
    for path, expected_hash in snapshots.items():
        if digest(path.read_bytes()) != expected_hash:
            raise FinalizationError(f"Input writer is not quiescent: {path}")
    _unchanged_lineage(lineage_gate)
    lineage = None
    if lineage_manifest_path is not None:
        lineage_path = Path(lineage_manifest_path).resolve(strict=True)
        lineage = {"path": str(lineage_path), "sha256": digest(lineage_path.read_bytes())}
    receipt = {
        "schema": "elastic_g1_finalization_v1", "phase": "VALIDATED_NOT_COMMITTED", "run_id": runid,
        "original_plan_sha256": plan_hash, "protocol_hash": protocol,
        "original_source_manifest_sha256": plan["source_manifest_sha256"], "scorer_identity": plan["scorer_identity"],
        "finalizer_source_sha256": digest(Path(__file__).read_bytes()), "scheduler_lineage": lineage,
        "scheduler_input_hashes": {p.relative_to(root).as_posix(): value for p, value in lineage_gate["snapshots"].items()},
        "verified_exclusive_units": len(lineage_gate["unit_rows"]),
        "migration_receipt": migration,
        "scientific_completion": False, "labels_opened": False, "observations": len(seen),
        "coverage": {"questions": 200, "seeds": 5, "contexts": 5, "edge_pairs": 3, "cells": 4, "unique_rows": 60000},
        "input_hashes": {p.relative_to(parent).as_posix(): value for p, value in snapshots.items()},
        "execution_path_remappings": mappings, "paid_costs": paid,
        "caller_precondition": "all original and elastic writers quiescent; scheduler ownership settled",
        "created_unix": time.time(), "committed_artifact_hashes": {},
    }
    if not commit:
        return receipt
    # Backups here preserve normalized legacy canonical bytes. The separately
    # verified migration receipt retains the complete pre-normalization bytes
    # and every partial tail, which must never be silently assigned zero cost.
    for index, shard in enumerate(shards):
        for name in ("raw_executions.jsonl", "physical_calls.jsonl"):
            backup = shard / ("original_" + name)
            if backup.exists() and backup.read_bytes() != originals[(index, name)]:
                raise FinalizationError(f"Existing normalized-legacy backup differs: {backup}")
    receipt["phase"] = "COMMITTING_BEFORE_LABELS"
    _atomic(receipt_path, json_bytes(receipt))
    try:
        for index, shard in enumerate(shards):
            for name in ("raw_executions.jsonl", "physical_calls.jsonl"):
                _exclusive_backup(shard / ("original_" + name), originals[(index, name)])
            merged_raw = b"".join(json_bytes(row) for row in shard_rows[index])
            merged_calls = b"".join(json_bytes(call) for call in shard_calls[index])
            _atomic(shard / "raw_executions.jsonl", merged_raw)
            _atomic(shard / "physical_calls.jsonl", merged_calls)
            frozen = {"outputs": len(shard_rows[index]), "raw_executions_sha256": digest(merged_raw),
                      "physical_calls_sha256": digest(merged_calls), "labels_opened": False,
                      "frozen_plan_sha256": plan_hash, "elastic_finalizer_sha256": receipt["finalizer_source_sha256"],
                      "original_raw_sha256": digest(originals[(index, "raw_executions.jsonl")]),
                      "original_physical_calls_sha256": digest(originals[(index, "physical_calls.jsonl")])}
            _atomic(shard / "OUTPUTS_FROZEN.json", json_bytes(frozen))
        freeze_path = parent / "ELASTIC_ALL_OUTPUTS_FROZEN.json"
        _atomic(freeze_path, json_bytes({"original_plan_sha256": plan_hash, "outputs": 60000,
                                       "labels_opened": False, "timestamp": time.time(),
                                       "logical_shards": [read_json(s / "OUTPUTS_FROZEN.json") for s in shards]}))
        _source_guard(module, root, plan)
        _unchanged_lineage(lineage_gate)
        if digest(safe_relative(root, migration["path"]).read_bytes()) != migration["sha256"]:
            raise FinalizationError("Migration receipt changed before gold access")
        # First gold access occurs only after every logical-shard raw file and
        # the global 60000-output freeze marker have been durably published.
        identifier = plan.get("label_path_identifier")
        label_path = Path(identifier) if identifier else root / "data/derived/motivation_labels.jsonl"
        if not label_path.is_absolute():
            label_path = root / label_path
        label_path = label_path.resolve(strict=True)
        if not label_path.is_relative_to(root):
            raise FinalizationError("Frozen label path escapes experiment root")
        receipt["labels_opened"] = True
        receipt["phase"] = "OPENING_LABELS_AFTER_GLOBAL_FREEZE"
        _atomic(receipt_path, json_bytes(receipt))
        label_bytes, labels = read_jsonl(label_path)
        label_by_id = {}
        for label in labels:
            if label["id"] in label_by_id:
                raise FinalizationError("Duplicate gold ID")
            label_by_id[label["id"]] = label
        if set(label_by_id) != set(qids):
            raise FinalizationError("Gold ID coverage differs from frozen 200-question design")
        receipt["phase"] = "SCORING_FROZEN_OUTPUTS"
        receipt["labels_sha256"] = digest(label_bytes)
        _atomic(receipt_path, json_bytes(receipt))
        scorer = module.load_scorer("graph_credit.scoring:score_item")
        combined = []
        for index, shard in enumerate(shards):
            scored = []
            for row in shard_rows[index]:
                quality = float(scorer(row["service_output"], label_by_id[row["question_id"]], task="gsm8k"))
                if not math.isfinite(quality) or not 0 <= quality <= 1:
                    raise FinalizationError("Frozen scorer returned invalid quality")
                scored.append({**row, "quality": quality})
            observations_bytes = b"".join(json_bytes(row) for row in scored)
            _atomic(shard / "observations.jsonl", observations_bytes)
            combined.extend(scored)
            _atomic(shard / "SCORING.json", json_bytes({
                "labels_sha256": digest(label_bytes), "diagnostic_scoring_count": len(scored),
                "natural_feedback_count": 0, "scorer": "graph_credit.scoring:score_item",
                "scorer_identity": plan["scorer_identity"], "source_manifest_sha256": plan["source_manifest_sha256"],
                "raw_executions_sha256": digest((shard / "raw_executions.jsonl").read_bytes()),
                "labels_used_to_change_service_outputs": False,
                "global_outputs_frozen_before_label_access": True, "scientific_completion": False}))
            status = read_json(shard / "STATUS.json") if (shard / "STATUS.json").exists() else {}
            status.update(phase="GENERATION_COMPLETE_AWAITING_INDEPENDENT_GATE", generation_complete=True,
                          scored=True, outputs_frozen=len(scored), workflow_complete=False,
                          scientific_completion=False, elastic_finalization=True,
                          successful_physical_calls=sum(call["status"] == "SUCCESS" for call in shard_calls[index]))
            _atomic(shard / "STATUS.json", json_bytes(status))
        _atomic(parent / "all_observations.jsonl", b"".join(json_bytes(row) for row in combined))
        _atomic(parent / "ELASTIC_PAID_COSTS.json", json_bytes(paid))
        _source_guard(module, root, plan)
        artifacts = [parent / "all_observations.jsonl", parent / "ELASTIC_PAID_COSTS.json", freeze_path]
        for shard in shards:
            artifacts.extend(shard / name for name in ("raw_executions.jsonl", "physical_calls.jsonl", "observations.jsonl",
                                                       "OUTPUTS_FROZEN.json", "SCORING.json", "STATUS.json",
                                                       "original_raw_executions.jsonl", "original_physical_calls.jsonl"))
        receipt["committed_artifact_hashes"] = {p.relative_to(parent).as_posix(): digest(p.read_bytes()) for p in artifacts}
        receipt.update(phase="COMPLETE", completed_unix=time.time(), workflow_complete=False,
                       boundary="Generation/ledger merge and original-scorer output complete; independent G1 analysis/audit still required")
        _atomic(receipt_path, json_bytes(receipt))
        return receipt
    except Exception as exc:
        receipt.update(phase="FAILED_REQUIRES_MANUAL_AUDIT", error_type=type(exc).__name__, error=str(exc))
        _atomic(receipt_path, json_bytes(receipt))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", action="store_true", help="Only after all original/elastic writers are quiescent")
    parser.add_argument("--migration-inflight-upper-bound", type=int, default=3)
    parser.add_argument("--lineage-manifest", type=Path)
    args = parser.parse_args()
    result = finalize(args.root, args.run_id, commit=args.commit,
                      migration_inflight_upper_bound=args.migration_inflight_upper_bound,
                      lineage_manifest_path=args.lineage_manifest)
    print(canonical({"phase": result["phase"], "observations": result["observations"],
                     "labels_opened": result["labels_opened"], "scientific_completion": False}))


if __name__ == "__main__":
    main()
