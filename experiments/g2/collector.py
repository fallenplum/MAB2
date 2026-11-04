"""Frozen G2 graph-output collection with exclusive question/seed ownership."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from types import SimpleNamespace
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_bytes(content):
    return hashlib.sha256(content).hexdigest()


def digest(value):
    return digest_bytes(canonical(value).encode("utf-8"))


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value, *, exclusive=False):
    """Commit a whole JSON artifact before an append-only index references it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (canonical(value) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".collectortmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            # All publishers of this destination are protected by an O_EXCL
            # question/seed claim. Refuse accidental reruns within that owner.
            if path.exists():
                raise FileExistsError("Committed collector artifact already exists")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest_bytes(content)


def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Collector paths must be relative POSIX paths")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("Invalid collector path component")
    path = root.joinpath(*parts)
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Collector artifact escapes project root")
    return path


def unique_masks(masks):
    """Return first-occurrence order and all prespecified comparison aliases."""
    if not isinstance(masks, list) or not masks:
        raise ValueError("A collection unit requires an explicit nonempty mask list")
    unique, positions = [], {}
    for index, mask in enumerate(masks):
        if (not isinstance(mask, list) or len(mask) != 12
                or any(type(bit) is not int or bit not in (0, 1) for bit in mask)):
            raise ValueError("Every mask must contain twelve binary integers")
        key = tuple(mask)
        if key not in positions:
            positions[key] = len(unique)
            unique.append({"mask": list(mask), "aliases": []})
        unique[positions[key]]["aliases"].append(index)
    return unique


def unique_mask_requests(requests):
    """Keep first execution order, merge aliases and honor named anchor bases."""
    if not isinstance(requests, list) or not requests:
        raise ValueError("An explicit nonempty mask_requests list is required")
    schedule, positions = [], {}
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError("Each mask request must be an object")
        mask = unique_masks([request.get("mask")])[0]["mask"]
        reference = request.get("reference_mask")
        if reference is not None:
            reference = unique_masks([reference])[0]["mask"]
            if tuple(reference) not in positions:
                raise ValueError("A requested reference mask must already have been executed")
        aliases = request.get("aliases")
        if not isinstance(aliases, list) or any(not isinstance(alias, dict) for alias in aliases):
            raise ValueError("Mask aliases must be an explicit list of metadata objects")
        key = tuple(mask)
        if key not in positions:
            positions[key] = len(schedule)
            schedule.append({"mask": mask, "reference_mask": reference, "aliases": [], "request_indices": []})
        target = schedule[positions[key]]
        target["aliases"].extend(aliases)
        target["request_indices"].append(index)
    return schedule


def unit_id(question_id, seed):
    if not isinstance(question_id, str) or type(seed) is not int:
        raise ValueError("Unit requires a string question ID and integer node seed")
    return digest([question_id, seed])[:20]


def claim_unit(output, question_id, seed, worker_id, plan_hash):
    folder = output / "claims"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (unit_id(question_id, seed) + ".json")
    record = {"schema": "g2_collection_claim_v1", "question_id": question_id, "seed": seed,
              "worker_id": worker_id, "plan_sha256": plan_hash, "pid": os.getpid(),
              "claimed_unix": time.time(), "labels_read": False}
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        # The exclusive owner may still be flushing its initial short JSON.
        # Waiting for that receipt never retries a model call or takes ownership.
        existing = None
        for attempt in range(20):
            try:
                existing = read_json(path)
                break
            except json.JSONDecodeError:
                if attempt == 19:
                    raise ValueError("Existing exclusive claim has no complete identity receipt")
                time.sleep(.01)
        if (existing.get("question_id") != question_id or existing.get("seed") != seed
                or existing.get("plan_sha256") != plan_hash):
            raise ValueError("Truncated unit hash collides with a different exclusive claim identity")
        return None
    return path


class StopRequested(RuntimeError):
    """A supervisor request stops at a committed graph boundary."""


def check_stop(context):
    path = context["output"] / "STOP_REQUESTED.json"
    if path.exists():
        value = read_json(path)
        if value.get("run_id") != context["plan"]["run_id"] or value.get("plan_sha256") != context["plan_hash"]:
            raise ValueError("Stop receipt belongs to a different frozen collection")
        raise StopRequested("Supervisor requested a stop at a committed graph boundary")


def require_unreleased_outputs(context):
    for name in ("GLOBAL_OUTPUTS_FROZEN.json", "FIRST_TRAIN_TUNE_LABELS_OPENED.json", "FIRST_HELDOUT_LABELS_OPENED.json"):
        if (context["output"] / name).exists():
            raise FileExistsError("Generation/output publication is closed after output freeze or gold access")


def collect_unit(executor, question, seed, mask_requests, segment, plan_hash, source_hash, *, partition=None, progress=None):
    """Execute each original mask once and keep every physical generation cost.

    The frozen Executor owns node fingerprints, union closures and per-node
    random streams. Exceptions propagate immediately. Partial committed graphs
    and any completed/failed uncommitted calls remain available for audit.
    """
    schedule = unique_mask_requests(mask_requests)
    key = unit_id(question["id"], seed)
    folder = segment / "units" / key
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / "UNIT_PLAN.json", {
        "schema": "g2_collection_unit_v1", "question_id": question["id"], "seed": seed,
        "plan_sha256": plan_hash, "protocol_hash": executor.protocol_hash,
        "mask_schedule": schedule, "labels_read": False,
    }, exclusive=True)
    rows, executions = [], {}
    for index, item in enumerate(schedule):
        mask = item["mask"]
        reference = item["reference_mask"]
        base = executions[tuple(reference)] if reference is not None else None
        result = executor.execute(question, mask, seed, base=base)
        executions[tuple(mask)] = result
        mask_hash = digest(mask)
        mask_id = "".join(map(str, mask))
        execution_relative = PurePosixPath("units") / key / "executions" / (mask_id + ".json")
        execution_path = segment.joinpath(*execution_relative.parts)
        execution_hash = atomic_json(execution_path, result.to_dict(), exclusive=True)
        raw_relative = PurePosixPath("units") / key / "raw" / (mask_id + ".json")
        row = {
            "schema": "g2_collected_graph_v1", "question_id": question["id"],
            "task": question.get("task", "gsm8k"), "split": question.get("split", "calibration"),
            "seed": seed, "unit_id": key, "mask_index": index, "mask_aliases": item["aliases"],
            "request_indices": item["request_indices"], "reference_mask": reference, "partition": partition,
            "graph_mask": mask, "mask_id": mask_id, "mask_sha256": mask_hash,
            "question_fingerprint": result.question_fingerprint,
            "protocol_hash": executor.protocol_hash, "plan_sha256": plan_hash,
            "source_manifest_sha256": source_hash, "execution_id": result.execution_id,
            "execution_path": execution_relative.as_posix(), "execution_sha256": execution_hash,
            "answer": result.answer, "service_output": result.nodes["aggregate"].generation.text,
            "service_message_metadata": result.nodes["aggregate"].message_metadata,
            "physical_cost": result.cost(), "logical_service_cost": result.cost(False),
            "quality": None, "labels_read": False, "committed_unix": time.time(),
        }
        raw_hash = atomic_json(segment.joinpath(*raw_relative.parts), row, exclusive=True)
        index_record = {**row, "raw_path": raw_relative.as_posix(), "raw_sha256": raw_hash}
        append_json(segment / "raw_executions.jsonl", index_record)
        rows.append(index_record)
        if progress is not None:
            progress(index_record)
    atomic_json(folder / "UNIT_OUTPUTS_FROZEN.json", {
        "schema": "g2_collection_unit_outputs_v1", "question_id": question["id"], "seed": seed,
        "plan_sha256": plan_hash, "protocol_hash": executor.protocol_hash,
        "observations": len(rows), "rows": [{key: row[key] for key in
              ("mask_sha256", "raw_path", "raw_sha256", "execution_path", "execution_sha256")} for row in rows],
        "frozen_unix": time.time(), "first_gold_opened_unix": None, "labels_read": False,
    }, exclusive=True)
    return rows


def import_frozen(root):
    sys.path.insert(0, str(root / "src"))
    module = importlib.import_module("graph_credit.executor")
    backend = importlib.import_module("graph_credit.backend")
    if Path(module.__file__).resolve() != root / "src/graph_credit/executor.py":
        raise ValueError("Imported graph_credit belongs to a different project root")
    return module, backend.HFBackend


def verify_manifest(root, manifest, *, expected_count=None):
    if not isinstance(manifest, dict) or not manifest or expected_count is not None and len(manifest) != expected_count:
        raise ValueError("Unexpected source manifest support")
    for relative, expected in manifest.items():
        path = safe_path(root, relative)
        expected_hash = expected.get("sha256") if isinstance(expected, dict) else expected
        if file_digest(path) != expected_hash:
            raise ValueError("Frozen source bytes changed: " + relative)
        if isinstance(expected, dict) and "bytes" in expected and path.stat().st_size != expected["bytes"]:
            raise ValueError("Frozen source size changed: " + relative)


def read_questions(root, descriptor, units, module):
    path = safe_path(root, descriptor["path"])
    if file_digest(path) != descriptor["sha256"]:
        raise ValueError("Unlabelled question-file SHA mismatch")
    questions = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            module.Executor.question_payload(row)
            if row["id"] in questions:
                raise ValueError("Duplicate question ID")
            questions[row["id"]] = row
    for unit in units:
        qid = unit["question_id"]
        if qid not in questions or descriptor["payload_sha256"].get(qid) != module.digest(module.Executor.question_payload(questions[qid])):
            raise ValueError("Question payload does not match the frozen plan")
    return questions


def validate_design_coverage(plan):
    """The frozen partition map is question ID to train/tune/test."""
    partitions = plan["partitions"]
    if (Counter(partitions.values()) != {"train": 200, "tune": 80, "test": 120}
            or len(plan["units"]) != 2000
            or {unit["question_id"] for unit in plan["units"]} != set(partitions)):
        raise ValueError("Collection units do not cover the frozen 200/80/120 question split")
    for qid, partition in partitions.items():
        units = [unit for unit in plan["units"] if unit["question_id"] == qid]
        if ({unit["seed"] for unit in units} != {2701, 2702, 2703, 2704, 2705}
                or any(unit["partition"] != partition or unit["mask_requests"] != plan["designs"][qid]["mask_requests"] for unit in units)):
            raise ValueError("Question units differ from their frozen five-seed design")
        for request in plan["designs"][qid]["mask_requests"]:
            if plan["masks"].get("".join(map(str, request["mask"]))) != request["mask"]:
                raise ValueError("Global mask lookup differs from an execution request")


def load_plan(root, plan_path):
    plan_path = plan_path.resolve(strict=True)
    if not plan_path.is_relative_to(root):
        raise ValueError("Plan file must belong to the project root")
    plan = read_json(plan_path)
    if (plan.get("schema") != "graph_credit_g2_plan_v1" or plan.get("labels_read") is not False
            or plan.get("status") != "PLAN_FROZEN"):
        raise ValueError("Unexpected G2 collection plan schema or feedback permission")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", plan["run_id"]):
        raise ValueError("Invalid G2 run ID")
    if plan["qualification"]["worker_ids"] != [f"gpu{i}" for i in range(8)]:
        raise ValueError("Frozen physical qualification pool must contain the eight declared workers")
    verify_manifest(root, plan["source_manifest"], expected_count=18)
    verify_manifest(root, plan["pipeline_source_manifest"])
    if plan["source_manifest_sha256"] != digest(plan["source_manifest"]):
        raise ValueError("Declared source manifest digest is inconsistent")
    self_hash = file_digest(Path(__file__))
    collector_sources = [value for name, value in plan["pipeline_source_manifest"].items() if PurePosixPath(name).name == "collector.py"]
    if len(collector_sources) != 1 or (collector_sources[0].get("sha256") if isinstance(collector_sources[0], dict) else collector_sources[0]) != self_hash:
        raise ValueError("Executing collector is not bound by the pipeline manifest")
    module, backend_class = import_frozen(root)
    values = dict(plan["graph"])
    if "role_prompts" in values:
        values["role_prompts"] = tuple(values["role_prompts"])
    config = module.GraphConfig(**values)
    if config.rounds != 2:
        raise ValueError("The approved formal G2 call budget requires two rounds")
    model = plan["model"]
    safe_path(root, model["path"])
    if model["identity"].get("model_files") != model["manifest"]:
        raise ValueError("Plan model identity and file manifest disagree")
    expected_executor = module.Executor(SimpleNamespace(identity=model["identity"]), config)
    if expected_executor.protocol_hash != plan["protocol_hash"]:
        raise ValueError("Plan protocol hash differs from the frozen executor/model/config")
    seen, expected_graphs, call_bound = {}, 0, 0
    for unit in plan["units"]:
        key = unit_id(unit["question_id"], unit["seed"])
        if key in seen:
            raise ValueError("Duplicate question/seed unit or truncated unit hash collision")
        seen[key] = (unit["question_id"], unit["seed"])
        if unit.get("partition") not in ("train", "tune", "test"):
            raise ValueError("G2 unit requires train/tune/test metadata")
        schedule = unique_mask_requests(unit["mask_requests"])
        expected_graphs += len(schedule)
        for item in schedule:
            call_bound += config.n_workers * config.rounds + 1 if item["reference_mask"] is None else len(module.affected_closure(item["reference_mask"], item["mask"], config.rounds))
    if expected_graphs > 52000 or call_bound > 124000:
        raise ValueError("Frozen G2 graph/call upper bound exceeds the approved collection budget")
    if plan["expected_units"] != len(seen) or plan["expected_outputs"] != expected_graphs:
        raise ValueError("Frozen design and declared output coverage disagree")
    if plan["formal_call_upper_bound"] != 124000:
        raise ValueError("Formal physical call budget differs from the approved design")
    questions = read_questions(root, plan["questions"], plan["units"], module)
    validate_design_coverage(plan)
    for prerequisite in plan["qualification"]["prerequisites"]:
        path = safe_path(root, prerequisite["path"])
        if file_digest(path) != prerequisite["sha256"] or read_json(path).get("status") != "PASS":
            raise ValueError("Pinned real v3 qualification prerequisite does not pass")
    for path_key, hash_key in (("path", "sha256"), ("receipt", "receipt_sha256")):
        if file_digest(safe_path(root, plan["features"][path_key])) != plan["features"][hash_key]:
            raise ValueError("Label-free frozen feature artifact changed")
    output = safe_path(root, plan["output_dir"])
    if plan["output_dir"] != "results/g2/" + plan["run_id"] or plan_path != output / "frozen_plan.json":
        raise ValueError("Only the canonical G2 frozen plan and output directory may run")
    return {"root": root, "plan_path": plan_path, "plan": plan, "plan_hash": file_digest(plan_path),
            "module": module, "backend_class": backend_class, "config": config,
            "questions": questions, "output": output,
            "expected_graphs": expected_graphs, "call_upper_bound": call_bound,
            "source_hash": digest(plan["source_manifest"]), "pipeline_hash": digest(plan["pipeline_source_manifest"])}


def require_frozen(context):
    marker = read_json(context["output"] / "PLAN_FROZEN.json")
    expected = {"state": "PLAN_FROZEN", "run_id": context["plan"]["run_id"],
                "plan_sha256": context["plan_hash"], "labels_read": False,
                "source_manifest_sha256": context["source_hash"],
                "pipeline_source_manifest_sha256": context["pipeline_hash"],
                "features_sha256": context["plan"]["features"]["sha256"]}
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("Plan must be frozen before model loading or generation")
    return marker


def load_backend(context, device, physical_log):
    backend = context["backend_class"](safe_path(context["root"], context["plan"]["model"]["path"]),
                                       device, physical_log)
    if backend.identity != context["plan"]["model"]["identity"]:
        raise ValueError("Actual frozen model/tokenizer/software identity mismatch")
    executor = context["module"].Executor(backend, context["config"])
    if executor.protocol_hash != context["plan"]["protocol_hash"]:
        raise ValueError("Actual executor protocol mismatch")
    return backend, executor


def runtime_manifest(context, backend, worker_id, mode):
    return {"schema": "g2_collection_runtime_v1", "worker_id": worker_id, "mode": mode,
            "run_id": context["plan"]["run_id"],
            "plan_sha256": context["plan_hash"], "source_manifest_sha256": context["source_hash"],
            "pipeline_source_manifest_sha256": context["pipeline_hash"], "model_identity": backend.identity,
            "protocol_hash": context["plan"]["protocol_hash"], "runtime": backend.runtime,
            "created_unix": time.time(), "labels_read": False, "first_gold_opened_unix": None}


def verify_qualification_pool(context):
    signatures = set()
    for worker_id in context["plan"]["qualification"].get("worker_ids", [f"gpu{i}" for i in range(8)]):
        report = read_json(context["output"] / "qualification" / worker_id / "QUALIFICATION.json")
        if (report.get("status") != "PASS" or report.get("exact_replay") is not True
                or report.get("plan_sha256") != context["plan_hash"] or report.get("labels_read") is not False
                or report.get("run_id") != context["plan"]["run_id"]
                or report.get("worker_id") != worker_id
                or report.get("model_identity") != context["plan"]["model"]["identity"]
                or report.get("protocol_hash") != context["plan"]["protocol_hash"]
                or report.get("source_manifest_sha256") != context["source_hash"]
                or report.get("pipeline_source_manifest_sha256") != context["pipeline_hash"]
                or not re.fullmatch(r"[0-9a-f]{64}", report.get("signature_sha256", ""))):
            raise ValueError("A physical GPU has no matching real collector qualification")
        signatures.add(report["signature_sha256"])
    if len(signatures) != 1:
        raise ValueError("Physical GPUs disagree on the frozen execution signature")


def execution_signature(execution):
    return {name: {"messages": node["messages"], "seed": node["seed"],
                   "raw": node["generation"]["text"], "wire": node["wire_message"],
                   "parsed": node["parsed"], "metadata": node["message_metadata"]}
            for name, node in execution["nodes"].items()}


def qualify(context, worker_id, device):
    require_frozen(context)
    require_unreleased_outputs(context)
    check_stop(context)
    settings = context["plan"]["qualification"]
    units = settings["units"]
    questions = read_questions(context["root"], settings["questions"], units, context["module"])
    segment = context["output"] / "qualification" / worker_id
    segment.mkdir(parents=True, exist_ok=False)
    backend, executor = load_backend(context, device, segment / "physical_calls.jsonl")
    atomic_json(segment / "run_manifest.json", runtime_manifest(context, backend, worker_id, "qualification"), exclusive=True)
    signature, replay_checks = [], []
    try:
        for unit in units:
            question = questions[unit["question_id"]]
            rows = collect_unit(executor, question, unit["seed"], unit["mask_requests"], segment,
                                context["plan_hash"], context["source_hash"], partition="qualification",
                                progress=lambda row: check_stop(context))
            by_mask = {tuple(row["graph_mask"]): row for row in rows}
            for row in rows:
                execution = read_json(segment / row["execution_path"])
                signature.append({"question_id": unit["question_id"], "seed": unit["seed"],
                                  "mask": row["graph_mask"], "nodes": execution_signature(execution)})
            for index in settings["independent_full_replay_request_indices"]:
                check_stop(context)
                mask = unit["mask_requests"][index]["mask"]
                saved = read_json(segment / by_mask[tuple(mask)]["execution_path"])
                repeated = executor.execute(question, mask, unit["seed"])
                atomic_json(segment / "independent_full_replays" / f"{unit_id(unit['question_id'], unit['seed'])}_{index}.json",
                            repeated.to_dict(), exclusive=True)
                replay_checks.append(execution_signature(saved) == execution_signature(repeated.to_dict()))
        if not replay_checks or not all(replay_checks):
            raise ValueError("Named-anchor cache and independent full execution disagree")
        report = {"schema": "g2_collector_qualification_v1", "status": "PASS", "worker_id": worker_id,
                  "run_id": context["plan"]["run_id"],
                  "plan_sha256": context["plan_hash"], "source_manifest_sha256": context["source_hash"],
                  "pipeline_source_manifest_sha256": context["pipeline_hash"], "model_identity": backend.identity,
                  "protocol_hash": executor.protocol_hash, "exact_replay": True,
                  "replay_checks": replay_checks, "signature_sha256": digest(signature),
                  "physical_calls": len(backend.physical_calls), "completed_unix": time.time(), "labels_read": False}
        if len(backend.physical_calls) > 89 * len(units):
            raise ValueError("Engineering qualification exceeded its frozen call bound")
        verify_manifest(context["root"], context["plan"]["source_manifest"], expected_count=18)
        verify_manifest(context["root"], context["plan"]["pipeline_source_manifest"])
        atomic_json(segment / "QUALIFICATION.json", report, exclusive=True)
        return report
    except BaseException as exc:
        atomic_json(segment / "QUALIFICATION.json", {"status": "FAIL", "error_type": type(exc).__name__,
                    "plan_sha256": context["plan_hash"], "physical_calls": len(backend.physical_calls), "labels_read": False})
        raise


def collect(context, worker_id, device):
    require_frozen(context)
    require_unreleased_outputs(context)
    verify_qualification_pool(context)
    segment = context["output"] / "segments" / worker_id
    segment.mkdir(parents=True, exist_ok=False)
    ledger = segment / "physical_calls.jsonl"
    ledger.touch(exist_ok=False)
    state = {"schema": "g2_collection_worker_v1", "worker_id": worker_id, "pid": os.getpid(),
             "run_id": context["plan"]["run_id"], "phase": "LOADING_MODEL", "plan_sha256": context["plan_hash"], "completed_units": 0,
             "observations": 0, "labels_read": False, "first_gold_opened_unix": None}
    def update(**values):
        state.update(values, updated_unix=time.time())
        atomic_json(segment / "STATUS.json", state)
    update()
    try:
        backend, executor = load_backend(context, device, ledger)
        atomic_json(segment / "run_manifest.json", runtime_manifest(context, backend, worker_id, "formal"), exclusive=True)
        for unit in context["plan"]["units"]:
            check_stop(context)
            claim = claim_unit(context["output"], unit["question_id"], unit["seed"], worker_id, context["plan_hash"])
            if claim is None:
                continue
            update(phase="GENERATING", current_unit=unit_id(unit["question_id"], unit["seed"]))
            def progress(row):
                update(observations=state["observations"] + 1, current_mask=row["mask_sha256"])
                check_stop(context)
            rows = collect_unit(executor, context["questions"][unit["question_id"]], unit["seed"],
                                unit["mask_requests"], segment, context["plan_hash"], context["source_hash"],
                                partition=unit["partition"], progress=progress)
            frozen = segment / "units" / unit_id(unit["question_id"], unit["seed"]) / "UNIT_OUTPUTS_FROZEN.json"
            atomic_json(claim.with_suffix(".done.json"), {"schema": "g2_collection_done_v1", "worker_id": worker_id,
                        "question_id": unit["question_id"], "seed": unit["seed"], "unit_id": unit_id(unit["question_id"], unit["seed"]),
                        "plan_sha256": context["plan_hash"],
                        "claim_sha256": file_digest(claim), "unit_frozen_path": frozen.relative_to(context["output"]).as_posix(),
                        "unit_frozen_sha256": file_digest(frozen), "observations": len(rows),
                        "completed_unix": time.time(), "labels_read": False}, exclusive=True)
            update(completed_units=state["completed_units"] + 1)
        verify_manifest(context["root"], context["plan"]["source_manifest"], expected_count=18)
        verify_manifest(context["root"], context["plan"]["pipeline_source_manifest"])
        update(phase="GENERATION_COMPLETE_UNSCORED", generation_complete=True, scored=False,
               physical_calls=len(backend.physical_calls))
        return state
    except BaseException as exc:
        update(phase="STOPPED" if isinstance(exc, StopRequested) else "FAILED", error_type=type(exc).__name__, generation_complete=False,
               inference_retries=0, unrecorded_inflight_call_upper_bound=1)
        raise


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def execution_from_dict(module, value):
    """Restore native records for cost and fingerprint validation without a model."""
    nodes = {name: module.NodeRecord(**{**node, "generation": module.Generation(**node["generation"])})
             for name, node in value["nodes"].items()}
    keys = ("execution_id", "question_id", "question_fingerprint", "seed", "protocol_hash", "closure", "answer", "elapsed_seconds")
    return module.ExecutionResult(**{key: value[key] for key in keys}, mask=tuple(value["mask"]), nodes=nodes)


def validate_execution(context, unit, item, row, execution, prior, ledger, consumed):
    """Verify native cache lineage and every fresh call against its physical ledger."""
    module = context["module"]
    payload = module.Executor.question_payload(context["questions"][unit["question_id"]])
    question_hash = module.digest(payload)
    if (execution.question_id != unit["question_id"] or execution.seed != unit["seed"]
            or execution.question_fingerprint != question_hash
            or execution.protocol_hash != context["plan"]["protocol_hash"]
            or list(execution.mask) != item["mask"] or execution.execution_id != row["execution_id"]):
        raise ValueError("Execution identity differs from its frozen question/seed/mask")
    expected_names = [f"r{r}:w{w}" for r in range(1, context["config"].rounds + 1) for w in range(4)] + ["aggregate"]
    if set(execution.nodes) != set(expected_names):
        raise ValueError("Execution has missing or extra graph nodes")
    base = prior.get(tuple(item["reference_mask"])) if item["reference_mask"] is not None else None
    closure = (module.affected_closure(base.mask, execution.mask, context["config"].rounds)
               if base is not None else set(expected_names))
    if execution.closure != sorted(closure):
        raise ValueError("Execution closure differs from its declared anchor")
    prompt_executor = module.Executor(SimpleNamespace(identity=context["plan"]["model"]["identity"], encode=lambda text: []), context["config"])
    outputs = {}
    for name in expected_names:
        node = execution.nodes[name]
        limit = context["config"].aggregate_tokens if name == "aggregate" else context["config"].message_tokens
        messages = prompt_executor._messages(payload, name, execution.mask, outputs)[0]
        seed = module.derive_seed(unit["question_id"], unit["seed"], name)
        fingerprint = module.digest({"question": question_hash, "node": name, "messages": messages,
                                     "seed": seed, "protocol": execution.protocol_hash, "max_new_tokens": limit})
        if (node.node != name or node.seed != seed or node.messages != messages or node.input_fingerprint != fingerprint
                or module.digest(node.integrity_value()) != node.output_digest):
            raise ValueError("Native node prompt, RNG stream, or output integrity changed")
        if node.generation.output_tokens > limit or node.generation.input_tokens > context["config"].max_input_tokens:
            raise ValueError("Native token budget was exceeded")
        if name not in closure:
            source = base.nodes[name]
            if (node.reused is not True or node.source_execution_id != base.execution_id
                    or node.integrity_value() != source.integrity_value()):
                raise ValueError("Cached node does not belong to the named anchor")
        else:
            call_id = node.generation.call_id
            call = ledger.get(call_id)
            if (node.reused is not False or node.source_execution_id is not None or call_id in consumed
                    or call is None or call.get("status") != "SUCCESS"
                    or call.get("seed") != seed or call.get("prompt_sha256") != module.digest(messages)
                    or call.get("input_tokens") != node.generation.input_tokens
                    or call.get("output_tokens") != node.generation.output_tokens
                    or call.get("elapsed_seconds") != node.generation.elapsed_seconds
                    or call.get("gpu_seconds") != node.generation.gpu_seconds
                    or call.get("gpu_peak_allocated_bytes") != node.generation.gpu_peak_allocated_bytes
                    or call.get("finish_reason") != node.generation.finish_reason):
                raise ValueError("Fresh node has missing, duplicated, or inconsistent physical accounting")
            consumed.add(call_id)
        outputs[name] = node
    if (execution.answer != execution.nodes["aggregate"].parsed["answer"] or row["answer"] != execution.answer
            or row["service_output"] != execution.nodes["aggregate"].generation.text
            or row["service_message_metadata"] != execution.nodes["aggregate"].message_metadata
            or row["physical_cost"] != execution.cost() or row["logical_service_cost"] != execution.cost(False)):
        raise ValueError("Raw answer or physical/logical cost differs from its execution")


def artifact(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path), "bytes": path.stat().st_size}


def freeze_outputs(context):
    """Commit the complete unlabeled output set only after all physical workers exit."""
    require_frozen(context)
    require_unreleased_outputs(context)
    check_stop(context)
    verify_qualification_pool(context)
    output, plan = context["output"], context["plan"]
    expected_workers = set(plan["qualification"]["worker_ids"])
    segments = {path.name: path for path in (output / "segments").iterdir() if path.is_dir()}
    if set(segments) != expected_workers:
        raise ValueError("Formal segment owners differ from the qualified physical GPU pool")
    ledger, ledgers, index_rows, segment_states = {}, {}, {}, {}
    for worker_id, segment in sorted(segments.items()):
        state = read_json(segment / "STATUS.json")
        if (state.get("phase") != "GENERATION_COMPLETE_UNSCORED" or state.get("generation_complete") is not True
                or state.get("labels_read") is not False or state.get("plan_sha256") != context["plan_hash"]):
            raise ValueError("A collector is unfinished or failed")
        manifest = read_json(segment / "run_manifest.json")
        expected = {"worker_id": worker_id, "run_id": plan["run_id"], "mode": "formal", "plan_sha256": context["plan_hash"],
                    "source_manifest_sha256": context["source_hash"], "pipeline_source_manifest_sha256": context["pipeline_hash"],
                    "model_identity": plan["model"]["identity"], "protocol_hash": plan["protocol_hash"], "labels_read": False}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Formal worker runtime provenance differs from the frozen plan")
        worker_ledger = {}
        for call in read_jsonl(segment / "physical_calls.jsonl"):
            call_id = call["call_id"]
            if call_id in ledger or call.get("status") not in ("SUCCESS", "FAILED"):
                raise ValueError("Duplicate or invalid physical generation record")
            worker_ledger[call_id] = call
            ledger[call_id] = call
        if len(worker_ledger) != state.get("physical_calls"):
            raise ValueError("Worker status and actual physical ledger disagree")
        ledgers[worker_id] = worker_ledger
        segment_states[worker_id] = artifact(context["root"], segment / "STATUS.json")
        rows = {}
        index_path = segment / "raw_executions.jsonl"
        for row in read_jsonl(index_path) if index_path.exists() else []:
            key = (row["question_id"], row["seed"], row["mask_sha256"])
            if key in rows or key in index_rows:
                raise ValueError("A unique question/seed/mask was committed more than once")
            rows[key] = row
            index_rows[key] = worker_id
        segment_states[worker_id]["rows"] = rows
    if len(ledger) > plan["formal_call_upper_bound"]:
        raise ValueError("Physical generation ledger exceeds the approved formal budget")
    expected_claims = {unit_id(unit["question_id"], unit["seed"]) for unit in plan["units"]}
    if ({path.stem for path in (output / "claims").glob("*.json") if not path.name.endswith(".done.json")} != expected_claims
            or {path.name.removesuffix(".done.json") for path in (output / "claims").glob("*.done.json")} != expected_claims):
        raise ValueError("Frozen units and actual claims/completion receipts differ")
    consumed, merged, receipts = set(), [], []
    for unit in plan["units"]:
        uid = unit_id(unit["question_id"], unit["seed"])
        claim_path = output / "claims" / (uid + ".json")
        claim, done = read_json(claim_path), read_json(claim_path.with_suffix(".done.json"))
        worker_id = claim["worker_id"]
        if worker_id not in expected_workers:
            raise ValueError("A claim owner was never qualified")
        shared = {"question_id": unit["question_id"], "seed": unit["seed"], "worker_id": worker_id,
                  "plan_sha256": context["plan_hash"], "labels_read": False}
        if (any(claim.get(key) != value or done.get(key) != value for key, value in shared.items())
                or done.get("claim_sha256") != file_digest(claim_path) or done.get("unit_id") != uid):
            raise ValueError("Completion receipt is not bound to its unique owner claim")
        segment = segments[worker_id]
        frozen_path = safe_path(output, done["unit_frozen_path"])
        if (frozen_path != segment / "units" / uid / "UNIT_OUTPUTS_FROZEN.json"
                or file_digest(frozen_path) != done["unit_frozen_sha256"]):
            raise ValueError("Unit freeze receipt path or bytes changed")
        frozen = read_json(frozen_path)
        schedule = unique_mask_requests(unit["mask_requests"])
        expected_unit_plan = {"schema": "g2_collection_unit_v1", "question_id": unit["question_id"], "seed": unit["seed"],
                              "plan_sha256": context["plan_hash"], "protocol_hash": plan["protocol_hash"],
                              "mask_schedule": schedule, "labels_read": False}
        if read_json(frozen_path.parent / "UNIT_PLAN.json") != expected_unit_plan:
            raise ValueError("Actual per-unit execution schedule changed")
        if (frozen.get("question_id") != unit["question_id"] or frozen.get("seed") != unit["seed"]
                or frozen.get("plan_sha256") != context["plan_hash"] or frozen.get("protocol_hash") != plan["protocol_hash"]
                or frozen.get("labels_read") is not False or frozen.get("first_gold_opened_unix") is not None
                or frozen.get("observations") != len(schedule) or done["observations"] != len(schedule)
                or len(frozen["rows"]) != len(schedule)):
            raise ValueError("Unit output freeze is incomplete or used gold feedback")
        prior = {}
        for index, (item, reference) in enumerate(zip(schedule, frozen["rows"])):
            mask_hash = digest(item["mask"])
            key = (unit["question_id"], unit["seed"], mask_hash)
            row = segment_states[worker_id]["rows"].pop(key)
            if (set(reference) != {"mask_sha256", "raw_path", "raw_sha256", "execution_path", "execution_sha256"}
                    or any(row.get(name) != value for name, value in reference.items())):
                raise ValueError("Unit freeze references differ from the committed raw index")
            expected_raw = {"schema": "g2_collected_graph_v1", "question_id": unit["question_id"], "seed": unit["seed"], "unit_id": uid,
                            "graph_mask": item["mask"], "mask_sha256": mask_hash, "mask_id": "".join(map(str, item["mask"])),
                            "mask_index": index, "mask_aliases": item["aliases"], "request_indices": item["request_indices"],
                            "reference_mask": item["reference_mask"], "partition": unit["partition"],
                            "plan_sha256": context["plan_hash"], "protocol_hash": plan["protocol_hash"],
                            "source_manifest_sha256": context["source_hash"], "labels_read": False, "quality": None}
            if any(row.get(name) != value for name, value in expected_raw.items()):
                raise ValueError("Committed graph does not match its prespecified unique mask request")
            raw_path, execution_path = safe_path(segment, row["raw_path"]), safe_path(segment, row["execution_path"])
            if (raw_path != frozen_path.parent / "raw" / (row["mask_id"] + ".json")
                    or execution_path != frozen_path.parent / "executions" / (row["mask_id"] + ".json")
                    or file_digest(raw_path) != row["raw_sha256"] or file_digest(execution_path) != row["execution_sha256"]):
                raise ValueError("Committed raw or execution bytes changed")
            raw = read_json(raw_path)
            if raw != {name: value for name, value in row.items() if name not in ("raw_path", "raw_sha256")}:
                raise ValueError("Append-only index and atomic raw object disagree")
            execution = execution_from_dict(context["module"], read_json(execution_path))
            validate_execution(context, unit, item, row, execution, prior, ledgers[worker_id], consumed)
            if row["question_fingerprint"] != execution.question_fingerprint:
                raise ValueError("Raw question fingerprint differs from its execution")
            prior[tuple(item["mask"])] = execution
            merged.append({**row, "raw_path": raw_path.relative_to(context["root"]).as_posix(),
                           "execution_path": execution_path.relative_to(context["root"]).as_posix(), "worker_id": worker_id})
        receipts.append(artifact(context["root"], frozen_path))
    if any(state["rows"] for state in segment_states.values()) or len(merged) != context["expected_graphs"]:
        raise ValueError("Unrequested graphs or missing unique mask outputs remain")
    # Keep successful orphan calls and failures explicit. They never become extra
    # samples, and their tokens/time remain in the total physical budget.
    orphan_ids = sorted(set(ledger) - consumed)
    qualification_calls = []
    for worker in sorted(expected_workers):
        folder = output / "qualification" / worker
        calls = list(read_jsonl(folder / "physical_calls.jsonl"))
        report = read_json(folder / "QUALIFICATION.json")
        if len(calls) != report["physical_calls"] or any(call.get("status") != "SUCCESS" for call in calls):
            raise ValueError("Qualification report and retained physical engineering cost disagree")
        qualification_calls.extend(calls)
    audit = {"schema": "graph_credit_g2_physical_cost_v1", "plan_sha256": context["plan_hash"], "labels_read": False,
             "formal_calls": len(ledger), "referenced_calls": len(consumed), "orphan_call_ids": orphan_ids,
             "failed_call_ids": sorted(key for key, call in ledger.items() if call["status"] == "FAILED"),
             "known_input_tokens": sum(call.get("input_tokens") or 0 for call in ledger.values()),
             "known_output_tokens": sum(call.get("output_tokens") or 0 for call in ledger.values()),
             "unknown_output_token_calls": sum(call.get("output_tokens") is None for call in ledger.values()),
             "model_elapsed_seconds": sum(call.get("elapsed_seconds") or 0 for call in ledger.values()),
             "qualification_calls": len(qualification_calls),
             "qualification_input_tokens": sum(call["input_tokens"] for call in qualification_calls),
             "qualification_output_tokens": sum(call["output_tokens"] for call in qualification_calls),
             "qualification_model_elapsed_seconds": sum(call["elapsed_seconds"] for call in qualification_calls),
             "inference_retries": 0, "unrecorded_inflight_call_upper_bound": 0,
             "physical_ledgers": [artifact(context["root"], segments[worker] / "physical_calls.jsonl") for worker in sorted(segments)],
             "qualification_ledgers": [artifact(context["root"], output / "qualification" / worker / "physical_calls.jsonl")
                                       for worker in sorted(expected_workers)],
             "qualification_receipts": [artifact(context["root"], output / "qualification" / worker / "QUALIFICATION.json")
                                        for worker in sorted(expected_workers)],
             "worker_runtime_manifests": [artifact(context["root"], segments[worker] / "run_manifest.json") for worker in sorted(segments)],
             "worker_terminal_states": [artifact(context["root"], segments[worker] / "STATUS.json") for worker in sorted(segments)],
             "unit_receipts": receipts}
    canonical_path = output / "observations.jsonl"
    temporary = canonical_path.with_name(canonical_path.name + "." + uuid.uuid4().hex + ".collectortmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in merged:
                handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, canonical_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    audit_path = output / "PHYSICAL_COST_AUDIT.json"
    atomic_json(audit_path, audit)
    artifacts = {"observations": artifact(context["root"], canonical_path), "physical_cost_audit": artifact(context["root"], audit_path)}
    report = {"schema": "graph_credit_g2_outputs_frozen_v1", "state": "GLOBAL_OUTPUTS_FROZEN", "run_id": plan["run_id"],
              "plan_sha256": context["plan_hash"], "source_manifest_sha256": context["source_hash"],
              "pipeline_source_manifest_sha256": context["pipeline_hash"], "protocol_hash": plan["protocol_hash"],
              "features_sha256": plan["features"]["sha256"], "features_receipt_sha256": plan["features"]["receipt_sha256"],
              "expected_units": len(plan["units"]), "completed_units": len(receipts),
              "expected_outputs": context["expected_graphs"], "observations": len(merged), "artifacts": artifacts,
              "artifact_hashes": {**{key: value["sha256"] for key, value in artifacts.items()},
                                  "features": plan["features"]["sha256"], "features_receipt": plan["features"]["receipt_sha256"]},
              "frozen_unix": time.time(), "first_gold_opened_unix": None, "labels_read": False}
    atomic_json(output / "GLOBAL_OUTPUTS_FROZEN.json", report, exclusive=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("qualify", "collect", "freeze-outputs"), default="collect")
    parser.add_argument("--worker-id")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    context = load_plan(root, args.plan)
    if args.mode == "freeze-outputs":
        result = freeze_outputs(context)
    else:
        if (args.worker_id not in context["plan"]["qualification"]["worker_ids"]
                or not re.fullmatch(r"cuda:[0-9]+", args.device)):
            parser.error("qualify/collect require a safe worker ID and explicit CUDA index")
        result = qualify(context, args.worker_id, args.device) if args.mode == "qualify" else collect(context, args.worker_id, args.device)
    print(canonical({key: result.get(key) for key in ("status", "phase", "observations", "labels_read")}), flush=True)


if __name__ == "__main__":
    main()
