"""Generate remaining immutable G1 question/seed units outside frozen source.

No labels are read. Original three logical shards, graph order, node RNG and
protocol hashes remain unchanged. Physical process scheduling is separate,
precommitted provenance. This worker never retries inference or overwrites a
previous segment. Its sole cache continuation is the last committed execution.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

RUN_ID = "motivation_v3_20261103"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def sha(content):
    return hashlib.sha256(content).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def confined(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes the experiment root")
    return path


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (canonical(value) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".workertmp")
    with tmp.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return sha(content)


def append(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL record is not an object")
                rows.append(value)
    return rows


def load_frozen(root):
    sys.path.insert(0, str(root / "src"))
    import graph_credit.executor as executor
    from graph_credit.backend import HFBackend
    if Path(executor.__file__).resolve() != root / "src/graph_credit/executor.py":
        raise ValueError("Previously imported graph_credit does not belong to this frozen root")
    name = "elastic_frozen_pilot_" + sha(str(root).encode())[:12]
    spec = importlib.util.spec_from_file_location(name, root / "scripts/run_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    return executor, HFBackend, pilot


def design_steps(plan):
    return [(context, pair[0], pair[1], cell) for context in plan["base_masks"]
            for pair in plan["edge_pairs"] for cell in ("11", "01", "00", "10")]


def step_mask(step):
    context, e, f, cell = step
    mask = list(context["mask"])
    mask[e], mask[f] = int(cell[0]), int(cell[1])
    return mask


def row_key(row):
    return (row["question_id"], row["seed"], row["context_family"], row["edge_e"], row["edge_f"], row["cell"])


def task_key(task):
    return task["logical_shard"], task["question_id"], task["seed"]


def validate_task(task, plan, completed):
    index = task.get("logical_shard")
    if type(index) is not int or index not in range(3):
        raise ValueError("Invalid logical shard")
    qid, seed, start = task.get("question_id"), task.get("seed"), task.get("start_index")
    if qid not in plan["question_ids"] or plan["question_ids"].index(qid) % 3 != index or seed not in plan["seeds"]:
        raise ValueError("Task violates original question/shard/seed allocation")
    if type(start) is not int or start not in range(60) or task.get("expected_new_rows") != 60 - start:
        raise ValueError("Task has an invalid committed prefix or expected row count")
    if completed.get((index, qid, seed), 0) != start:
        raise ValueError("Task does not continue the exact committed prefix")
    if start and (not task.get("resume_execution_path") or not task.get("resume_execution_sha256")):
        raise ValueError("Partial unit requires the committed base execution")


def deserialize_execution(data, module):
    """Reconstruct only declared frozen dataclass fields and validate every node."""
    nodes = {}
    def node_order(name):
        if name == "aggregate":
            return (1000, 0)
        match = re.fullmatch(r"r(\d+):w(\d+)", name)
        if not match:
            raise ValueError("Invalid expanded-node name in resume execution")
        return (int(match[1]), int(match[2]))
    # canonical JSON sorts aggregate before r1. Restore original topological
    # insertion order, including floating-point cost summation order.
    for name in sorted(data["nodes"], key=node_order):
        value = data["nodes"][name]
        native = dict(value)
        native["generation"] = module.Generation(**native["generation"])
        record = module.NodeRecord(**native)
        if record.node != name or module.digest(record.integrity_value()) != record.output_digest:
            raise ValueError("Resume node content digest mismatch")
        nodes[name] = record
    values = {field.name: data[field.name] for field in dataclasses.fields(module.ExecutionResult)}
    values.update(nodes=nodes, mask=module.validate_mask(data["mask"]))
    result = module.ExecutionResult(**values)
    if data.get("physical_cost") != result.cost() or data.get("logical_service_cost") != result.cost(False):
        raise ValueError("Resume execution cost does not match its immutable node ledger")
    return result


def validate_base(base, executor, question, mask, seed, module):
    """A data hash alone is insufficient: rederive closed-input fingerprints."""
    payload = executor.question_payload(question)
    if (base.question_id != payload["id"] or base.question_fingerprint != module.digest(payload)
            or base.mask != tuple(mask) or base.seed != seed or base.protocol_hash != executor.protocol_hash):
        raise ValueError("Resume execution identity does not match this unit and protocol")
    order = [f"r{r}:w{w}" for r in range(1, executor.config.rounds + 1) for w in range(4)] + ["aggregate"]
    if set(base.nodes) != set(order):
        raise ValueError("Resume execution lacks the full expanded node set")
    outputs = {}
    for name in order:
        node = base.nodes[name]
        messages, received, self_tokens = executor._messages(payload, name, base.mask, outputs)
        node_seed = module.derive_seed(payload["id"], seed, name)
        cap = executor.config.aggregate_tokens if name == "aggregate" else executor.config.message_tokens
        fingerprint = module.digest({"question": base.question_fingerprint, "node": name, "messages": messages,
                                     "seed": node_seed, "protocol": executor.protocol_hash, "max_new_tokens": cap})
        if (node.input_fingerprint != fingerprint or node.messages != messages or node.seed != node_seed
                or node.received_message_tokens != received or node.self_state_tokens != self_tokens):
            raise ValueError("Resume node closed-input fingerprint mismatch")
        wire, parsed, metadata = module.normalize_message(node.generation.text, executor.backend, cap)
        if wire != node.wire_message or parsed != node.parsed or metadata != node.message_metadata:
            raise ValueError("Resume node wire message differs from frozen normalization")
        outputs[name] = node
    if base.answer != outputs["aggregate"].parsed["answer"]:
        raise ValueError("Resume final answer differs from the saved aggregate")


def bootstrap(root, task_path, worker_id, module, pilot):
    """Verify immutable allocation/receipts/source/data before loading any model."""
    manifest_bytes = task_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "graph_credit_elastic_tasks_v1" or manifest.get("worker_id") != worker_id or manifest.get("run_id") != RUN_ID or manifest.get("logical_shards") != 3 or manifest.get("labels_read") is not False:
        raise ValueError("Unexpected elastic task manifest")
    base = root / "results/pilot" / RUN_ID
    plans = [(base / f"shard_{i:03d}/frozen_plan.json").read_bytes() for i in range(3)]
    if len(set(plans)) != 1 or sha(plans[0]) != manifest["frozen_plan_sha256"]:
        raise ValueError("Original logical shards do not share the bound frozen plan")
    plan = json.loads(plans[0])
    if plan.get("num_shards") != 3 or plan.get("intended_shards") != 3 or len(design_steps(plan)) != 60:
        raise ValueError("Original three-shard 60-cell unit design changed")
    sources = pilot.source_manifest(root)
    if sources != plan["source_manifest"] or module.digest(sources) != plan["source_manifest_sha256"] or pilot.scorer_identity(plan.get("scorer")) != plan["scorer_identity"]:
        raise ValueError("Frozen source/config/scoring identity changed")
    receipt_path = confined(root, manifest["migration_receipt"])
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if sha(receipt_bytes) != manifest["migration_receipt_sha256"] or receipt.get("state") != "COMPLETE" or receipt.get("run_id") != RUN_ID or receipt.get("labels_read") is not False or receipt.get("frozen_plan_sha256") != manifest["frozen_plan_sha256"]:
        raise ValueError("Migration receipt is not a completed label-free handoff")
    scheduler = manifest["scheduler_sources"]
    for relative, expected in scheduler.items():
        if sha(confined(root, relative).read_bytes()) != expected:
            raise ValueError("Frozen scheduler source changed: " + relative)
    self_relative = Path(__file__).resolve().relative_to(root).as_posix()
    if scheduler.get(self_relative) != sha(Path(__file__).read_bytes()):
        raise ValueError("This worker is not bound by scheduler provenance")
    allocation = read(root / "artifacts/private/performance/allocation.json")
    if allocation.get("run_id") != RUN_ID or allocation.get("scheduler_sources") != scheduler or allocation.get("frozen_plan_sha256") != manifest["frozen_plan_sha256"] or allocation.get("migration_receipt_sha256") != manifest["migration_receipt_sha256"]:
        raise ValueError("Allocation and worker manifests disagree")
    normalized = {(item["logical_shard"], Path(item["path"]).name): item for item in receipt["files"]}
    completed, last_rows = {}, {}
    steps = design_steps(plan)
    for index in range(3):
        shard = base / f"shard_{index:03d}"
        if (shard / "SCORING.json").exists() or (shard / "OUTPUTS_FROZEN.json").exists():
            raise ValueError("Legacy shard was already scored/frozen before elastic generation")
        raw_path = shard / "raw_executions.jsonl"
        if sha(raw_path.read_bytes()) != normalized[(index, "raw_executions.jsonl")]["normalized_sha256"]:
            raise ValueError("Legacy committed observations changed after migration")
        rows = jsonl(raw_path)
        assigned = [qid for i, qid in enumerate(plan["question_ids"]) if i % 3 == index]
        expected = [(qid, seed, context["name"], e, f, cell) for qid in assigned for seed in plan["seeds"] for context, e, f, cell in steps]
        if [row_key(row) for row in rows] != expected[:len(rows)] or len(rows) > len(expected):
            raise ValueError("Legacy records are not the exact immutable design prefix")
        for row in rows:
            key = (index, row["question_id"], row["seed"])
            completed[key] = completed.get(key, 0) + 1
            last_rows[key] = row
    shared = manifest.get("claim_mode") == "shared_atomic_units"
    if manifest.get("claim_mode") not in (None, "static", "shared_atomic_units"):
        raise ValueError("Unknown task claim mode")
    global_tasks, seen = [], set()
    own_found = False
    for worker in allocation["workers"]:
        path = confined(root, worker["task_manifest"])
        content = path.read_bytes()
        if sha(content) != worker["task_manifest_sha256"]:
            raise ValueError("Allocated task file hash mismatch")
        other = json.loads(content)
        if other.get("worker_id") != worker["worker_id"] or other.get("scheduler_sources") != scheduler or other.get("frozen_plan_sha256") != manifest["frozen_plan_sha256"] or other.get("migration_receipt_sha256") != manifest["migration_receipt_sha256"] or other.get("labels_read") is not False:
            raise ValueError("Global task ownership provenance mismatch")
        if worker["worker_id"] == worker_id:
            own_found = path == task_path and content == manifest_bytes
        if shared and (other.get("claim_mode") != "shared_atomic_units" or other.get("tasks") != manifest["tasks"]):
            raise ValueError("Shared-claim workers must bind the identical global task list")
        if not shared and other.get("claim_mode") == "shared_atomic_units":
            raise ValueError("Static and shared task ownership cannot be mixed")
        local_seen = set()
        for task in other["tasks"]:
            validate_task(task, plan, completed)
            key = task_key(task)
            if key in local_seen:
                raise ValueError("Duplicate unit inside one worker's task manifest")
            local_seen.add(key)
            if key in seen and not shared:
                raise ValueError("Question/seed unit is owned by multiple physical workers")
            if key not in seen:
                seen.add(key)
                global_tasks.append(task)
        if sum(t["expected_new_rows"] for t in other["tasks"]) != other.get("expected_new_rows"):
            raise ValueError("Worker task coverage count mismatch")
    expected_units = {(i % 3, qid, seed) for i, qid in enumerate(plan["question_ids"]) for seed in plan["seeds"] if completed.get((i % 3, qid, seed), 0) < 60}
    if not own_found or seen != expected_units or sum(t["expected_new_rows"] for t in global_tasks) != allocation.get("remaining_rows"):
        raise ValueError("Allocation does not cover exactly all remaining immutable units")
    data_path = root / "data/derived/motivation_questions.jsonl"
    if sha(data_path.read_bytes()) != plan["data_sha256"]:
        raise ValueError("Frozen question file changed")
    questions = jsonl(data_path)
    if [q["id"] for q in questions] != plan["question_ids"] or [module.digest(module.Executor.question_payload(q)) for q in questions] != plan["question_payload_sha256"]:
        raise ValueError("Question identity or payload differs from the frozen plan")
    for task in manifest["tasks"]:
        if task["start_index"]:
            last = last_rows[task_key(task)]
            if task.get("resume_execution_path") != last["execution_path"] or task.get("resume_execution_sha256") != last["execution_sha256"]:
                raise ValueError("Resume pointer is not the last committed execution")
    return manifest, plan, {q["id"]: q for q in questions}, sha(manifest_bytes)


def claim_unit(root, task, worker_id, manifest_sha):
    """One immutable owner; an existing or crashed claim is never stolen."""
    folder = root / "results/pilot" / RUN_ID / "elastic_claims"
    folder.mkdir(parents=True, exist_ok=True)
    key = sha(canonical(list(task_key(task))).encode())
    path = folder / (key + ".json")
    claim = {"schema": "graph_credit_elastic_unit_claim_v1", "worker_id": worker_id,
             "task_manifest_sha256": manifest_sha, "task": task, "pid": os.getpid(),
             "claimed_unix": time.time(), "labels_read": False}
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(canonical(claim) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return None
    return path


def native_config(plan, module):
    values = dict(plan["config"].get("graph", plan["config"].get("executor", {})))
    if "role_prompts" in values:
        values["role_prompts"] = tuple(values["role_prompts"])
    return module.GraphConfig(**values)


def execute_task(root, task, plan, question, executor, module, segment, task_manifest_sha, progress=None):
    """Append only missing cells; every retained base remains bound to its old cost."""
    steps = design_steps(plan)
    start, seed = task["start_index"], task["seed"]
    base = None
    if start:
        shard = root / "results/pilot" / RUN_ID / f"shard_{task['logical_shard']:03d}"
        path = confined(shard, task["resume_execution_path"])
        data = path.read_bytes()
        if sha(data) != task["resume_execution_sha256"]:
            raise ValueError("Resume execution bytes changed")
        base = deserialize_execution(json.loads(data), module)
        validate_base(base, executor, question, step_mask(steps[start - 1]), seed, module)
    rows = []
    for index in range(start, len(steps)):
        context, e, f, cell = steps[index]
        mask = step_mask(steps[index])
        result = executor.execute(question, mask, seed, base=base)
        base = result
        filename = f"{sha(question['id'].encode())[:20]}_{seed}_{context['name']}_{e}_{f}_{cell}.json"
        path = segment / "executions" / filename
        if path.exists():
            raise FileExistsError("Refuse to overwrite an existing elastic execution")
        execution_sha = atomic(path, result.to_dict())
        row = {"question_id": question["id"], "task": question.get("task", "gsm8k"), "split": question.get("split", "dev"),
               "context_id": context["name"] + "_" + module.digest(context["mask"])[:12], "context_family": context["name"],
               "edge_e": e, "edge_f": f, "seed": seed, "cell": cell, "mask_e": int(cell[0]), "mask_f": int(cell[1]),
               "graph_mask": mask, "protocol_hash": executor.protocol_hash, "frozen_plan_sha256": sha((root / "results/pilot" / RUN_ID / f"shard_{task['logical_shard']:03d}/frozen_plan.json").read_bytes()),
               "quality": None, "answer": result.answer, "service_output": result.nodes["aggregate"].generation.text,
               "service_message_metadata": result.nodes["aggregate"].message_metadata, "execution_id": result.execution_id,
               "execution_path": "executions/" + filename, "execution_sha256": execution_sha,
               "physical_cost": result.cost(), "logical_service_cost": result.cost(False),
               "elastic_task_manifest_sha256": task_manifest_sha, "elastic_unit_start_index": start,
               "elastic_cell_index": index, "logical_shard": task["logical_shard"]}
        append(segment / "raw_executions.jsonl", row)
        rows.append(row)
        if progress:
            progress(row)
    if len(rows) != task["expected_new_rows"]:
        raise ValueError("Task output coverage differs from the frozen assignment")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    task_path = args.task_manifest.resolve(strict=True)
    if not task_path.is_relative_to(root) or not re.fullmatch(r"[A-Za-z0-9_-]+", args.worker_id) or not args.device.isdigit():
        raise ValueError("Unsafe worker identity, task path or CUDA index")
    module, backend_class, pilot = load_frozen(root)
    manifest, plan, questions, manifest_sha = bootstrap(root, task_path, args.worker_id, module, pilot)
    private = root / "artifacts/private/performance"
    worker_status = private / f"worker_{args.worker_id}.json"
    claim = private / f"worker_{args.worker_id}.claim"
    with claim.open("x", encoding="utf-8") as handle:
        handle.write(canonical({"pid": os.getpid(), "task_manifest_sha256": manifest_sha}) + "\n")
    state = {"schema": "graph_credit_elastic_worker_v1", "worker_id": args.worker_id, "pid": os.getpid(),
             "task_manifest_sha256": manifest_sha, "labels_read": False, "scientific_completion": False,
             "generated_rows": 0, "completed_tasks": 0, "expected_new_rows": manifest["expected_new_rows"]}
    segments, segment_counts = {}, {}
    backend = None
    def update(phase, **values):
        state.update(phase=phase, updated_at_unix=time.time(), **values)
        atomic(worker_status, state)
    try:
        update("LOADING_MODEL")
        backend = backend_class(root / "models/Qwen2.5-3B-Instruct", "cuda:" + args.device)
        if backend.identity != plan["model_identity"] or backend.identity.get("model_files") != plan["model_manifest"]:
            raise ValueError("Physical worker model/tokenizer/software identity differs from original execution")
        executor = module.Executor(backend, native_config(plan, module))
        if executor.protocol_hash != plan["protocol_hash"]:
            raise ValueError("Physical worker protocol hash differs from the frozen v3 protocol")
        runtime = {"schema": "graph_credit_elastic_runtime_v1", "worker_id": args.worker_id,
                   "task_manifest_sha256": manifest_sha, "frozen_plan_sha256": manifest["frozen_plan_sha256"],
                   "protocol_hash": executor.protocol_hash, "model_identity": backend.identity,
                   "runtime": backend.runtime, "source_manifest": plan["source_manifest"],
                   "source_manifest_sha256": plan["source_manifest_sha256"], "scorer_identity": plan["scorer_identity"],
                   "executor_config": dataclasses.asdict(executor.config), "scheduler_sources": manifest["scheduler_sources"],
                   "migration_receipt_sha256": manifest["migration_receipt_sha256"], "labels_read": False,
                   "scientific_completion": False}
        for task in manifest["tasks"]:
            claim_path = None
            if manifest.get("claim_mode") == "shared_atomic_units":
                claim_path = claim_unit(root, task, args.worker_id, manifest_sha)
                if claim_path is None:
                    state["skipped_claimed_units"] = state.get("skipped_claimed_units", 0) + 1
                    continue
            logical = task["logical_shard"]
            if logical not in segments:
                segment = root / "results/pilot" / RUN_ID / f"shard_{logical:03d}/elastic_segments" / args.worker_id
                segment.mkdir(parents=True, exist_ok=False)
                (segment / "executions").mkdir()
                atomic(segment / "task_manifest.json", manifest)
                atomic(segment / "run_manifest.json", runtime)
                segments[logical] = segment
                segment_counts[logical] = 0
            segment = segments[logical]
            backend.physical_log = segment / "physical_calls.jsonl"
            def progress(row):
                state["generated_rows"] += 1
                segment_counts[logical] += 1
                update("GENERATING", logical_shard=logical, question_id=task["question_id"], seed=task["seed"], cell_index=row["elastic_cell_index"])
                atomic(segment / "STATUS.json", {**state, "segment_rows": segment_counts[logical], "scored": False})
            new_rows = execute_task(root, task, plan, questions[task["question_id"]], executor, module, segment, manifest_sha, progress)
            if claim_path is not None:
                atomic(claim_path.with_suffix(".done.json"), {
                    "schema": "graph_credit_elastic_unit_completion_v1", "worker_id": args.worker_id,
                    "task_manifest_sha256": manifest_sha, "claim_sha256": sha(claim_path.read_bytes()),
                    "task": task, "new_rows": len(new_rows), "last_execution_sha256": new_rows[-1]["execution_sha256"],
                    "segment": segment.relative_to(root).as_posix(), "completed_unix": time.time(), "labels_read": False})
            state["completed_tasks"] += 1
            update("GENERATING")
        if pilot.source_manifest(root) != plan["source_manifest"] or pilot.scorer_identity(plan.get("scorer")) != plan["scorer_identity"]:
            raise ValueError("Frozen source/config/scorer changed during elastic generation")
        if sha(task_path.read_bytes()) != manifest_sha:
            raise ValueError("Task ownership manifest changed during generation")
        update("GENERATION_COMPLETE_UNSCORED", generation_complete=True, scored=False)
        for logical, segment in segments.items():
            atomic(segment / "STATUS.json", {**state, "logical_shard": logical, "segment_rows": segment_counts[logical], "scored": False})
        print(canonical({"worker_id": args.worker_id, "phase": state["phase"], "generated_rows": state["generated_rows"], "labels_read": False}), flush=True)
        return 0
    except Exception as exc:
        update("FAILED", error_type=type(exc).__name__, error=str(exc), generation_complete=False, scored=False)
        for segment in segments.values():
            atomic(segment / "STATUS.json", state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
