"""Read-only, CPU-only integrity acceptance of the completed local G1 mirror.

This verifies existing results; it never reads gold, scores answers, imports the
inference backend, or changes an experiment artifact. Integrity PASS and the
scientific signal gate are deliberately separate fields.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time

RUN_ID = "motivation_v3_20261103"
PRIVATE = "artifacts/private/performance/"
SCHEDULER_FILES = {
    PRIVATE + name for name in (
        "elastic_worker.py", "elastic_finalize.py", "elastic_supervisor.py",
        "elastic_migrate.py", "cross_gpu_audit.py", "elastic_resume_qualification.py")}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class MirrorWaitingError(FileNotFoundError):
    def __init__(self, message, relative_path=None):
        self.relative_path = relative_path
        super().__init__(message)


class MirrorCorruptionError(RuntimeError):
    def __init__(self, message, relative_path=None):
        self.relative_path = relative_path
        super().__init__(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _need(condition, message, relative_path=None):
    if not condition:
        raise MirrorCorruptionError(message, relative_path)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "Duplicate JSON member: " + key)
        result[key] = value
    return result


def _decode(data, relative_path=None):
    try:
        return json.loads(data, object_pairs_hook=_pairs)
    except (ValueError, UnicodeError, MirrorCorruptionError) as exc:
        raise MirrorCorruptionError("Invalid JSON: " + str(exc), relative_path) from exc


def _relative(root, path):
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return str(path)


def safe_file(base, relative, *, root=None):
    """Reject traversal, Windows drive escapes, symlinks and accidental gold reads."""
    origin = Path(root or base).resolve()
    base = Path(base).resolve()
    _need(isinstance(relative, str) and bool(relative), "Empty artifact path", str(relative))
    rel = PurePosixPath(relative)
    display = _relative(origin, base.joinpath(*rel.parts))
    _need("\\" not in relative and ":" not in relative and not rel.is_absolute()
          and all(part not in ("", ".", "..") for part in relative.split("/")),
          "Unsafe relative artifact path: " + relative, display)
    candidate = base.joinpath(*rel.parts)
    _need(not any(token in candidate.name.lower() for token in ("labels", "gold")),
          "Gold/label file is outside mirror verifier scope", display)
    for part in (candidate, *candidate.parents):
        if part == base:
            break
        _need(not part.is_symlink(), "Symlink is not an immutable mirror artifact", display)
    if not candidate.exists():
        raise MirrorWaitingError("Mirror artifact not present yet: " + display, display)
    resolved = candidate.resolve()
    _need(resolved.is_relative_to(base) and resolved.is_file(), "Artifact escapes base or is not a file", display)
    return resolved


def _json_file(root, relative):
    path = safe_file(root, relative, root=root)
    value = _decode(path.read_bytes(), relative)
    _need(isinstance(value, dict), "Expected JSON object", relative)
    return value


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_hash(root, relative, expected):
    _need(isinstance(expected, str) and SHA_RE.fullmatch(expected), "Invalid expected SHA256", relative)
    path = safe_file(root, relative, root=root)
    actual = file_sha(path)
    _need(actual == expected, "Mirror SHA256 mismatch: " + relative, relative)
    return actual


def _emit(callback, stage, **values):
    if callback is not None:
        callback({"stage": stage, "observed_unix": time.time(), **values})


def _jsonl(root, relative):
    path = safe_file(root, relative, root=root)
    with path.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            _need(line.endswith(b"\n") and bool(line.strip()), "Incomplete/blank JSONL row", relative)
            row = _decode(line, relative)
            _need(isinstance(row, dict), "JSONL row is not an object", relative)
            yield number, row


def required_artifacts():
    required = {"all_observations.jsonl", "ELASTIC_PAID_COSTS.json", "ELASTIC_ALL_OUTPUTS_FROZEN.json"}
    for index in range(3):
        required.update(f"shard_{index:03d}/{name}" for name in (
            "raw_executions.jsonl", "physical_calls.jsonl", "observations.jsonl",
            "OUTPUTS_FROZEN.json", "SCORING.json", "STATUS.json",
            "original_raw_executions.jsonl", "original_physical_calls.jsonl"))
    return required


def verify_committed(root, run_relative, receipt, callback=None):
    hashes = receipt.get("committed_artifact_hashes")
    _need(isinstance(hashes, dict) and required_artifacts() <= set(hashes),
          "COMPLETE receipt lacks committed artifacts", run_relative + "/ELASTIC_FINALIZATION.json")
    for number, (relative, expected) in enumerate(hashes.items(), 1):
        path = safe_file(root / run_relative, relative, root=root)
        check_hash(root, _relative(root, path), expected)
        _emit(callback, "VERIFY_COMMITTED_ARTIFACTS", completed=number, total=len(hashes))


def original_input_relative(relative):
    """Pre-merge canonical hashes bind original_* backups, never merged files."""
    parts = PurePosixPath(relative).parts
    if len(parts) == 2 and re.fullmatch(r"shard_00[0-2]", parts[0]) and parts[1] in {
            "raw_executions.jsonl", "physical_calls.jsonl"}:
        return parts[0] + "/original_" + parts[1]
    return relative


def verify_input_hashes(root, run_relative, receipt, callback=None):
    values = receipt.get("input_hashes")
    _need(isinstance(values, dict) and bool(values), "Missing original input hashes", run_relative + "/ELASTIC_FINALIZATION.json")
    for number, (relative, expected) in enumerate(values.items(), 1):
        mapped = original_input_relative(relative)
        check_hash(root, run_relative + "/" + mapped, expected)
        _emit(callback, "VERIFY_ORIGINAL_AND_SEGMENT_INPUTS", completed=number, total=len(values))


def _normalize_backup(data, relative):
    lines, complete = data.splitlines(keepends=True), []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line, object_pairs_hook=_pairs)
        except (ValueError, UnicodeError):
            _need(index == len(lines) - 1, "Invalid middle migration backup record", relative)
            return len(complete), b"".join(complete), line
        _need(isinstance(value, dict), "Non-object migration backup row", relative)
        complete.append(line.rstrip(b"\r\n") + b"\n")
    return len(complete), b"".join(complete), b""


def verify_migration(root, run_relative, receipt, plan_hash, source_hash):
    binding = receipt.get("migration_receipt", {})
    relative = binding.get("path", "artifacts/audit/elastic_migration_20261103.json")
    receipt_hash = check_hash(root, relative, binding.get("sha256"))
    migration = _json_file(root, relative)
    _need(migration.get("schema") == "graph_credit_elastic_migration_v1"
          and migration.get("state") == "COMPLETE" and migration.get("labels_read") is False
          and migration.get("run_id") == receipt["run_id"] and migration.get("logical_shards") == 3
          and migration.get("frozen_plan_sha256") == plan_hash
          and migration.get("source_manifest_sha256") == source_hash,
          "Migration receipt identity mismatch", relative)
    files = migration.get("files", [])
    expected = {f"{run_relative}/shard_{i:03d}/{name}" for i in range(3)
                for name in ("raw_executions.jsonl", "physical_calls.jsonl")}
    _need(len(files) == 6 and {item.get("path") for item in files} == expected,
          "Migration receipt must bind six distinct original files", relative)
    for item in files:
        index = item.get("logical_shard")
        name = PurePosixPath(item["path"]).name
        _need(type(index) is int and index in range(3)
              and item["path"] == f"{run_relative}/shard_{index:03d}/{name}", "Migration shard mismatch", relative)
        backup_rel = item.get("backup_path")
        expected_backup = f"{run_relative}/shard_{index:03d}/migration_20261103/{name}.original_bytes"
        _need(backup_rel == expected_backup, "Migration full-bytes backup path mismatch", relative)
        check_hash(root, backup_rel, item.get("backup_sha256"))
        rows, normalized, tail = _normalize_backup(safe_file(root, backup_rel).read_bytes(), backup_rel)
        original_rel = f"{run_relative}/shard_{index:03d}/original_{name}"
        check_hash(root, original_rel, digest(normalized))
        _need(digest(normalized) == item.get("normalized_sha256") and rows == item.get("valid_rows")
              and len(tail) == item.get("partial_tail_bytes")
              and (digest(tail) if tail else None) == item.get("partial_tail_sha256"),
              "Migration normalization/tail binding mismatch", backup_rel)
    return relative, receipt_hash


def verify_static_guard(root, plan, callback=None):
    sources = plan.get("source_manifest")
    _need(isinstance(sources, dict) and sources, "Missing frozen source manifest")
    _need(digest(canonical(sources).encode()) == plan.get("source_manifest_sha256"), "Frozen source manifest SHA mismatch")
    actual_names = set()
    for folder, suffix in (("src", ".py"), ("scripts", ".py"), ("configs", ".json")):
        actual_names.update(path.relative_to(root).as_posix() for path in (root / folder).rglob("*")
                            if path.is_file() and path.suffix == suffix and "__pycache__" not in path.parts)
    _need(actual_names == set(sources), "Frozen source/config file inventory changed")
    for relative, item in sources.items():
        check_hash(root, relative, item.get("sha256"))
        _need(safe_file(root, relative).stat().st_size == item.get("bytes"), "Frozen source size mismatch", relative)
    models = plan.get("model_manifest")
    identity = plan.get("model_identity", {})
    _need(isinstance(models, dict) and models and identity.get("model_files") == models,
          "Model/tokenizer manifest differs from frozen backend identity")
    _need(SHA_RE.fullmatch(identity.get("tokenizer_chat_template_sha256", "")), "Missing frozen tokenizer template SHA")
    model_relative = "models/Qwen2.5-3B-Instruct"
    model_root = root / model_relative
    actual_models = {path.relative_to(model_root).as_posix() for path in model_root.rglob("*")
                     if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}
                     and ".cache" not in path.relative_to(model_root).parts}
    _need(actual_models == set(models), "Model/tokenizer file inventory changed", model_relative)
    for number, (relative, item) in enumerate(models.items(), 1):
        full_relative = model_relative + "/" + relative
        check_hash(root, full_relative, item.get("sha256"))
        _need(safe_file(root, full_relative).stat().st_size == item.get("bytes"), "Model/tokenizer size mismatch", full_relative)
        _emit(callback, "VERIFY_MODEL_TOKENIZER_BYTES", completed=number, total=len(models))
    scorer = plan.get("scorer_identity", {})
    scorer_relative = "src/" + scorer.get("callable", "").split(":")[0].replace(".", "/") + ".py"
    _need(scorer_relative in sources and scorer.get("module_sha256") == sources[scorer_relative]["sha256"]
          and scorer.get("module_bytes") == sources[scorer_relative]["bytes"], "Frozen scorer module identity differs")
    _need(identity.get("backend_code_sha256") == sources.get("src/graph_credit/backend.py", {}).get("sha256"),
          "Frozen backend code identity differs")
    tokenizer_config = _json_file(root, model_relative + "/tokenizer_config.json")
    template = tokenizer_config.get("chat_template")
    _need(isinstance(template, str) and digest(template.encode()) == identity["tokenizer_chat_template_sha256"],
          "Frozen tokenizer template identity differs", model_relative + "/tokenizer_config.json")


def verify_allocation(root, receipt, plan_hash, migration_hash, callback=None):
    lineage = receipt.get("scheduler_lineage", {})
    raw = lineage.get("path", "")
    # Server absolute paths are machine-specific; the immutable filename has a
    # fixed local root-relative location and its actual SHA must still match.
    relative = PRIVATE + "allocation.json"
    _need(isinstance(raw, str) and raw.replace("\\", "/").endswith("/" + relative), "Unexpected scheduler allocation path")
    allocation_hash = check_hash(root, relative, lineage.get("sha256"))
    allocation = _json_file(root, relative)
    _need(allocation.get("schema") == "graph_credit_elastic_allocation_v1"
          and allocation.get("run_id") == receipt["run_id"] and allocation.get("logical_shards") == 3
          and allocation.get("frozen_plan_sha256") == plan_hash
          and allocation.get("migration_receipt_sha256") == migration_hash
          and allocation.get("labels_read") is False and allocation.get("claim_mode") == "shared_atomic_units",
          "Allocation identity mismatch", relative)
    scheduler = allocation.get("scheduler_sources")
    _need(isinstance(scheduler, dict) and set(scheduler) == SCHEDULER_FILES,
          "Allocation must bind exactly six private scheduler code files", relative)
    for code, expected in scheduler.items():
        check_hash(root, code, expected)
    _need(receipt.get("finalizer_source_sha256") == scheduler[PRIVATE + "elastic_finalize.py"],
          "Finalizer source is not allocation-bound", relative)
    workers = allocation.get("workers", [])
    seen = set()
    tasks = None
    for worker in workers:
        name = worker.get("worker_id")
        _need(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name) and name not in seen,
              "Duplicate/invalid allocated worker", relative)
        seen.add(name)
        path = worker["task_manifest"]
        check_hash(root, path, worker.get("task_manifest_sha256"))
        manifest = _json_file(root, path)
        _need(manifest.get("schema") == "graph_credit_elastic_tasks_v1"
              and manifest.get("worker_id") == name and manifest.get("run_id") == receipt["run_id"]
              and manifest.get("frozen_plan_sha256") == plan_hash and manifest.get("logical_shards") == 3
              and manifest.get("migration_receipt_sha256") == migration_hash
              and manifest.get("scheduler_sources") == scheduler and manifest.get("labels_read") is False
              and manifest.get("claim_mode") == "shared_atomic_units",
              "Task manifest differs from immutable allocation", path)
        if tasks is None:
            tasks = manifest.get("tasks")
        _need(isinstance(tasks, list) and manifest.get("tasks") == tasks
              and manifest.get("expected_new_rows") == sum(t["expected_new_rows"] for t in tasks),
              "Shared task list/count differs", path)
    _need(bool(seen), "Allocation has no physical workers", relative)
    inputs = receipt.get("scheduler_input_hashes")
    _need(isinstance(inputs, dict) and set(scheduler) <= set(inputs)
          and relative in inputs, "Missing scheduler input lineage hashes", relative)
    for number, (path, expected) in enumerate(inputs.items(), 1):
        check_hash(root, path, expected)
        _emit(callback, "VERIFY_SCHEDULER_LINEAGE", completed=number, total=len(inputs))
    return allocation_hash


def _design_meta(plan):
    qids, seeds, pairs, contexts = (plan[key] for key in ("question_ids", "seeds", "edge_pairs", "base_masks"))
    _need(len(set(qids)) == len(qids) and len(set(seeds)) == len(seeds), "Duplicate frozen questions/seeds")
    qindex = {qid: index for index, qid in enumerate(qids)}
    context_map = {item["name"] + "_" + digest(canonical(item["mask"]).encode())[:12]: item["name"] for item in contexts}
    return qindex, set(seeds), set(map(tuple, pairs)), context_map


def verify_observations(root, run_relative, plan, plan_hash, *, expected_counts=(20100, 20100, 19800), callback=None):
    qindex, seeds, pairs, contexts = _design_meta(plan)
    relative = run_relative + "/all_observations.jsonl"
    seen, counts, batch = set(), Counter(), []
    executions = set()
    def flush(executor):
        if not batch:
            return
        list(executor.map(lambda value: check_hash(root, *value), batch))
        batch.clear()
        _emit(callback, "VERIFY_EXECUTION_FILES", observations=len(seen), execution_files=len(executions), total=sum(expected_counts))
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="mirror-sha") as executor:
        for _, row in _jsonl(root, relative):
            try:
                key = tuple(row[field] for field in ("question_id", "context_id", "edge_e", "edge_f", "seed", "cell"))
            except KeyError as exc:
                raise MirrorCorruptionError("Missing observation coordinate: " + str(exc), relative) from exc
            qid, context, e, f, seed, cell = key
            _need(qid in qindex and seed in seeds and (e, f) in pairs and context in contexts
                  and cell in {"00", "01", "10", "11"} and all(type(x) is int for x in (e, f, seed)),
                  "Observation outside frozen Cartesian design", relative)
            _need(key not in seen, "Duplicate observation key", relative)
            seen.add(key)
            index = qindex[qid] % 3
            counts[index] += 1
            _need(row.get("protocol_hash") == plan["protocol_hash"] and row.get("frozen_plan_sha256") == plan_hash
                  and row.get("context_family") == contexts[context], "Observation plan/protocol/context mismatch", relative)
            _need(type(row.get("quality")) in (int, float) and 0 <= row["quality"] <= 1,
                  "Observation is not scored with bounded quality", relative)
            shard = root / run_relative / f"shard_{index:03d}"
            execution = safe_file(shard, row.get("execution_path"), root=root)
            execution_relative = _relative(root, execution)
            # Every row is checked, including repeated references with different
            # expected hashes; only progress's file count deduplicates paths.
            batch.append((execution_relative, row.get("execution_sha256")))
            executions.add(execution_relative)
            if len(batch) >= 128:
                flush(executor)
        flush(executor)
    actual = tuple(counts[index] for index in range(3))
    _need(actual == tuple(expected_counts), "Incomplete/wrong logical-shard observation support", relative)
    total_design = len(qindex) * len(seeds) * len(pairs) * len(contexts) * 4
    _need(len(seen) == total_design == sum(expected_counts), "Frozen Cartesian coverage is incomplete", relative)
    return actual, len(executions)


def verify_scoring(root, run_relative, receipt, plan, plan_hash, counts):
    freeze_rel = run_relative + "/ELASTIC_ALL_OUTPUTS_FROZEN.json"
    global_freeze = _json_file(root, freeze_rel)
    _need(global_freeze.get("outputs") == sum(counts) and global_freeze.get("original_plan_sha256") == plan_hash
          and global_freeze.get("labels_opened") is False, "Global output-freeze marker mismatch", freeze_rel)
    shard_freezes = []
    label_hashes = set()
    for index, count in enumerate(counts):
        prefix = f"{run_relative}/shard_{index:03d}/"
        frozen = _json_file(root, prefix + "OUTPUTS_FROZEN.json")
        raw_sha = receipt["committed_artifact_hashes"][f"shard_{index:03d}/raw_executions.jsonl"]
        physical_sha = receipt["committed_artifact_hashes"][f"shard_{index:03d}/physical_calls.jsonl"]
        _need(frozen.get("outputs") == count and frozen.get("raw_executions_sha256") == raw_sha
              and frozen.get("physical_calls_sha256") == physical_sha
              and frozen.get("frozen_plan_sha256") == plan_hash and frozen.get("labels_opened") is False,
              "Shard freeze count/raw SHA differs", prefix + "OUTPUTS_FROZEN.json")
        scoring = _json_file(root, prefix + "SCORING.json")
        _need(scoring.get("diagnostic_scoring_count") == count and scoring.get("raw_executions_sha256") == raw_sha
              and scoring.get("source_manifest_sha256") == plan["source_manifest_sha256"]
              and scoring.get("scorer_identity") == plan["scorer_identity"]
              and scoring.get("global_outputs_frozen_before_label_access") is True
              and scoring.get("labels_used_to_change_service_outputs") is False,
              "Scoring is not bound to frozen raw outputs", prefix + "SCORING.json")
        _need(frozen.get("original_raw_sha256") == receipt["input_hashes"][f"shard_{index:03d}/raw_executions.jsonl"]
              and frozen.get("original_physical_calls_sha256") == receipt["input_hashes"][f"shard_{index:03d}/physical_calls.jsonl"],
              "Legacy raw/call freeze binding mismatch", prefix + "OUTPUTS_FROZEN.json")
        status = _json_file(root, prefix + "STATUS.json")
        _need(status.get("generation_complete") is True and status.get("scored") is True
              and status.get("outputs_frozen") == count and status.get("scientific_completion") is False,
              "Shard terminal status inconsistent", prefix + "STATUS.json")
        for name in ("raw_executions.jsonl", "observations.jsonl"):
            _need(sum(1 for _ in _jsonl(root, prefix + name)) == count,
                  "Canonical shard record count mismatch", prefix + name)
        label_hashes.add(scoring.get("labels_sha256"))
        shard_freezes.append(frozen)
    _need(global_freeze.get("logical_shards") == shard_freezes, "Global freeze does not bind shard freezes", freeze_rel)
    _need(label_hashes == {receipt.get("labels_sha256")} and SHA_RE.fullmatch(receipt.get("labels_sha256", "")),
          "Scoring label-hash lineage differs (gold is never read)", run_relative + "/ELASTIC_FINALIZATION.json")


def verify_analysis(root, run_relative, plan_hash, protocol, observations_hash):
    relative = run_relative + "/G1_ANALYSIS.json"
    analysis = _json_file(root, relative)
    support, provenance = analysis.get("support", {}), analysis.get("provenance", {})
    gate = analysis.get("scientific_gate", {}).get("status")
    if gate not in {"PASS", "INSUFFICIENT_SIGNAL"}:
        raise MirrorWaitingError("Independent G1 analysis is not a complete terminal gate", relative)
    _need(support.get("input_rows") == 60000 and support.get("exact_duplicate_rows_ignored") == 0
          and support.get("unscored_rows") == 0 and support.get("complete_question_count") == 200
          and support.get("incomplete_group_count") == 0, "Analysis support mismatch", relative)
    coverage = provenance.get("complete_frozen_design_coverage", {})
    _need(provenance.get("plan_hash_verified") is True and provenance.get("supplied_plan_sha256") == plan_hash
          and provenance.get("frozen_plan_hashes") == [plan_hash] and provenance.get("protocol_hashes") == [protocol]
          and provenance.get("observations_sha256") == observations_hash
          and coverage.get("verified") is True and coverage.get("missing_group_count") == 0
          and coverage.get("expected_scored_rows") == 60000 and not analysis.get("validation_errors"),
          "Analysis frozen-plan/coverage/observation provenance mismatch", relative)
    return gate


def verify_terminal(root, run_id=RUN_ID, progress_callback=None):
    root = Path(root).resolve()
    _need(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", run_id), "Invalid run_id")
    run_relative = "results/pilot/" + run_id
    final_relative = run_relative + "/ELASTIC_FINALIZATION.json"
    initial_receipt_hash = file_sha(safe_file(root, final_relative))
    receipt = _json_file(root, final_relative)
    if receipt.get("phase") != "COMPLETE":
        raise MirrorWaitingError("Finalization has not reached COMPLETE", final_relative)
    _need(receipt.get("schema") == "elastic_g1_finalization_v1" and receipt.get("run_id") == run_id
          and receipt.get("observations") == 60000 and receipt.get("scientific_completion") is False,
          "Invalid COMPLETE finalization receipt", final_relative)
    verify_committed(root, run_relative, receipt, progress_callback)
    plan_bytes = [safe_file(root, f"{run_relative}/shard_{index:03d}/frozen_plan.json").read_bytes() for index in range(3)]
    _need(all(value == plan_bytes[0] for value in plan_bytes), "Logical-shard frozen plans differ", run_relative + "/shard_000/frozen_plan.json")
    plan, plan_hash = _decode(plan_bytes[0]), digest(plan_bytes[0])
    _need(receipt.get("original_plan_sha256") == plan_hash and receipt.get("protocol_hash") == plan.get("protocol_hash")
          and receipt.get("original_source_manifest_sha256") == plan.get("source_manifest_sha256"),
          "Finalization frozen-plan lineage mismatch", final_relative)
    _need(len(plan.get("question_ids", [])) == 200 and len(plan.get("seeds", [])) == 5
          and len(plan.get("edge_pairs", [])) == 3 and len(plan.get("base_masks", [])) == 5,
          "Unexpected full G1 frozen-design dimensions", run_relative + "/shard_000/frozen_plan.json")
    verify_static_guard(root, plan, progress_callback)
    verify_input_hashes(root, run_relative, receipt, progress_callback)
    migration_relative, migration_hash = verify_migration(root, run_relative, receipt, plan_hash, plan["source_manifest_sha256"])
    allocation_hash = verify_allocation(root, receipt, plan_hash, migration_hash, progress_callback)
    counts, execution_count = verify_observations(root, run_relative, plan, plan_hash, callback=progress_callback)
    verify_scoring(root, run_relative, receipt, plan, plan_hash, counts)
    observations_hash = check_hash(root, run_relative + "/all_observations.jsonl", receipt["committed_artifact_hashes"]["all_observations.jsonl"])
    gate = verify_analysis(root, run_relative, plan_hash, plan["protocol_hash"], observations_hash)
    status = _json_file(root, "RUN_STATUS.json")
    hashes = {"finalization": file_sha(safe_file(root, final_relative)),
              "analysis": file_sha(safe_file(root, run_relative + "/G1_ANALYSIS.json")),
              "allocation": allocation_hash, "migration_receipt": migration_hash,
              "frozen_plan": plan_hash, "all_observations": observations_hash}
    _need(hashes["finalization"] == initial_receipt_hash, "Finalization receipt changed during local verification", final_relative)
    _emit(progress_callback, "TERMINAL_LOCAL_INTEGRITY_PASS", observations=sum(counts), scientific_gate=gate)
    return {"schema": "graph_credit_mirror_terminal_verification_v1", "status": "PASS", "run_id": run_id,
            "verification_scope": "local_mirror_integrity", "remote_stage": status.get("stage"),
            "observations": sum(counts), "shard_counts": list(counts), "execution_files_verified": execution_count,
            "execution_references_verified": sum(counts),
            "scientific_gate": gate, "scientific_completion": False, "labels_read": False,
            "artifact_sha256": hashes, "receipt_sha256": hashes["finalization"],
            "observations_sha256": observations_hash, "analysis_sha256": hashes["analysis"],
            "completed_unix": time.time()}
