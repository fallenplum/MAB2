"""Production CPU-only G2 prediction freeze and separately released evaluation.

All paths are project-relative, sources and immutable input bytes are checked
before label access, and first-access events are exclusive durable artifacts.
The original frozen numeric scorer is the only scoring implementation used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid

import numpy as np

SEEDS = (2701, 2702, 2703, 2704, 2705)
COUNTS = {"train": 200, "tune": 80, "test": 120}
SHA = re.compile(r"[0-9a-f]{64}\Z")


class WorkflowError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise WorkflowError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON member: " + key)
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=_object)


def safe_path(root, relative, *, existing=True):
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
            "Evidence path must be relative POSIX text")
    parts = relative.split("/")
    require(all(part not in ("", ".", "..") for part in parts), "Unsafe evidence path components")
    root = Path(root)
    candidate = root.joinpath(*parts)
    require(candidate.is_relative_to(root), "Evidence path escapes project root")
    current = root
    for part in parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if existing:
                raise
            break
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
                "Evidence symlinks/reparse points are forbidden")
    if existing:
        require(candidate.is_file(), "Evidence is not a regular file: " + relative)
    return candidate


def relative_path(root, path):
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    require(path.is_relative_to(root), "Artifact is outside project root")
    return path.relative_to(root).as_posix()


def _no_label_read(relative):
    name = Path(relative).name.lower()
    require(not (name.endswith(".jsonl") and any(word in name for word in ("label", "gold"))),
            "Label JSONL may only be opened by the gated first-access reader")


def bytes_read(root, relative):
    _no_label_read(relative)
    return safe_path(root, relative).read_bytes()


def read_json(root, relative):
    value = decode(bytes_read(root, relative))
    require(isinstance(value, dict), "Expected JSON object: " + relative)
    return value


def file_sha(root, relative):
    _no_label_read(relative)
    value = hashlib.sha256()
    with safe_path(root, relative).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_sha(root, relative, expected):
    require(isinstance(expected, str) and SHA.fullmatch(expected), "Invalid expected SHA for " + relative)
    actual = file_sha(root, relative)
    require(actual == expected, "Frozen artifact SHA changed: " + relative)
    return actual


def _sync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_json(root, relative, value, *, exclusive=False):
    path = safe_path(root, relative, existing=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_path(root, relative, existing=False)
    content = (canonical(value) + "\n").encode("utf-8")
    if exclusive:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".workflowtmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name == "nt":
                os.rename(temporary, path)  # Windows rename atomically refuses an existing destination.
            else:
                os.link(temporary, path)  # POSIX link publishes complete bytes without replacing an event.
        finally:
            if temporary.exists():
                temporary.unlink()
    else:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".workflowtmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            safe_path(root, relative, existing=False)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    _sync_directory(path.parent)
    return hashlib.sha256(content).hexdigest()


def verify_manifest(root, manifest):
    require(isinstance(manifest, dict) and bool(manifest), "Source manifest is empty")
    for relative, item in manifest.items():
        require(isinstance(item, dict) and "sha256" in item, "Invalid source manifest entry")
        check_sha(root, relative, item["sha256"])
        if "bytes" in item:
            require(safe_path(root, relative).stat().st_size == item["bytes"], "Source size changed: " + relative)


def load_context(root, plan_path):
    root = Path(root).resolve(strict=True)
    plan_relative = relative_path(root, plan_path)
    plan_bytes = bytes_read(root, plan_relative)
    plan = decode(plan_bytes)
    require(plan.get("schema") == "graph_credit_g2_plan_v1" and plan.get("status") == "PLAN_FROZEN"
            and plan.get("labels_read") is False,
            "Expected a frozen label-free G2 collection plan")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", plan.get("run_id", "")), "Invalid G2 run ID")
    verify_manifest(root, plan["source_manifest"])
    verify_manifest(root, plan["pipeline_source_manifest"])
    source_hash, pipeline_hash = digest(plan["source_manifest"]), digest(plan["pipeline_source_manifest"])
    require(source_hash == plan.get("source_manifest_sha256"), "Original source manifest identity differs")
    own_relative = relative_path(root, Path(__file__).resolve())
    require(own_relative in plan["pipeline_source_manifest"], "Executing workflow is outside frozen pipeline")
    require(Counter(plan["partitions"].values()) == Counter(COUNTS), "G2 requires train200/tune80/test120")
    expected = {}
    for unit in plan["units"]:
        qid, seed = unit["question_id"], unit["seed"]
        require(qid in plan["partitions"] and type(seed) is int and seed in SEEDS
                and unit.get("partition") == plan["partitions"][qid], "Unit partition/seed mismatch")
        masks = {tuple(request["mask"]) for request in unit["mask_requests"]}
        for mask in masks:
            require(len(mask) == 12 and all(type(bit) is int and bit in (0, 1) for bit in mask), "Invalid planned graph")
            key = (qid, seed, mask)
            require(key not in expected, "Duplicate planned question/mask/seed")
            expected[key] = unit["partition"]
    require(len(plan["units"]) == plan.get("expected_units") == 2000,
            "G2 requires exactly 2000 unique question/seed units")
    require(len(expected) == plan.get("expected_outputs") and len(expected) <= 52000, "Unique graph budget mismatch")
    require({(qid, seed) for qid, seed, mask in expected} == {(qid, seed) for qid in plan["partitions"] for seed in SEEDS},
            "Incomplete frozen question/seed Cartesian coverage")
    output = plan["output_dir"].rstrip("/")
    require(output == "results/g2/" + plan["run_id"]
            and plan_relative == output + "/frozen_plan.json", "G2 plan/output must use their canonical frozen paths")
    safe_path(root, output + "/PLAN_FROZEN.json")
    return {"root": root, "plan": plan, "plan_relative": plan_relative,
            "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "source_sha256": source_hash,
            "pipeline_sha256": pipeline_hash, "expected": expected, "output": output}


def stage_path(context, name):
    return context["output"] + "/" + name


def verify_context_unchanged(context):
    check_sha(context["root"], context["plan_relative"], context["plan_sha256"])
    verify_manifest(context["root"], context["plan"]["source_manifest"])
    verify_manifest(context["root"], context["plan"]["pipeline_source_manifest"])


def load_predictor(context):
    names = [name for name in context["plan"]["pipeline_source_manifest"] if Path(name).name == "predictor.py"]
    require(len(names) == 1, "Frozen pipeline must bind one predictor")
    spec = importlib.util.spec_from_file_location("g2_frozen_predictor", safe_path(context["root"], names[0]))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scorer(context):
    relative = "src/graph_credit/scoring.py"
    require(relative in context["plan"]["source_manifest"], "Original scorer is not source-bound")
    spec = importlib.util.spec_from_file_location("g2_original_frozen_scorer", safe_path(context["root"], relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "score_item", None)), "Original scoring callable is absent")
    return module.score_item


def load_features(context):
    root, descriptor = context["root"], context["plan"]["features"]
    feature_hash = check_sha(root, descriptor["path"], descriptor["sha256"])
    receipt_hash = check_sha(root, descriptor["receipt"], descriptor["receipt_sha256"])
    receipt = read_json(root, descriptor["receipt"])
    require(receipt.get("status") == "PASS_LABEL_FREE" and receipt.get("labels_read") is False
            and receipt.get("feature_sha256") == feature_hash and receipt.get("pca_fit_questions") == 200
            and receipt.get("transformed_questions") == 400 and receipt.get("dimensions") == 9,
            "Feature receipt does not establish train-only frozen PCA")
    with np.load(safe_path(root, descriptor["path"]), allow_pickle=False) as values:
        ids = values["question_ids"].tolist()
        phi = np.asarray(values["phi"], dtype=np.float64)
    require(len(ids) == len(set(ids)) == 400 and set(ids) == set(context["plan"]["partitions"])
            and phi.shape == (400, 9) and np.isfinite(phi).all() and np.all(phi[:, -1] == 1),
            "Feature question coverage/dimensions differ")
    train_ids = [qid for qid in ids if context["plan"]["partitions"][qid] == "train"]
    require(receipt.get("train_ids_sha256") == digest(train_ids), "PCA training IDs differ from frozen partition")
    return dict(zip(ids, phi)), feature_hash, receipt_hash


def _validate_graph(context, row):
    root, plan = context["root"], context["plan"]
    mask = tuple(row["graph_mask"])
    key = (row["question_id"], row["seed"], mask)
    require(key in context["expected"] and row.get("partition") == context["expected"][key], "Unplanned graph/partition")
    require(row.get("quality") is None and row.get("labels_read") is False
            and row.get("plan_sha256") == context["plan_sha256"]
            and row.get("source_manifest_sha256") == context["source_sha256"]
            and row.get("protocol_hash") == plan["protocol_hash"], "Raw graph feedback/source/protocol differs")
    require(row.get("mask_id") == "".join(map(str, mask)) and row.get("mask_sha256") == digest(list(mask)),
            "Raw graph mask identity differs")
    raw_relative, execution_relative = row["raw_path"], row["execution_path"]
    check_sha(root, raw_relative, row["raw_sha256"])
    check_sha(root, execution_relative, row["execution_sha256"])
    original = read_json(root, raw_relative)
    require("/units/" in raw_relative, "Canonical raw path has no frozen unit")
    segment = raw_relative.split("/units/", 1)[0]
    require(execution_relative == segment + "/" + original["execution_path"], "Mapped execution path differs from original raw")
    for name, value in original.items():
        if name != "execution_path":
            require(row.get(name) == value, "Global raw mapping changed original field: " + name)
    execution = read_json(root, execution_relative)
    require(execution.get("question_id") == key[0] and execution.get("seed") == key[1]
            and execution.get("mask") == list(mask) and execution.get("protocol_hash") == plan["protocol_hash"]
            and execution.get("execution_id") == row.get("execution_id")
            and execution.get("question_fingerprint") == plan["questions"]["payload_sha256"][key[0]],
            "Execution identity differs from raw observation")
    require(execution["nodes"]["aggregate"]["generation"]["text"] == row["service_output"], "Service output changed after generation")
    return key


def verify_global(context):
    root, plan = context["root"], context["plan"]
    frozen_plan = read_json(root, stage_path(context, "PLAN_FROZEN.json"))
    require(frozen_plan.get("state") == "PLAN_FROZEN" and frozen_plan.get("plan_sha256") == context["plan_sha256"]
            and frozen_plan.get("source_manifest_sha256") == context["source_sha256"]
            and frozen_plan.get("pipeline_source_manifest_sha256") == context["pipeline_sha256"]
            and frozen_plan.get("protocol_hash") == plan["protocol_hash"], "Plan was not frozen under the current source/protocol")
    marker_relative = stage_path(context, "GLOBAL_OUTPUTS_FROZEN.json")
    marker_hash = file_sha(root, marker_relative)
    marker = read_json(root, marker_relative)
    require(marker.get("schema") == "graph_credit_g2_outputs_frozen_v1" and marker.get("state") == "GLOBAL_OUTPUTS_FROZEN"
            and marker.get("run_id") == plan["run_id"] and marker.get("plan_sha256") == context["plan_sha256"]
            and marker.get("source_manifest_sha256") == context["source_sha256"]
            and marker.get("pipeline_source_manifest_sha256") == context["pipeline_sha256"]
            and marker.get("protocol_hash") == plan["protocol_hash"]
            and marker.get("expected_units") == marker.get("completed_units") == 2000
            and marker.get("expected_outputs") == marker.get("observations") == len(context["expected"])
            and marker.get("labels_read") is False and marker.get("first_gold_opened_unix") is None,
            "Global generation has not reached the complete label-free frozen state")
    phi, feature_hash, feature_receipt_hash = load_features(context)
    require(marker.get("features_sha256") == feature_hash and marker.get("features_receipt_sha256") == feature_receipt_hash,
            "Global freeze does not bind actual feature/PCA receipt bytes")
    descriptors = marker.get("artifacts", {})
    artifact_hashes = marker.get("artifact_hashes", {})
    require(set(artifact_hashes) == {"observations", "physical_cost_audit", "features", "features_receipt"}
            and artifact_hashes.get("features") == feature_hash
            and artifact_hashes.get("features_receipt") == feature_receipt_hash,
            "Global artifact hashes omit/change features/PCA receipt")
    for name in ("observations", "physical_cost_audit"):
        item = descriptors[name]
        require(artifact_hashes[name] == item["sha256"], "Global artifact descriptor/hash alias mismatch")
        check_sha(root, item["path"], item["sha256"])
        require(safe_path(root, item["path"]).stat().st_size == item["bytes"], "Global artifact size mismatch")
    rows, seen = [], set()
    observation = descriptors["observations"]
    with safe_path(root, observation["path"]).open("rb") as handle, ThreadPoolExecutor(max_workers=6) as pool:
        batch = []
        def flush():
            for key in pool.map(lambda row: _validate_graph(context, row), batch):
                require(key not in seen, "Duplicate unique question/mask/seed graph output")
                seen.add(key)
            rows.extend(batch)
            batch.clear()
        for line in handle:
            require(line.endswith(b"\n") and bool(line.strip()), "Global JSONL has an incomplete/blank record")
            row = decode(line)
            require(isinstance(row, dict), "Global graph row is not an object")
            batch.append(row)
            if len(batch) == 128:
                flush()
        if batch:
            flush()
    require(seen == set(context["expected"]), "Global generation does not cover every unique frozen graph")
    verify_context_unchanged(context)
    check_sha(root, marker_relative, marker_hash)
    check_sha(root, observation["path"], observation["sha256"])
    cost_audit = read_json(root, descriptors["physical_cost_audit"]["path"])
    for name in ("physical_ledgers", "qualification_ledgers", "unit_receipts"):
        for item in cost_audit.get(name, []):
            check_sha(root, item["path"], item["sha256"])
            require(safe_path(root, item["path"]).stat().st_size == item["bytes"], "Generation cost lineage size changed")
    return {"marker": marker, "marker_sha256": marker_hash, "rows": rows, "phi": phi,
            "feature_sha256": feature_hash, "feature_receipt_sha256": feature_receipt_hash,
            "physical_cost_audit": cost_audit}


@contextmanager
def workflow_lease(context):
    relative = stage_path(context, "workflow.lock")
    path = safe_path(context["root"], relative, existing=False)
    with path.open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _release(context, filename, *, state, binding_key, binding_sha, label_fields):
    relative = stage_path(context, filename)
    path = safe_path(context["root"], relative, existing=False)
    if not path.exists():
        return None
    value = read_json(context["root"], relative)
    require(value.get("state") == state and value.get("plan_sha256") == context["plan_sha256"]
            and value.get(binding_key) == binding_sha
            and type(value.get("released_unix")) in (int, float) and math.isfinite(value["released_unix"]),
            "Label release is not bound to the current completed freeze")
    require(all(isinstance(value.get(field), str) and SHA.fullmatch(value[field]) for field in label_fields),
            "Label release omits a valid expected label SHA")
    return value, file_sha(context["root"], relative)


def _labels_after_event(context, partitions, expected_hashes, event_name, binding):
    root, plan = context["root"], context["plan"]
    event_relative = stage_path(context, event_name)
    prior = None
    event_path = safe_path(root, event_relative, existing=False)
    if event_path.exists():
        prior = read_json(root, event_relative)
        require(prior.get("schema") == "graph_credit_g2_first_label_access_v1"
                and prior.get("event") == "HANDLES_OPENED_BEFORE_FIRST_READ"
                and prior.get("plan_sha256") == context["plan_sha256"]
                and prior.get("run_id") == plan["run_id"]
                and prior.get("source_manifest_sha256") == context["source_sha256"]
                and prior.get("pipeline_source_manifest_sha256") == context["pipeline_sha256"]
                and prior.get("expected_labels_sha256") == expected_hashes
                and all(prior.get(key) == value for key, value in binding.items())
                and set(prior.get("handles", {})) == set(partitions)
                and all(prior["handles"][part]["path"] == plan["label_release"][part] for part in partitions),
                "Existing first-label event belongs to different immutable inputs")
    with ExitStack() as stack:
        handles, identities = {}, {}
        for partition in partitions:
            relative = plan["label_release"][partition]
            path = safe_path(root, relative)
            handles[partition] = stack.enter_context(path.open("rb"))
            info = os.fstat(handles[partition].fileno())
            identities[partition] = {"path": relative, "bytes_at_open": info.st_size,
                                     "device": info.st_dev, "inode": info.st_ino,
                                     "handle_opened_unix": time.time()}
        event = {"schema": "graph_credit_g2_first_label_access_v1", "event": "HANDLES_OPENED_BEFORE_FIRST_READ",
                 "plan_sha256": context["plan_sha256"], "run_id": plan["run_id"],
                 "created_unix": time.time(), "first_gold_opened_unix": min(item["handle_opened_unix"] for item in identities.values()),
                 "handles": identities, "expected_labels_sha256": expected_hashes,
                 "source_manifest_sha256": context["source_sha256"], "pipeline_source_manifest_sha256": context["pipeline_sha256"],
                 **binding}
        if prior is None:
            event_hash = atomic_json(root, event_relative, event, exclusive=True)
        else:
            event_hash = file_sha(root, event_relative)
        attempt_relative = stage_path(context, "label_access_attempts/" + uuid.uuid4().hex[:20] + ".json")
        atomic_json(root, attempt_relative, {**event, "first_event_path": event_relative,
                                          "first_event_sha256": event_hash, "is_first_access": prior is None}, exclusive=True)
        labels = {}
        for partition, handle in handles.items():
            content = handle.read()  # The first read occurs after the exclusive fsynced event.
            require(hashlib.sha256(content).hexdigest() == expected_hashes[partition], "Released label bytes SHA mismatch")
            require(content.endswith(b"\n"), "Released labels have an incomplete trailing record")
            values = {}
            for line in content.splitlines():
                require(bool(line.strip()), "Blank released label record")
                row = decode(line)
                require(isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] not in values,
                        "Duplicate/malformed released label ID")
                values[row["id"]] = row
            expected_ids = {qid for qid, part in plan["partitions"].items() if part == partition}
            require(set(values) == expected_ids and len(values) == COUNTS[partition], "Released labels differ from exact frozen partition")
            labels[partition] = values
    return labels, event_hash


def _scored(context, global_data, labels, scorer):
    result = {partition: [] for partition in labels}
    for row in global_data["rows"]:
        partition = row["partition"]
        if partition not in labels:
            continue
        value = float(scorer(row["service_output"], labels[partition][row["question_id"]], task="gsm8k"))
        require(math.isfinite(value) and value in (0, 1), "Original GSM scorer returned an invalid quality")
        result[partition].append({"question_id": row["question_id"], "seed": row["seed"],
                                  "graph_mask": row["graph_mask"], "mask_id": row["mask_id"],
                                  "partition": partition, "quality": value})
    return result


def _contrasts(context, partitions=None):
    plan = context["plan"]
    return [row for qid, design in plan["designs"].items()
            if partitions is None or plan["partitions"][qid] in partitions for row in design["contrasts"]]


def _status(context, state, **extra):
    value = {"schema": "graph_credit_g2_cpu_workflow_v1", "run_id": context["plan"]["run_id"],
             "state": state, "plan_sha256": context["plan_sha256"], "updated_unix": time.time(),
             "scientific_completion": False, **extra}
    atomic_json(context["root"], stage_path(context, "WORKFLOW_STATUS.json"), value)
    return value


def verify_predictions(context, global_data, predictor):
    root = context["root"]
    marker_relative = stage_path(context, "PREDICTIONS_FROZEN.json")
    marker = read_json(root, marker_relative)
    require(marker.get("schema") == "graph_credit_g2_predictions_frozen_v1" and marker.get("state") == "PREDICTIONS_FROZEN"
            and marker.get("plan_sha256") == context["plan_sha256"]
            and marker.get("source_manifest_sha256") == context["source_sha256"]
            and marker.get("pipeline_source_manifest_sha256") == context["pipeline_sha256"]
            and marker.get("global_outputs_frozen_sha256") == global_data["marker_sha256"]
            and marker.get("feature_sha256") == global_data["feature_sha256"]
            and marker.get("features_receipt_sha256") == global_data["feature_receipt_sha256"],
            "Prediction freeze lineage differs from this generation/features/source")
    check_sha(root, stage_path(context, "parameters.json"), marker["parameters_sha256"])
    check_sha(root, stage_path(context, "heldout_predictions.json"), marker["predictions_sha256"])
    check_sha(root, stage_path(context, "FIRST_TRAIN_TUNE_LABELS_OPENED.json"), marker["first_train_tune_labels_opened_sha256"])
    check_sha(root, stage_path(context, "TRAIN_TUNE_LABEL_RELEASE.json"), marker["train_tune_label_release_sha256"])
    parameters = read_json(root, stage_path(context, "parameters.json"))
    predictions = decode(bytes_read(root, stage_path(context, "heldout_predictions.json")))
    require(isinstance(predictions, list) and len(predictions) == 120 * 24
            and predictor.digest(predictions) == marker["predictions_semantic_sha256"]
            and parameters["parameter_sha256"] == marker["parameters_semantic_sha256"], "Prediction semantic digest/support changed")
    require(predictor.digest({key: value for key, value in parameters.items() if key != "parameter_sha256"}) == parameters["parameter_sha256"],
            "Predictor parameter semantic SHA mismatch")
    expected = predictor.predict(parameters, global_data["phi"], context["plan"]["masks"], _contrasts(context, {"test"}))
    require(predictions == expected, "Heldout predictions differ from frozen parameters and feature transform")
    return marker, parameters, predictions, file_sha(root, marker_relative)


def predict_stage(context, global_data, predictor):
    root = context["root"]
    existing = safe_path(root, stage_path(context, "PREDICTIONS_FROZEN.json"), existing=False)
    if existing.exists():
        marker, parameters, predictions, marker_hash = verify_predictions(context, global_data, predictor)
        analysis = safe_path(root, stage_path(context, "G2_ANALYSIS.json"), existing=False)
        if analysis.exists():
            value = read_json(root, stage_path(context, "G2_ANALYSIS.json"))
            require(value.get("analysis_semantic_sha256") == digest({key: item for key, item in value.items() if key != "analysis_semantic_sha256"})
                    and value.get("plan_sha256") == context["plan_sha256"] and value.get("predictions_frozen_sha256") == marker_hash
                    and value.get("scientific_gate") in {"G2_PASS", "INSUFFICIENT_PREDICTIVE_GAIN"},
                    "Existing analysis no longer matches the completed predictor")
            check_sha(root, stage_path(context, "G2_COST_SUMMARY.json"), value["cost_summary_sha256"])
            check_sha(root, stage_path(context, "FIRST_HELDOUT_LABELS_OPENED.json"), value["first_heldout_labels_opened_sha256"])
            check_sha(root, stage_path(context, "HELDOUT_LABEL_RELEASE.json"), value["heldout_label_release_sha256"])
            return _status(context, "G2_EVALUATION_COMPLETE_REVIEW_REQUIRED", scientific_gate=value["scientific_gate"],
                           analysis_sha256=file_sha(root, stage_path(context, "G2_ANALYSIS.json")), idempotent_already_complete=True)
        return _status(context, "PREDICTIONS_FROZEN_AWAITING_HELDOUT_LABELS", predictions_frozen_sha256=marker_hash,
                       idempotent_already_complete=True, heldout_labels_read=False)
    release = _release(context, "TRAIN_TUNE_LABEL_RELEASE.json", state="TRAIN_TUNE_LABELS_RELEASED",
                       binding_key="global_outputs_frozen_sha256", binding_sha=global_data["marker_sha256"],
                       label_fields=("train_labels_sha256", "tune_labels_sha256"))
    if release is None:
        return _status(context, "WAITING_FOR_TRAIN_TUNE_LABEL_RELEASE", labels_read=False)
    cpu_started, wall_started = time.process_time(), time.monotonic()
    release_value, release_hash = release
    label_hashes = {part: release_value[part + "_labels_sha256"] for part in ("train", "tune")}
    labels, event_hash = _labels_after_event(context, ("train", "tune"), label_hashes, "FIRST_TRAIN_TUNE_LABELS_OPENED.json",
                                           {"global_outputs_frozen_sha256": global_data["marker_sha256"], "label_release_sha256": release_hash})
    scored = _scored(context, global_data, labels, load_scorer(context))
    parameters = predictor.fit_and_select(scored["train"], scored["tune"], global_data["phi"], context["plan"]["masks"], _contrasts(context))
    require(len(parameters["train_question_ids"]) == 200 and len(parameters["tune_question_ids"]) == 80,
            "Fitted predictor uses a different training/tuning population")
    predictions = predictor.predict(parameters, global_data["phi"], context["plan"]["masks"], _contrasts(context, {"test"}))
    require(len(predictions) == 2880, "Heldout prediction coverage is not 120 x 24")
    verify_context_unchanged(context)
    check_sha(root, stage_path(context, "GLOBAL_OUTPUTS_FROZEN.json"), global_data["marker_sha256"])
    parameter_hash = atomic_json(root, stage_path(context, "parameters.json"), parameters)
    prediction_hash = atomic_json(root, stage_path(context, "heldout_predictions.json"), predictions)
    marker = {"schema": "graph_credit_g2_predictions_frozen_v1", "state": "PREDICTIONS_FROZEN", "created_unix": time.time(),
              "run_id": context["plan"]["run_id"], "plan_sha256": context["plan_sha256"],
              "source_manifest_sha256": context["source_sha256"], "pipeline_source_manifest_sha256": context["pipeline_sha256"],
              "parameters_sha256": parameter_hash, "predictions_sha256": prediction_hash,
              "parameters_semantic_sha256": parameters["parameter_sha256"], "predictions_semantic_sha256": predictor.digest(predictions),
              "feature_sha256": global_data["feature_sha256"], "features_receipt_sha256": global_data["feature_receipt_sha256"],
              "global_outputs_frozen_sha256": global_data["marker_sha256"],
              "first_train_tune_labels_opened_sha256": event_hash, "train_tune_label_release_sha256": release_hash,
              "training_labels_sha256": label_hashes, "train_questions": 200, "tune_questions": 80, "heldout_questions": 120,
              "heldout_prediction_rows": 2880, "heldout_labels_read": False,
              "cpu_seconds": time.process_time() - cpu_started, "wall_seconds": time.monotonic() - wall_started,
              "scientific_completion": False}
    marker_hash = atomic_json(root, stage_path(context, "PREDICTIONS_FROZEN.json"), marker, exclusive=True)
    return _status(context, "PREDICTIONS_FROZEN_AWAITING_HELDOUT_LABELS", predictions_frozen_sha256=marker_hash,
                   heldout_labels_read=False)


def generation_cost_summary(context, global_data):
    """Count each physical formal/qualification ledger once, retaining unknowns."""
    audit = global_data["physical_cost_audit"]
    seen, summaries = set(), {}
    metrics = ("input_tokens", "output_tokens", "gpu_seconds", "elapsed_seconds")
    for category, field in (("formal", "physical_ledgers"), ("qualification", "qualification_ledgers")):
        summary = {"calls": 0, "successful_calls": 0, "failed_calls": 0,
                   "metrics": {name: {"known_sum": 0.0, "unknown_calls": 0} for name in metrics}}
        for descriptor in audit.get(field, []):
            check_sha(context["root"], descriptor["path"], descriptor["sha256"])
            with safe_path(context["root"], descriptor["path"]).open("rb") as handle:
                for line in handle:
                    require(line.endswith(b"\n") and line.strip(), "Incomplete physical-cost record")
                    row = decode(line)
                    call_id = row.get("call_id")
                    require(isinstance(call_id, str) and call_id and call_id not in seen,
                            "Physical call was counted twice across formal/qualification ledgers")
                    require(row.get("status") in {"SUCCESS", "FAILED"}, "Unknown physical-call status")
                    seen.add(call_id)
                    summary["calls"] += 1
                    summary["successful_calls" if row["status"] == "SUCCESS" else "failed_calls"] += 1
                    for name in metrics:
                        value = row.get(name)
                        if value is None:
                            summary["metrics"][name]["unknown_calls"] += 1
                        else:
                            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                                    "Malformed physical metric: " + name)
                            summary["metrics"][name]["known_sum"] += value
        for value in summary["metrics"].values():
            value["complete_total"] = value["known_sum"] if value["unknown_calls"] == 0 else None
        summaries[category] = summary
    require(summaries["formal"]["calls"] == audit.get("formal_calls"), "Formal paid-call audit differs from ledgers")
    summaries["formal"]["orphan_call_ids"] = audit.get("orphan_call_ids", [])
    summaries["formal"]["failed_call_ids"] = audit.get("failed_call_ids", [])
    summaries["inference_retries"] = audit.get("inference_retries")
    summaries["unrecorded_inflight_call_upper_bound"] = audit.get("unrecorded_inflight_call_upper_bound")
    return summaries


def evaluate_stage(context, global_data, predictor):
    root = context["root"]
    marker, parameters, predictions, marker_hash = verify_predictions(context, global_data, predictor)
    release = _release(context, "HELDOUT_LABEL_RELEASE.json", state="HELDOUT_LABELS_RELEASED",
                       binding_key="predictions_frozen_sha256", binding_sha=marker_hash, label_fields=("heldout_labels_sha256",))
    if release is None:
        return _status(context, "PREDICTIONS_FROZEN_AWAITING_HELDOUT_LABELS", predictions_frozen_sha256=marker_hash,
                       heldout_labels_read=False)
    release_value, release_hash = release
    analysis_relative = stage_path(context, "G2_ANALYSIS.json")
    prior_analysis = None
    if safe_path(root, analysis_relative, existing=False).exists():
        prior_analysis = read_json(root, analysis_relative)
        value = prior_analysis
        claimed = prior_analysis.get("analysis_semantic_sha256")
        require(claimed == digest({key: item for key, item in value.items() if key != "analysis_semantic_sha256"})
                and value.get("plan_sha256") == context["plan_sha256"] and value.get("predictions_frozen_sha256") == marker_hash
                and value.get("heldout_label_release_sha256") == release_hash
                and value.get("heldout_labels_sha256") == release_value["heldout_labels_sha256"]
                and value.get("scientific_gate") in {"G2_PASS", "INSUFFICIENT_PREDICTIVE_GAIN"},
                "Existing analysis is incomplete or belongs to a different release")
        check_sha(root, stage_path(context, "G2_COST_SUMMARY.json"), value["cost_summary_sha256"])
        check_sha(root, stage_path(context, "FIRST_HELDOUT_LABELS_OPENED.json"), value["first_heldout_labels_opened_sha256"])
    cpu_started, wall_started = time.process_time(), time.monotonic()
    labels, event_hash = _labels_after_event(context, ("test",), {"test": release_value["heldout_labels_sha256"]},
                                           "FIRST_HELDOUT_LABELS_OPENED.json", {"predictions_frozen_sha256": marker_hash,
                                           "global_outputs_frozen_sha256": global_data["marker_sha256"], "label_release_sha256": release_hash})
    scored = _scored(context, global_data, labels, load_scorer(context))
    result = predictor.evaluate(scored["test"], predictions, context["plan"]["masks"], prediction_freeze={
        "state": "PREDICTIONS_FROZEN", "predictions_sha256": marker["predictions_semantic_sha256"]})
    require(result["primary"]["question_count"] == 120 and result.get("contrasts_per_question") == 24,
            "Evaluation does not cover the complete heldout population")
    if prior_analysis is not None:
        require(digest({key: prior_analysis[key] for key in result}) == digest(result),
                "Deterministic heldout replay differs from the existing analysis")
        return _status(context, "G2_EVALUATION_COMPLETE_REVIEW_REQUIRED", scientific_gate=result["scientific_gate"],
                       analysis_sha256=file_sha(root, analysis_relative), deterministic_replay_verified=True,
                       idempotent_already_complete=True)
    verify_context_unchanged(context)
    check_sha(root, stage_path(context, "PREDICTIONS_FROZEN.json"), marker_hash)
    cost = {"schema": "graph_credit_g2_complete_cost_summary_v1", "plan_sha256": context["plan_sha256"],
            "generation": generation_cost_summary(context, global_data),
            "generation_source": global_data["marker"]["artifacts"]["physical_cost_audit"],
            "predict_cpu_seconds": marker["cpu_seconds"], "predict_wall_seconds": marker["wall_seconds"],
            "evaluate_cpu_seconds": time.process_time() - cpu_started, "evaluate_wall_seconds": time.monotonic() - wall_started,
            "physical_ledgers_counted_once": True, "alias_labels_added": False,
            "label_access_attempt_directory": stage_path(context, "label_access_attempts"),
            "cpu_attempt_directory": stage_path(context, "cpu_attempts"),
            "additional_or_interrupted_cpu_attempt_cost": "Not included in successful-phase timings; retained in access/CPU attempt events, never imputed zero",
            "scope": "Frozen canonical G2 formal and engineering cost audit plus the two CPU workflow phases"}
    cost_hash = atomic_json(root, stage_path(context, "G2_COST_SUMMARY.json"), cost)
    result.update(run_id=context["plan"]["run_id"], plan_sha256=context["plan_sha256"],
                  source_manifest_sha256=context["source_sha256"], pipeline_source_manifest_sha256=context["pipeline_sha256"],
                  predictions_frozen_sha256=marker_hash, heldout_label_release_sha256=release_hash,
                  heldout_labels_sha256=release_value["heldout_labels_sha256"], first_heldout_labels_opened_sha256=event_hash,
                  cost_summary_sha256=cost_hash, completed_unix=time.time(),
                  coverage={"unique_graph_outputs": len(global_data["rows"]), "question_seed_units": 2000,
                            "train_questions": 200, "tune_questions": 80, "heldout_questions": 120,
                            "heldout_credit_targets": 2880, "complete": True},
                  independent_replay_status="PENDING_INDEPENDENT_ROOT_AUDIT")
    result["analysis_semantic_sha256"] = digest(result)
    analysis_hash = atomic_json(root, analysis_relative, result, exclusive=True)
    return _status(context, "G2_EVALUATION_COMPLETE_REVIEW_REQUIRED", scientific_gate=result["scientific_gate"], analysis_sha256=analysis_hash)


@contextmanager
def cpu_attempt(context, mode):
    relative = stage_path(context, "cpu_attempts/" + mode + "_" + uuid.uuid4().hex[:16] + ".json")
    cpu_started, wall_started = time.process_time(), time.monotonic()
    record = {"schema": "graph_credit_g2_cpu_attempt_v1", "state": "RUNNING", "mode": mode,
              "plan_sha256": context["plan_sha256"], "started_unix": time.time(),
              "cpu_seconds": None, "wall_seconds": None}
    atomic_json(context["root"], relative, record, exclusive=True)
    try:
        yield
    except BaseException as exc:
        record.update(state="FAILED", error_type=type(exc).__name__)
        raise
    else:
        record["state"] = "COMPLETED"
    finally:
        record.update(completed_unix=time.time(), cpu_seconds=time.process_time() - cpu_started,
                      wall_seconds=time.monotonic() - wall_started)
        atomic_json(context["root"], relative, record)


def run(root, plan_path, mode):
    require(mode in {"predict", "evaluate"}, "Unknown CPU workflow mode")
    context = load_context(root, plan_path)
    with workflow_lease(context), cpu_attempt(context, mode):
        global_data = verify_global(context)
        predictor = load_predictor(context)
        return predict_stage(context, global_data, predictor) if mode == "predict" else evaluate_stage(context, global_data, predictor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("predict", "evaluate"), required=True)
    args = parser.parse_args()
    try:
        result = run(args.root, args.plan, args.mode)
    except Exception as exc:
        print(canonical({"state": "CPU_WORKFLOW_FAILED", "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
        return 2
    print(canonical(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
