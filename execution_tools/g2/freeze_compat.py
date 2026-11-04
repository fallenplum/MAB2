"""Explicit, label-free recovery of native execution insertion order for G2 freeze.

The frozen collector, executor and scientific inputs remain byte-identical. Only
deserialization order is restored before the original exact validator runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time


TIMING_FIELDS = {"model_elapsed_seconds", "gpu_seconds"}
RECEIPT_FOLDER = "freeze_compatibility"
INCIDENT_SHA256 = {
    "failure_status": "ac269f13f75cdd6cfaf8fd0803f77e613b0791c7789b02b30ed98c357d5f3f89",
    "stop_request": "7c0d8fa202deac51bbcce7e2a066c20117e1d85e035e9a2f208f3825c7405039",
    "stop_receipt": "a4a6fde67f6b56977253c8ad4f0435a79163e23b895f2fa4e4c4f773a607106e",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exclusive_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def native_node_order(rounds):
    require(type(rounds) is int and rounds in (2, 3), "Unsupported native graph rounds")
    return [f"r{r}:w{w}" for r in range(1, rounds + 1) for w in range(4)] + ["aggregate"]


def ordered_deserializer(original, module, value, rounds):
    """Retain every native record, changing only its dictionary iteration order."""
    expected = native_node_order(rounds)
    require(isinstance(value.get("nodes"), dict) and set(value["nodes"]) == set(expected),
            "Execution has missing or extra nodes before native-order recovery")
    execution = original(module, value)
    require(set(execution.nodes) == set(expected), "Native deserializer changed graph node support")
    execution.nodes = {name: execution.nodes[name] for name in expected}
    return execution


def compare_roundtrip(collector, module, value, row, rounds):
    """Reject every mismatch that cannot be explained by native summation order."""
    original = collector.execution_from_dict(module, value)
    restored = ordered_deserializer(collector.execution_from_dict, module, value, rounds)
    require(row["answer"] == value["answer"] == restored.answer == restored.nodes["aggregate"].parsed["answer"],
            "Answer differs before order recovery")
    require(row["service_output"] == restored.nodes["aggregate"].generation.text
            and row["service_message_metadata"] == restored.nodes["aggregate"].message_metadata,
            "Raw aggregate payload differs before order recovery")
    changed = []
    for name, physical in (("physical_cost", True), ("logical_service_cost", False)):
        require(value[name] == row[name], "Stored execution and raw cost differ")
        expected, old, new = row[name], original.cost(physical), restored.cost(physical)
        require(expected == new, "Native-order cost does not exactly match the raw cost")
        require(set(old) == set(new) == set(expected), "Cost field support changed")
        require(all(type(expected[key]) is type(new[key]) and type(value[name][key]) is type(new[key]) for key in new),
                "Stored cost numeric types differ from their native representation")
        differences = {key for key in old if old[key] != new[key]}
        require(differences <= TIMING_FIELDS, "Order recovery changed a non-timing cost field")
        for field in differences:
            require(type(old[field]) is float and type(new[field]) is float,
                    "Order-sensitive timing field is not a native float")
            changed.append(name + "." + field)
    return restored, changed


def runtime_identity(context):
    expected = context["plan"]["model"]["identity"]["python"]
    require(platform.python_implementation() == "CPython" and platform.python_version() == expected,
            "Recovery must use the original CPython version")
    return {"implementation": platform.python_implementation(), "python": platform.python_version(),
            "sys_version": sys.version, "hexversion": sys.hexversion,
            "executable": sys.executable,
            "executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            "sum_order_probe": {"native": sum([.1] * 8 + [1.0]), "json_sorted": sum([1.0] + [.1] * 8)}}


def assert_label_free(context):
    output, plan = context["output"], context["plan"]
    forbidden = ("GLOBAL_OUTPUTS_FROZEN.json", "TRAIN_TUNE_LABEL_RELEASE.json", "HELDOUT_LABEL_RELEASE.json",
                 "FIRST_TRAIN_TUNE_LABELS_OPENED.json", "FIRST_HELDOUT_LABELS_OPENED.json",
                 "LOCAL_DATA_OWNER_FIRST_LABEL_READ.json", "PREDICTIONS_FROZEN.json", "G2_ANALYSIS.json")
    require(not any((output / name).exists() for name in forbidden), "Freeze recovery is closed after output freeze or gold exposure")
    require(not any(context["collector"].safe_path(context["root"], path).exists()
                    for path in plan["label_release"].values()), "Released label files exist before recovery")


def stopped_failure_evidence(context):
    """Bind the historical failed supervisor without stopping or launching processes."""
    import psutil
    collector, output = context["collector"], context["output"]
    state = collector.read_json(output / "STATUS.json")
    require(state.get("schema") == "graph_credit_g2_supervisor_v1" and state.get("phase") == "FAILED"
            and state.get("run_id") == context["plan"]["run_id"] and state.get("plan_sha256") == context["plan_hash"]
            and state.get("labels_read") is False and state.get("active_children") == []
            and state.get("error") == "Collector global output freeze failed", "Supervisor failure is not the acknowledged freeze incident")
    original_stop_path = output / "STOP_REQUESTED.json"
    stop_path = context.get("historical_stop_archive", original_stop_path)
    stop = collector.read_json(stop_path)
    require(stop.get("schema") == "graph_credit_g2_stop_v1" and stop.get("run_id") == context["plan"]["run_id"]
            and stop.get("plan_sha256") == context["plan_hash"] and stop.get("labels_read") is False
            and stop.get("reason") == "Collector global output freeze failed", "Stop request is not the historical freeze failure")
    stop_receipt = collector.read_json(output / "private/STOP_RECEIPT.json")
    require(stop_receipt.get("plan_sha256") == context["plan_hash"] and stop_receipt.get("labels_read") is False
            and stop_receipt.get("unsettled_child_pids") == []
            and stop_receipt.get("terminated_owned_pids") == []
            and stop_receipt.get("identity_mismatch_pids_not_signalled") == []
            and stop_receipt.get("unverified_inflight_calls_upper_bound") == 0
            and stop_receipt.get("persistence_errors") == [], "Failure stop has unsettled processes or accounting")
    children = list(collector.read_jsonl(output / "private/CHILDREN.jsonl"))
    require(all(row.get("plan_sha256") == context["plan_hash"] for row in children), "Historical child belongs to a foreign plan")
    collect_children = [row for row in children if row.get("mode") == "collect"]
    require(len(collect_children) == 8 and {row.get("gpu") for row in collect_children} == set(range(8)),
            "Original eight physical collector identities are missing")
    identities = [{"pid": state["pid"], "create_time": state["process_create_time"], "owner": state["owner_username"]}] + children
    for identity in identities:
        require(identity.get("create_time") is not None, "Historical process identity has no creation time")
        try:
            process = psutil.Process(identity["pid"])
            if process.create_time() == identity["create_time"]:
                require(process.username() == identity["owner"], "Historical process owner is inconsistent")
                raise ValueError("An original supervisor or collector process is still alive")
        except psutil.NoSuchProcess:
            pass
    stop_descriptor = collector.artifact(context["root"], stop_path)
    # Keep the historical identity stable after the explicitly authorized move.
    stop_descriptor["path"] = original_stop_path.relative_to(context["root"]).as_posix()
    evidence = {"failure_status": collector.artifact(context["root"], output / "STATUS.json"),
                "stop_request": stop_descriptor,
                "stop_receipt": collector.artifact(context["root"], output / "private/STOP_RECEIPT.json"),
                "children": collector.artifact(context["root"], output / "private/CHILDREN.jsonl"),
                "all_historical_processes_absent_or_pid_reused": True}
    require(all(evidence[key]["sha256"] == expected for key, expected in INCIDENT_SHA256.items()),
            "Historical incident bytes differ from the independently observed failure")
    return evidence


def bindings(context):
    collector = context["collector"]
    collector.verify_manifest(context["root"], context["plan"]["source_manifest"], expected_count=18)
    collector.verify_manifest(context["root"], context["plan"]["pipeline_source_manifest"], expected_count=5)
    collector.require_frozen(context)
    assert_label_free(context)
    return {"run_id": context["plan"]["run_id"], "plan_sha256": context["plan_hash"],
            "source_manifest_sha256": context["source_hash"], "pipeline_source_manifest_sha256": context["pipeline_hash"],
            "protocol_hash": context["plan"]["protocol_hash"], "model_identity_sha256": collector.digest(context["plan"]["model"]["identity"]),
            "wrapper_sha256": collector.file_digest(Path(__file__)), "runtime": runtime_identity(context),
            "failure_evidence": stopped_failure_evidence(context), "labels_read": False, "first_gold_opened_unix": None}


def diagnose(context):
    collector = context["collector"]
    binding = bindings(context)
    changes, seen, descriptors, consumed, all_call_ids, completed_units = Counter(), set(), [], set(), set(), set()
    units = {(unit["question_id"], unit["seed"]): unit for unit in context["plan"]["units"]}
    for worker in context["plan"]["qualification"]["worker_ids"]:
        segment = context["output"] / "segments" / worker
        state = collector.read_json(segment / "STATUS.json")
        require(state.get("phase") == "GENERATION_COMPLETE_UNSCORED" and state.get("generation_complete") is True
                and state.get("labels_read") is False and state.get("plan_sha256") == context["plan_hash"], "Collector is not complete and label-free")
        descriptors.extend(collector.artifact(context["root"], segment / name) for name in
                           ("STATUS.json", "run_manifest.json", "physical_calls.jsonl", "raw_executions.jsonl"))
        ledger = {}
        for call in collector.read_jsonl(segment / "physical_calls.jsonl"):
            require(call["call_id"] not in all_call_ids and call.get("status") in ("SUCCESS", "FAILED"),
                    "Duplicate or invalid physical generation record")
            all_call_ids.add(call["call_id"])
            ledger[call["call_id"]] = call
        current_unit, prior, schedule, next_index = None, {}, [], 0
        for row in collector.read_jsonl(segment / "raw_executions.jsonl"):
            key = (row["question_id"], row["seed"], row["mask_sha256"])
            require(key not in seen and row["plan_sha256"] == context["plan_hash"] and row["labels_read"] is False
                    and row["quality"] is None, "Duplicate or non-frozen raw row")
            seen.add(key)
            raw_path = collector.safe_path(segment, row["raw_path"])
            execution_path = collector.safe_path(segment, row["execution_path"])
            require(collector.file_digest(raw_path) == row["raw_sha256"] and collector.file_digest(execution_path) == row["execution_sha256"],
                    "Raw or execution bytes differ from their committed hash")
            require(collector.read_json(raw_path) == {name: value for name, value in row.items() if name not in ("raw_path", "raw_sha256")},
                    "Raw atomic object differs from its original index")
            value = collector.read_json(execution_path)
            restored, fields = compare_roundtrip(collector, context["module"], value, row, context["config"].rounds)
            changes.update(fields)
            unit_key = (row["question_id"], row["seed"])
            if unit_key != current_unit:
                require(next_index == len(schedule), "Original unit index is incomplete or interleaved")
                require(unit_key in units and unit_key not in completed_units, "Unplanned or repeated collection unit")
                completed_units.add(unit_key)
                current_unit, prior, next_index = unit_key, {}, 0
                schedule = collector.unique_mask_requests(units[unit_key]["mask_requests"])
            require(row["mask_index"] == next_index and next_index < len(schedule), "Original mask schedule order changed")
            item = schedule[next_index]
            unit = units[unit_key]
            mask_hash = collector.digest(item["mask"])
            expected_raw = {"schema": "g2_collected_graph_v1", "question_id": unit["question_id"], "seed": unit["seed"],
                            "unit_id": collector.unit_id(unit["question_id"], unit["seed"]), "graph_mask": item["mask"],
                            "mask_sha256": mask_hash, "mask_id": "".join(map(str, item["mask"])), "mask_index": next_index,
                            "mask_aliases": item["aliases"], "request_indices": item["request_indices"],
                            "reference_mask": item["reference_mask"], "partition": unit["partition"],
                            "plan_sha256": context["plan_hash"], "protocol_hash": context["plan"]["protocol_hash"],
                            "source_manifest_sha256": context["source_hash"], "labels_read": False, "quality": None}
            require(all(row.get(name) == expected for name, expected in expected_raw.items()),
                    "Committed raw metadata differs from the frozen original mask request")
            require(row["question_fingerprint"] == restored.question_fingerprint,
                    "Raw question fingerprint differs from its native execution")
            collector.validate_execution(context, units[unit_key], item, row, restored, prior, ledger, consumed)
            prior[tuple(item["mask"])] = restored
            next_index += 1
        require(next_index == len(schedule), "Original worker ends inside a collection unit")
    require(len(seen) == context["expected_graphs"], "Diagnosis does not cover every expected graph")
    require(completed_units == set(units), "Diagnosis does not cover every collection unit")
    require(changes, "No native-order accounting discrepancy was found")
    require(bindings(context) == binding, "Source, stop, runtime or before-gold boundary changed during diagnosis")
    report = {"schema": "graph_credit_g2_freeze_order_preflight_v1", "status": "PASS_ORDER_ONLY_DIAGNOSIS",
              **binding, "observations": len(seen), "changed_field_counts": dict(sorted(changes.items())),
              "immutable_segment_descriptors": sorted(descriptors, key=lambda value: value["path"]),
              "native_node_order": native_node_order(context["config"].rounds),
              "original_execution_validations": len(seen), "complete_original_freeze_pending": True,
              "scientific_completion": False, "created_unix": time.time()}
    path = context["output"] / RECEIPT_FOLDER / "PREFLIGHT.json"
    collector.atomic_json(path, report, exclusive=True)
    return {"status": report["status"], "preflight_path": path.relative_to(context["root"]).as_posix(),
            "preflight_sha256": collector.file_digest(path), "observations": len(seen), "changed_field_counts": report["changed_field_counts"],
            "stop_sha256": binding["failure_evidence"]["stop_request"]["sha256"], "labels_read": False}


@contextmanager
def compatibility_hooks(context, approved_binding, on_global=None):
    collector = context["collector"]
    original_deserializer, original_atomic = collector.execution_from_dict, collector.atomic_json
    def restore(module, value):
        return ordered_deserializer(original_deserializer, module, value, context["config"].rounds)
    def recheck_boundary():
        collector.check_stop(context)
        require(bindings(context) == approved_binding, "Acknowledged stop, provenance or before-gold boundary changed")
    def publish(path, value, **options):
        if Path(path) == context["output"] / "GLOBAL_OUTPUTS_FROZEN.json":
            recheck_boundary()
            if on_global is not None:
                on_global(value)
            collector.check_stop(context)
        return original_atomic(path, value, **options)
    collector.execution_from_dict, collector.atomic_json = restore, publish
    try:
        yield
    finally:
        collector.execution_from_dict, collector.atomic_json = original_deserializer, original_atomic


def recover_freeze(context, preflight_sha256, acknowledge_stop_sha256):
    collector, output = context["collector"], context["output"]
    path = output / RECEIPT_FOLDER / "PREFLIGHT.json"
    require(collector.file_digest(path) == preflight_sha256, "Preflight acknowledgement differs from its actual bytes")
    preflight, current = collector.read_json(path), bindings(context)
    require(preflight.get("schema") == "graph_credit_g2_freeze_order_preflight_v1"
            and preflight.get("status") == "PASS_ORDER_ONLY_DIAGNOSIS"
            and preflight.get("observations") == context["expected_graphs"]
            and all(preflight.get(key) == value for key, value in current.items()), "Preflight does not match this recovery runtime and evidence")
    require(acknowledge_stop_sha256 == current["failure_evidence"]["stop_request"]["sha256"], "Historical stop was not explicitly acknowledged")
    for descriptor in preflight["immutable_segment_descriptors"]:
        actual = collector.artifact(context["root"], collector.safe_path(context["root"], descriptor["path"]))
        require(actual == descriptor, "Immutable generation segment changed after preflight")
    intent = {"schema": "graph_credit_g2_freeze_order_authorization_v1", **current,
              "preflight_sha256": preflight_sha256, "acknowledge_stop_sha256": acknowledge_stop_sha256,
              "action": "RESTORE_NATIVE_NODE_ORDER_THEN_RUN_ORIGINAL_EXACT_FREEZE", "created_unix": time.time()}
    intent_path = output / RECEIPT_FOLDER / "AUTHORIZATION.json"
    collector.atomic_json(intent_path, intent, exclusive=True)
    failed_status_path = output / "STATUS.json"
    failed_archive = output / RECEIPT_FOLDER / "STATUS.failed.json"
    failed_bytes = failed_status_path.read_bytes()
    require(collector.digest_bytes(failed_bytes) == current["failure_evidence"]["failure_status"]["sha256"],
            "Historical FAILED status changed before archival")
    exclusive_bytes(failed_archive, failed_bytes)
    stop_path = output / "STOP_REQUESTED.json"
    archive = output / RECEIPT_FOLDER / "STOP_REQUESTED.consumed.json"
    require(not archive.exists() and collector.file_digest(stop_path) == acknowledge_stop_sha256,
            "Historical stop changed before its authorized archival")
    # Preserve the exact historical bytes. The original check_stop remains live
    # and rejects any stop request subsequently created at the original path.
    stop_path.rename(archive)
    require(collector.file_digest(archive) == acknowledge_stop_sha256, "Archived historical stop bytes changed")
    context["historical_stop_archive"] = archive
    consumed = {"schema": "graph_credit_g2_freeze_stop_consumed_v1", "run_id": context["plan"]["run_id"],
                "plan_sha256": context["plan_hash"], "authorization_sha256": collector.file_digest(intent_path),
                "original_stop": current["failure_evidence"]["stop_request"],
                "archived_stop": collector.artifact(context["root"], archive),
                "original_check_stop_unchanged": True, "consumed_unix": time.time(), "labels_read": False}
    consumed_path = output / RECEIPT_FOLDER / "STOP_CONSUMED.json"
    collector.atomic_json(consumed_path, consumed, exclusive=True)
    require(bindings(context) == current, "Recovery boundary changed after historical stop archival")
    collector.check_stop(context)
    def before_global(report):
        receipt = {"schema": "graph_credit_g2_freeze_order_validation_v1", **current, "status": "PASS_ORIGINAL_EXACT_VALIDATION",
                   "preflight_sha256": preflight_sha256, "authorization_sha256": collector.file_digest(intent_path),
                   "stop_consumed_sha256": collector.file_digest(consumed_path),
                   "archived_stop_path": archive.relative_to(context["root"]).as_posix(),
                   "original_validator_unchanged": True, "observations": report["observations"],
                   "original_global_marker_sha256": collector.digest_bytes((collector.canonical(report) + "\n").encode("utf-8")),
                   "artifact_hashes": report["artifact_hashes"], "validated_unix": time.time(), "scientific_completion": False}
        # This proof is durable before the original GLOBAL marker becomes visible.
        collector.atomic_json(output / RECEIPT_FOLDER / "VALIDATION.json", receipt, exclusive=True)
    with compatibility_hooks(context, current, before_global):
        report = collector.freeze_outputs(context)
    collector.check_stop(context)
    global_path = output / "GLOBAL_OUTPUTS_FROZEN.json"
    validation_path = output / RECEIPT_FOLDER / "VALIDATION.json"
    validation = collector.read_json(validation_path)
    global_hash = collector.file_digest(global_path)
    require(global_hash == validation["original_global_marker_sha256"], "Original GLOBAL bytes differ from their validation proof")
    success_path = output / RECEIPT_FOLDER / "RECOVERY_COMPLETE.json"
    success = {"schema": "graph_credit_g2_freeze_order_recovery_v1", "status": "PASS_ORIGINAL_EXACT_FREEZE",
               **current, "global_outputs_frozen_sha256": global_hash, "observations": report["observations"],
               "completed_units": report["completed_units"], "preflight_sha256": preflight_sha256,
               "authorization_sha256": collector.file_digest(intent_path), "validation_sha256": collector.file_digest(validation_path),
               "stop_consumed_sha256": collector.file_digest(consumed_path),
               "archived_failed_status": collector.artifact(context["root"], failed_archive),
               "archived_stop": collector.artifact(context["root"], archive),
               "scientific_completion": False, "completed_unix": time.time()}
    collector.check_stop(context)
    collector.atomic_json(success_path, success, exclusive=True)
    import psutil
    process = psutil.Process()
    recovered_status = {"schema": "graph_credit_g2_supervisor_v1", "publisher": "freeze_compat",
                        "phase": "WAITING_TRAIN_TUNE_LABELS", "run_id": current["run_id"],
                        "plan_sha256": current["plan_sha256"], "source_manifest_sha256": current["source_manifest_sha256"],
                        "pipeline_source_manifest_sha256": current["pipeline_source_manifest_sha256"],
                        "pid": process.pid, "process_create_time": process.create_time(), "owner_username": process.username(),
                        "active_children": [], "generation_complete": True, "workflow_complete": False,
                        "scientific_completion": False, "labels_read": False, "expected_units": report["expected_units"],
                        "expected_outputs": report["expected_outputs"], "successful_gpus": list(range(8)),
                        "global_outputs_frozen_sha256": global_hash,
                        "coverage": {"status": "PASS_LABEL_FREE_COVERAGE", "units": report["completed_units"],
                                     "unique_raw_graphs": report["observations"]},
                        "recovered_from": {"failed_status": success["archived_failed_status"], "stop_request": success["archived_stop"],
                                           "recovery_receipt": collector.artifact(context["root"], success_path)},
                        "updated_unix": time.time()}
    collector.check_stop(context)
    collector.atomic_json(output / "STATUS.json", recovered_status)
    return {"status": "GLOBAL_OUTPUTS_FROZEN_WITH_EXPLICIT_NATIVE_ORDER_COMPATIBILITY", "observations": report["observations"],
            "global_outputs_frozen_sha256": global_hash, "recovery_receipt_sha256": collector.file_digest(success_path), "labels_read": False}


def load_context(root, plan_path):
    require(plan_path.is_relative_to(root), "Recovery plan must belong to the project root")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    names = [name for name in plan["pipeline_source_manifest"] if Path(name).name == "collector.py"]
    require(len(names) == 1, "Frozen collector source is ambiguous")
    collector_path = root.joinpath(*names[0].split("/"))
    require(collector_path.resolve().is_relative_to(root), "Frozen collector path escapes root")
    descriptor = plan["pipeline_source_manifest"][names[0]]
    require(hashlib.sha256(collector_path.read_bytes()).hexdigest() == descriptor["sha256"], "Frozen collector bytes changed")
    spec = importlib.util.spec_from_file_location("g2_original_collector_freeze_compatibility", collector_path)
    collector = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = collector
    spec.loader.exec_module(collector)
    context = collector.load_plan(root, plan_path)
    context["collector"] = collector
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("diagnose", "freeze"), required=True)
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--acknowledge-stop-sha256")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    context = load_context(root, plan_path.resolve(strict=True))
    if args.mode == "diagnose":
        result = diagnose(context)
    else:
        require(args.preflight_sha256 and args.acknowledge_stop_sha256, "Freeze requires explicit preflight and historical stop acknowledgements")
        result = recover_freeze(context, args.preflight_sha256, args.acknowledge_stop_sha256)
    print(context["collector"].canonical(result), flush=True)


if __name__ == "__main__":
    main()
