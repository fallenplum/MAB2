"""Read-only cycle2 qualification evidence audit. Never signs acceptance.

Reads only explicit frozen plan/source and qualification output paths. No question
file, gold, labels, S results, GPU, model load, or output-directory writes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True


def load_runner(path):
    spec = importlib.util.spec_from_file_location("qualification_frozen_runner", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def raw_format(text):
    """Independent schema inspection; no answer correctness/labels involved."""
    value = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", value, re.I | re.S)
    if fence: value = fence.group(1).strip()
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out: raise ValueError("duplicate key")
            out[k] = v
        return out
    try:
        v = json.loads(value, object_pairs_hook=unique)
        if not isinstance(v, dict) or set(v) != {"answer", "evidence", "uncertainty"}: return False
        if not isinstance(v["evidence"], str) or not isinstance(v["uncertainty"], str): return False
        return isinstance(v["answer"], str) or (type(v["answer"]) in (int, float) and math.isfinite(v["answer"]))
    except (ValueError, TypeError): return False


def producer_float_sum(values, python_version):
    """Exact producer arithmetic, not a tolerance or a change to frozen artifacts.

    Python 3.12 changed built-in float sum. Frozen server 3.11 used ordinary
    left-to-right additions; local 3.13 audit must reproduce that explicitly.
    """
    version = tuple(int(x) for x in python_version.split(".")[:2])
    if version < (3, 12):
        total = 0
        for value in values: total += value
        return total
    if sys.version_info[:2] < (3, 12):
        raise RuntimeError("Audit runtime cannot reproduce post-3.12 producer float sum")
    return sum(values)


def original_execution_cost(result, physical_only, python_version):
    ordering = [f"r{r}:w{w}" for r in (1, 2) for w in range(4)]+["aggregate"]
    ns = [result.nodes[k] for k in ordering if not physical_only or not result.nodes[k].reused]
    return {"calls": len(ns), "input_tokens": sum(n.generation.input_tokens for n in ns),
            "output_tokens": sum(n.generation.output_tokens for n in ns),
            "received_message_tokens": sum(n.received_message_tokens for n in ns),
            "self_state_tokens": sum(n.self_state_tokens for n in ns),
            "model_elapsed_seconds": producer_float_sum((n.generation.elapsed_seconds for n in ns), python_version),
            "gpu_seconds": producer_float_sum((n.generation.gpu_seconds for n in ns), python_version),
            "peak_allocated_bytes": max((n.generation.gpu_peak_allocated_bytes for n in ns), default=0)}


def audit(plan_path, plan_sha256, output_root, source_root, runner_path):
    nb = load_runner(runner_path)
    nb.require(nb.file_sha(plan_path) == plan_sha256, "Frozen plan bytes mismatch")
    p = nb.read_json(plan_path)
    producer_python = p["model"]["identity"].get("python", f"{sys.version_info.major}.{sys.version_info.minor}")
    nb.require(p["schema"] == nb.SCHEMA and p["status"] == "FROZEN" and p["labels_read"] is False, "Not a frozen label-free cycle2 plan")
    nb.require(nb.file_sha(runner_path) == p["runner_sha256"], "Frozen runner bytes mismatch")
    module, _ = nb.load_operator(source_root, p["source_manifest"])
    cfg = module.GraphConfig(**{**p["graph"], "role_prompts": tuple(p["graph"]["role_prompts"])})
    from types import SimpleNamespace
    op = module.Executor(SimpleNamespace(identity=p["model"]["identity"]), cfg)
    nb.require(op.protocol_hash == p["executor_protocol_hash"], "Pinned operator identity mismatch")
    nb.require(len(p["episodes"]) == 8, "Episode count mismatch")
    for e in p["episodes"]:
        i = e["episode"]
        nb.require(type(i) is int and 0 <= i < 8 and e["seed"] == 3701+i,
                   "Episode/seed mapping mismatch")
        nb.require(e["permutation"] == [(j+i//2)%4 for j in range(4)] and e["complement"] is bool(i%2),
                   "Episode must match protocol shift=e//2, complement=e%2")
        nb.require(e["menu"] == nb.episode_menu(e["permutation"], e["complement"]), "Episode menu mismatch")
    nb.require([e["episode"] for e in p["episodes"]] == list(range(8)), "Repeated/reordered episode")
    menu = p["qualification_menu"]; nb.validate_menu(menu, module, qualification=True)
    nb.require(menu[:14] == nb.default_menu(), "Qualification template changed")
    replay_targets = [r["replay_of"] for r in menu if r.get("replay_of")]
    nb.require("B_toggle_9" in replay_targets, "Missing independently executed B_toggle_9 qualification replay")
    qunits = [u for u in p["units"] if u["phase"] == "qualification"]
    nb.require(len(qunits) == len({u["question_id"] for u in qunits}) == 6, "Need six distinct qualification units")
    nb.require(len({u["unit_id"] for u in qunits}) == 6 and all(u["seed"] == 3799 for u in qunits), "Qualification namespace mismatch")
    roots = Path(output_root); workers = []; all_formats = Counter(); unit_rows = []; reports = {}
    all_calls = set(); total_logical = total_physical_references = 0
    findings = []
    for setting in p["workers"]:
        index = setting["index"]; folder = roots/("worker_"+str(index))
        assigned = [u for u in qunits if u["worker_index"] == index]
        nb.require(assigned, "Missing worker qualification coverage")
        report_path = folder/"QUALIFICATION_COMPLETE.json"; report = nb.read_json(report_path)
        reports[str(index)] = nb.file_sha(report_path)
        nb.require(report["status"] == "COMPLETE_UNSCORED" and report["phase"] == "qualification"
                   and report["worker_index"] == index and report["plan_sha256"] == plan_sha256
                   and report["executor_protocol_hash"] == p["executor_protocol_hash"] and report["labels_read"] is False,
                   "Worker qualification report mismatch")
        listed = {u["unit_id"]: u["sha256"] for u in report["completed_units"]}
        nb.require(set(listed) == {u["unit_id"] for u in assigned} and len(listed) == len(report["completed_units"]), "Worker report unit coverage mismatch")
        resident = nb.read_jsonl(folder/"resident_wall.jsonl")
        nb.require(all(r.get("plan_sha256") == plan_sha256 for r in resident), "Foreign resident event")
        nb.require(all(r.get("phase") == "qualification" for r in resident if r["event"] == "START"), "Development already started before qualification audit")
        used = nb.resident_used(resident)
        nb.require(used <= setting["resident_wall_cap_seconds"], "Worker resident cap exceeded")
        nb.require(all(not r.get("resident_cap_exceeded", False) for r in resident), "A resident attempt exceeded its reservation")
        runtimes = list(folder.glob("RUNTIME_*.json")); nb.require(runtimes, "No loaded-model runtime receipt")
        for path in runtimes:
            r = nb.read_json(path)
            nb.require(r["model_identity"] == p["model"]["identity"] and r["plan_sha256"] == plan_sha256 and r["labels_read"] is False, "Runtime model identity mismatch")
        events = nb.read_jsonl(folder/"call_journal.jsonl")
        starts = {r["reservation_id"]: r for r in events if r["event"] == "CALL_START"}
        success = {r["call_id"]: r for r in events if r["event"] == "CALL_SUCCESS"}
        journal = SimpleNamespace(starts=starts, success=success)
        native_rows = nb.read_jsonl(folder/"physical_calls.jsonl")
        native = {r["call_id"]: r for r in native_rows}
        nb.require(len(native) == len(native_rows) and not all_calls.intersection(native), "Repeated native call ID across records/workers")
        all_calls.update(native)
        nb.require(all(e.get("unit_id") in listed and e.get("plan_sha256") == plan_sha256 for e in events), "Journal contains foreign unit/plan")
        costs = nb.ledger_summary(folder)
        paid = dict(success)
        for call_id, row in native.items():
            if row["status"] == "SUCCESS" and call_id not in paid: paid[call_id] = row
        costs["known_gpu_seconds"] = producer_float_sum((r["gpu_seconds"] for r in paid.values()), producer_python)
        nb.require(costs == report["ledger_summary"], "Post-report call ledger changed")
        nb.require(costs["attempted_calls"] <= setting["max_call_attempts"], "Worker call cap exceeded")
        nb.require(len(success) == sum(r["event"] == "CALL_SUCCESS" for r in events), "Repeated journal success ID")
        nb.require(set(success) == {k for k, r in native.items() if r["status"] == "SUCCESS"}, "Native/journal successful calls differ")
        if costs["failed_wrapper_calls"] or costs["pending_call_reservations"]:
            findings.append({"worker": index, "kind": "failed_or_pending_calls_require_manual_cost_review", "counts": costs})
        worker_unit_seconds, references = [], set()
        for unit in assigned:
            ud = folder/"units"/unit["unit_id"]; done_path = ud/"UNIT_COMPLETE.json"; done = nb.read_json(done_path)
            nb.require(nb.file_sha(done_path) == listed[unit["unit_id"]] and done["unit"] == unit and done["plan_sha256"] == plan_sha256
                       and done["executor_protocol_hash"] == op.protocol_hash and done["labels_read"] is False and done["scored"] is False,
                       "Unit complete receipt binding mismatch")
            nb.require(set(done["graphs"]) == {r["name"] for r in menu}, "Missing/extra graph checkpoint")
            prior = {}; unit_logical = unit_physical = 0; replay_checks = []
            for request in menu:
                path = ud/(request["name"]+".json"); rec = nb.read_json(path)
                nb.require(nb.file_sha(path) == done["graphs"][request["name"]], "Graph bytes changed")
                nb.require(rec["schema"] == "mab2_cycle2_graph_checkpoint_v1" and rec["unit_id"] == unit["unit_id"]
                           and rec["request"] == request and rec["plan_sha256"] == plan_sha256 and rec["labels_read"] is False
                           and rec["quality"] is None and nb.digest(rec["execution"]) == rec["execution_value_sha256"], "Graph checkpoint binding mismatch")
                value = rec["execution"]; result = nb.restore_execution(module, value)
                nb.require(result.question_id == unit["question_id"] and result.question_fingerprint == p["questions"]["payload_sha256"][unit["question_id"]]
                           and result.seed == unit["seed"] and list(result.mask) == request["mask"] and result.protocol_hash == op.protocol_hash, "Execution identity mismatch")
                nb.require(value["physical_cost"] == original_execution_cost(result, True, producer_python)
                           and value["logical_service_cost"] == original_execution_cost(result, False, producer_python), "Execution derived cost mismatch")
                names = {f"r{r}:w{w}" for r in (1, 2) for w in range(4)} | {"aggregate"}
                nb.require(set(result.nodes) == names, "Missing/extra graph nodes")
                base = prior.get(request.get("base")); closure = module.affected_closure(base.mask, result.mask) if base else names
                nb.require(result.closure == sorted(closure), "Affected closure mismatch")
                nb.validate_execution_journal(result, journal)
                for name, node in result.nodes.items():
                    generation = node.generation; call = native[generation.call_id]
                    cap = cfg.aggregate_tokens if name == "aggregate" else cfg.message_tokens
                    expected_seed = module.derive_seed(unit["question_id"], unit["seed"], name)
                    fp = module.digest({"question": result.question_fingerprint, "node": name, "messages": node.messages,
                                        "seed": expected_seed, "protocol": op.protocol_hash, "max_new_tokens": cap})
                    nb.require(node.node == name and node.seed == expected_seed and node.input_fingerprint == fp
                               and module.digest(node.integrity_value()) == node.output_digest, "Node integrity/prompt/seed mismatch")
                    nb.require(call["seed"] == expected_seed and call["prompt_sha256"] == nb.digest(node.messages)
                               and all(call[k] == getattr(generation, k) for k in ("input_tokens", "output_tokens", "gpu_seconds", "gpu_peak_allocated_bytes", "finish_reason")), "Native generation binding mismatch")
                    nb.require(all(call["generation_config"][k] == getattr(cfg, k) for k in ("do_sample", "temperature", "top_p"))
                               and call["generation_config"]["max_new_tokens"] == cap, "Native decoding config mismatch")
                    nb.require(generation.output_tokens <= cap and generation.input_tokens <= cfg.max_input_tokens, "Generation exceeds token cap")
                    if name not in closure:
                        nb.require(node.reused and node.source_execution_id == base.execution_id and node.integrity_value() == base.nodes[name].integrity_value(), "Illegal cache lineage")
                    else:
                        nb.require(not node.reused and node.source_execution_id is None, "Affected node illegally reused")
                        success_row = success[generation.call_id]
                        nb.require(success_row["unit_id"] == unit["unit_id"] and success_row["request_name"] == request["name"], "Fresh call context mismatch")
                        references.add(generation.call_id)
                        fmt = raw_format(generation.text)
                        nb.require(fmt == node.message_metadata["format_ok"], "Independent schema parser disagrees with executor")
                        all_formats["physical_outputs"] += 1
                        all_formats["malformed"] += int(not fmt)
                        all_formats["length_finish"] += int(generation.finish_reason == "length")
                        all_formats["message_trimmed"] += int(node.message_metadata["message_trimmed"])
                        if name == "aggregate":
                            all_formats["aggregate_outputs"] += 1
                            all_formats["aggregate_malformed"] += int(not fmt)
                            all_formats["aggregate_length_finish"] += int(generation.finish_reason == "length")
                nb.require(result.answer == result.nodes["aggregate"].parsed["answer"], "Aggregate answer serialization differs")
                if request.get("replay_of"):
                    equal = nb.execution_signature(result) == nb.execution_signature(prior[request["replay_of"]])
                    nb.require(equal, "Independent full replay differs from cached branch"); replay_checks.append(equal)
                prior[request["name"]] = result; unit_logical += result.cost(False)["calls"]; unit_physical += result.cost()["calls"]
            nb.require(done["logical_calls"] == unit_logical == 9*len(menu) and done["physical_calls_referenced"] == unit_physical == nb.estimated_calls(menu, module), "Unit cost totals mismatch")
            nb.require(done["replay_checks"] == replay_checks and all(replay_checks), "Unit replay receipt mismatch")
            attempts = list(ud.glob("attempt_*.START.json")); nb.require(attempts, "Unit has no attempt start")
            initial_time = min(nb.read_json(f)["started_unix"] for f in attempts)
            seconds = done["completed_unix"]-initial_time; nb.require(seconds >= 0, "Nonmonotone unit wall-clock receipt")
            worker_unit_seconds.append(seconds)
            total_logical += unit_logical; total_physical_references += unit_physical
            unit_rows.append({"unit_id": unit["unit_id"], "worker_index": index, "graphs": len(menu), "replays": len(replay_checks),
                              "physical_calls_referenced": unit_physical, "logical_calls": unit_logical, "elapsed_including_attempts_seconds": seconds,
                              "complete_sha256": listed[unit["unit_id"]]})
        unreferenced = set(success)-references
        if unreferenced: findings.append({"worker": index, "kind": "paid_successful_calls_outside_committed_graphs", "count": len(unreferenced)})
        remaining_calls = sum(nb.estimated_calls(nb.unit_menu(p, u), module) for u in p["units"] if u["worker_index"] == index and u["phase"] != "qualification")
        qual_calls = len(assigned)*nb.estimated_calls(menu, module)
        mean_rate = sum(worker_unit_seconds)/qual_calls
        max_rate = max(worker_unit_seconds)/nb.estimated_calls(menu, module)
        observed_load_and_overhead = max(0., used-sum(worker_unit_seconds))
        projection = remaining_calls*mean_rate+observed_load_and_overhead
        conservative = remaining_calls*max_rate+observed_load_and_overhead
        remaining_wall = setting["resident_wall_cap_seconds"]-used
        margins = {str(multiplier): {"additional_formal_seconds": remaining_calls*max_rate*multiplier+observed_load_and_overhead,
                                    "fits_remaining_cap": remaining_calls*max_rate*multiplier+observed_load_and_overhead <= remaining_wall}
                   for multiplier in (1.0, 1.25, 1.5)}
        workers.append({"worker_index": index, "resident_wall_seconds": used, "remaining_resident_seconds": remaining_wall,
                        "costs": costs, "formal_calls_remaining": remaining_calls, "qualification_calls_referenced": qual_calls,
                        "qualification_units": len(assigned), "observed_load_and_overhead_seconds": observed_load_and_overhead,
                        "qualification_unit_wall_seconds_including_fsync": worker_unit_seconds,
                        "steady_mean_seconds_per_fresh_call": mean_rate,
                        "steady_slowest_unit_seconds_per_fresh_call": max_rate,
                        "steady_slowest_unit_margins_plus_one_reload": margins,
                        "successful_native_calls_per_resident_second": costs["successful_native_or_journal_calls"]/used if used else None,
                        "committed_calls_per_unit_wall_second": 1/mean_rate if mean_rate else None,
                        "conservative_steady_committed_calls_per_second": 1/max_rate if max_rate else None,
                        "projected_formal_seconds_mean_rate_plus_one_reload": projection,
                        "projected_formal_seconds_max_unit_rate_plus_one_reload": conservative,
                        "projection_fits_remaining_mean": projection <= remaining_wall,
                        "projection_fits_remaining_max_unit": conservative <= remaining_wall,
                        "projection_scope": "two-unit empirical rate estimate, not guaranteed throughput or acceptance"})
    nb.require(sum(w["resident_wall_seconds"] for w in workers) <= p["budget"]["resident_wall_gpu_seconds"], "Global resident cap exceeded")
    nb.require(sum(w["costs"]["attempted_calls"] for w in workers) <= p["budget"]["max_call_attempts"], "Global attempt cap exceeded")
    return {"schema": "mab2_cycle2_qualification_evidence_audit_v1", "status": "EVIDENCE_CHECKS_COMPLETE_REVIEW_REQUIRED",
            "automatic_acceptance": False, "qualification_acceptance_written": False, "plan_sha256": plan_sha256,
            "runner_sha256": p["runner_sha256"], "executor_protocol_hash": op.protocol_hash,
            "cost_arithmetic": {"producer_python": producer_python, "node_order": "r1:w0..3,r2:w0..3,aggregate",
                                "comparison": "exact; reproduce producer float sum; no relaxed tolerance",
                                "python_sum_reference": "https://docs.python.org/3/library/functions.html#sum"},
            "episode_transform_binding_independently_checked": True,
            "runner_known_boundary": "Frozen runner checks transform set; this audit additionally fixes each episode-to-transform mapping",
            "labels_read": False, "S_scores_read": False, "question_file_read": False,
            "units": unit_rows, "worker_report_hashes": reports, "workers": workers, "format_counts": dict(all_formats),
            "total_projected_candidate_resident_hours_by_steady_margin": {
                key: sum(w["resident_wall_seconds"]+w["steady_slowest_unit_margins_plus_one_reload"][key]["additional_formal_seconds"] for w in workers)/3600
                for key in ("1.0", "1.25", "1.5")},
            "logical_calls": total_logical, "committed_physical_calls": total_physical_references,
            "findings_for_manual_review": findings,
            "acceptance_requires": ["root review of format/truncation and any failures", "remaining budget and full-coverage feasibility", "separate signed qualification receipt"],
            "no_effect_or_quality_evidence": True}


def forecast_progress(plan_path, plan_sha256, output_root, source_root, runner_path, *, now=None):
    """Partial read-only timing snapshot. It cannot accept/reject scientific quality."""
    nb = load_runner(runner_path); now = time.time() if now is None else now
    nb.require(nb.file_sha(plan_path) == plan_sha256, "Plan bytes mismatch")
    p = nb.read_json(plan_path); nb.require(nb.file_sha(runner_path) == p["runner_sha256"], "Runner bytes mismatch")
    module, _ = nb.load_operator(source_root, p["source_manifest"])
    out = []
    for setting in p["workers"]:
        folder = Path(output_root)/("worker_"+str(setting["index"]))
        resident = nb.read_jsonl(folder/"resident_wall.jsonl")
        starts = {r["attempt_id"]: r for r in resident if r["event"] == "START"}
        ends = {r["attempt_id"]: r for r in resident if r["event"] == "END"}
        nb.require(len(starts)+len(ends) == len(resident) and set(ends) <= set(starts), "Invalid resident progress ledger")
        nb.require(all(r.get("plan_sha256") == plan_sha256 for r in resident), "Foreign plan in resident ledger")
        used = sum(r["resident_wall_seconds"] for r in ends.values())
        open_ids = set(starts)-set(ends); nb.require(len(open_ids) <= 1, "Multiple unclosed resident attempts")
        for key in open_ids:
            nb.require(now >= starts[key]["started_unix"], "Snapshot precedes worker start")
            used += now-starts[key]["started_unix"]
        rows = nb.read_jsonl(folder/"physical_calls.jsonl")
        successes = [r for r in rows if r["status"] == "SUCCESS"]
        failures = [r for r in rows if r["status"] != "SUCCESS"]
        nb.require(len({r["call_id"] for r in rows}) == len(rows), "Repeated physical call ID")
        model_seconds = sum(r["elapsed_seconds"] for r in successes)
        if successes:
            observed_span = max(r["started_unix"]+r["elapsed_seconds"] for r in successes)-min(r["started_unix"] for r in successes)
        else: observed_span = 0.
        n = len(successes)
        formal_calls = sum(nb.estimated_calls(nb.unit_menu(p, u), module) for u in p["units"] if u["worker_index"] == setting["index"] and u["phase"] != "qualification")
        qual_calls = sum(nb.estimated_calls(nb.unit_menu(p, u), module) for u in p["units"] if u["worker_index"] == setting["index"] and u["phase"] == "qualification")
        # Residual qualification estimate is conservative after failures: successful
        # but uncommitted calls may need replay, so expose this limitation explicitly.
        remaining_qual_lower = max(0, qual_calls-n)
        resident_rate = n/used if used and n else None
        span_rate = n/observed_span if observed_span and n else None
        model_rate = n/model_seconds if model_seconds and n else None
        projection = (formal_calls+remaining_qual_lower)/resident_rate if resident_rate else None
        out.append({"worker_index": setting["index"], "successful_native_calls": n, "failed_native_calls": len(failures),
                    "resident_wall_seconds_snapshot": used, "open_resident_attempts": len(open_ids),
                    "native_model_calls_per_second": model_rate, "elapsed_generation_span_calls_per_second": span_rate,
                    "including_load_resident_calls_per_second": resident_rate,
                    "planned_formal_calls": formal_calls, "qualification_calls_remaining_lower_bound": remaining_qual_lower,
                    "projected_additional_wall_seconds_at_resident_rate": projection,
                    "projected_full_candidate_worker_seconds": used+projection if projection is not None else None,
                    "remaining_current_cap_seconds": setting["resident_wall_cap_seconds"]-used,
                    "fits_current_cap_at_observed_resident_rate": (used+projection <= setting["resident_wall_cap_seconds"]) if projection is not None else None,
                    "uncommitted_successes_may_need_reexecution": bool(failures)})
    complete = all(w["projected_full_candidate_worker_seconds"] is not None for w in out)
    return {"schema": "mab2_cycle2_qualification_progress_forecast_v1", "status": "TIMING_ONLY_NO_ACCEPTANCE",
            "plan_sha256": plan_sha256, "snapshot_unix": now, "workers": out,
            "projected_total_resident_gpu_hours": sum(w["projected_full_candidate_worker_seconds"] for w in out)/3600 if complete else None,
            "planned_formal_calls": sum(w["planned_formal_calls"] for w in out),
            "qualification_acceptance_written": False, "labels_read": False, "S_scores_read": False,
            "scientific_failure_inferred": False,
            "limits": ["Partial native model calls differ in prompt/length; rate can drift", "Includes initial load in resident-rate projection, therefore conservative when load amortizes", "No upper-confidence throughput guarantee", "Remote/local clocks must agree for open-attempt resident snapshot", "A resource amendment must be explicit and preserve prior costs"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output-root", "source-root", "runner"): ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--plan-sha256", required=True); ap.add_argument("--progress-only", action="store_true"); a = ap.parse_args()
    try:
        function = forecast_progress if a.progress_only else audit
        result = function(a.plan, a.plan_sha256, a.output_root, a.source_root, a.runner)
    except Exception as exc:
        result = {"status": "EVIDENCE_AUDIT_BLOCKED", "automatic_acceptance": False,
                  "error_type": type(exc).__name__, "error": str(exc), "labels_read": False, "S_scores_read": False}
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 2
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
