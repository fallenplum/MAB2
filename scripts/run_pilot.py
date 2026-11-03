#!/usr/bin/env python3
"""Run a preregistered 2x2 edge-intervention pilot on real frozen local models.

No gold field can enter the executor. The complete global intervention plan is
committed before the first generation, and labels are opened only after every
assigned output has been frozen to disk. Small runs are marked PILOT_ONLY.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_credit.executor import Executor, GraphConfig, canonical, digest, logical_edges


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path.name}:{line_no}")
                rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> str:
    content = (canonical(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(content).hexdigest()


def append_json(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def default_contexts() -> list[dict[str, Any]]:
    edges = logical_edges()
    selected = {
        "full": set(edges),
        "chain": {(0, 1), (1, 2), (2, 3)},
        "star": {(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)},
        "fixed_random": {(0, 2), (1, 0), (1, 3), (2, 1), (3, 0), (3, 2)},
        "sequential_deletion": set(edges) - {edges[i] for i in (2, 5, 8, 11)},
    }
    return [{"name": name, "mask": [int(edge in members) for edge in edges]}
            for name, members in selected.items()]


def load_scorer(spec: str | None):
    if not spec:
        return None
    module_name, name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), name)


def source_manifest(project_root: Path) -> dict[str, dict[str, Any]]:
    """Bind the actual staged code/config bytes, with no machine-specific paths."""
    manifest = {}
    for folder, suffix in (("src", ".py"), ("scripts", ".py"), ("configs", ".json")):
        for path in sorted((project_root / folder).rglob("*")):
            if path.is_file() and path.suffix == suffix and "__pycache__" not in path.parts:
                content = path.read_bytes()
                manifest[path.relative_to(project_root).as_posix()] = {
                    "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    if "src/graph_credit/executor.py" not in manifest or "src/graph_credit/backend.py" not in manifest:
        raise ValueError("Incomplete project source tree for provenance locking")
    return manifest


def scorer_identity(spec: str | None) -> dict[str, Any]:
    scorer_spec = spec or "graph_credit.scoring:score_item"
    module_name, function_name = scorer_spec.split(":", 1)
    module = importlib.import_module(module_name)
    if not callable(getattr(module, function_name)) or not getattr(module, "__file__", None):
        raise ValueError("Frozen scorer must be a callable in an inspectable source module")
    content = Path(module.__file__).read_bytes()
    return {"callable": scorer_spec, "version": str(getattr(module, "VERSION", "unversioned")),
            "module_sha256": hashlib.sha256(content).hexdigest(), "module_bytes": len(content)}


def evaluate_message_records(records, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate (node_name, NodeRecord) pairs without labels or model calls.

    Only newly generated physical node outputs enter message-gate denominators.
    A reused node must not inflate compliance counts. Paired same-input repeats
    remain counted physical calls, not independent population-quality samples.
    Complete finite numeric JSON answers accepted by the normalizer are valid;
    repairing incomplete JSON must never turn a format failure into a success.
    """
    gate = {
        "worker_min_format_valid_rate": 0.9,
        "worker_max_length_finish_rate": 0.1,
        "worker_max_trimmed_rate": 0.1,
        "aggregator_min_format_valid_rate": 0.9,
        "aggregator_max_length_finish_rate": 0.1,
        "aggregator_max_trimmed_rate": 0.1,
        "require_each_role": True,
    }
    supplied = {} if config is None else config
    if not isinstance(supplied, dict) or set(supplied) - set(gate):
        raise ValueError("Unknown or invalid qualification_message_gate configuration")
    gate.update(supplied)
    if type(gate["require_each_role"]) is not bool:
        raise ValueError("qualification require_each_role must be boolean")
    for key, value in gate.items():
        if key != "require_each_role" and (type(value) not in (int, float) or not 0 <= value <= 1):
            raise ValueError(f"qualification threshold {key} must be finite and within [0,1]")

    def counts(records):
        output = {"generated": len(records), "format_valid": 0, "format_invalid": 0,
                  "length_finishes": 0, "trimmed": 0, "type_normalized": 0,
                  "missing_or_invalid_metadata": 0, "unknown_finish_reasons": 0}
        for node in records:
            metadata = node.message_metadata
            valid_metadata = (type(metadata.get("format_ok")) is bool
                              and type(metadata.get("message_trimmed")) is bool)
            output["missing_or_invalid_metadata"] += int(not valid_metadata)
            valid = valid_metadata and metadata["format_ok"]
            output["format_valid"] += int(valid)
            output["format_invalid"] += int(not valid)
            output["length_finishes"] += int(node.generation.finish_reason == "length")
            output["unknown_finish_reasons"] += int(node.generation.finish_reason not in ("length", "eos", "stop", "eos_token"))
            output["trimmed"] += int(metadata.get("message_trimmed", False) is True)
            output["type_normalized"] += int(metadata.get("type_normalized", False) is True)
        denominator = output["generated"]
        for numerator, rate in (("format_valid", "format_valid_rate"),
                                ("length_finishes", "length_finish_rate"), ("trimmed", "trimmed_rate")):
            output[rate] = output[numerator] / denominator if denominator else None
        return output

    def message_check(all_records):
        fresh = [(name, node) for name, node in all_records if not node.reused]
        worker_records = [node for name, node in fresh if name != "aggregate"]
        aggregator_records = [node for name, node in fresh if name == "aggregate"]
        by_role = {f"w{role}": counts([node for name, node in fresh if name.endswith(f":w{role}")])
                   for role in range(4)}
        workers, aggregator = counts(worker_records), counts(aggregator_records)
        reasons = []
        def assess(label, counter, prefix):
            if counter["generated"] == 0:
                reasons.append(f"{label}: no physical generated outputs")
                return
            if counter["missing_or_invalid_metadata"] or counter["unknown_finish_reasons"]:
                reasons.append(f"{label}: missing/invalid schema metadata or unknown finish reasons")
            if counter["format_valid_rate"] < gate[prefix + "_min_format_valid_rate"]:
                reasons.append(f"{label}: format_valid_rate={counter['format_valid_rate']:.6f} below frozen threshold")
            if counter["length_finish_rate"] > gate[prefix + "_max_length_finish_rate"]:
                reasons.append(f"{label}: length_finish_rate={counter['length_finish_rate']:.6f} above frozen threshold")
            if counter["trimmed_rate"] > gate[prefix + "_max_trimmed_rate"]:
                reasons.append(f"{label}: trimmed_rate={counter['trimmed_rate']:.6f} above frozen threshold")
        assess("workers", workers, "worker")
        assess("aggregator", aggregator, "aggregator")
        if gate["require_each_role"]:
            for role, counter in by_role.items():
                assess(role, counter, "worker")
        return {"status": "PASS" if not reasons else "FAIL", "reasons": reasons,
                "workers": workers, "aggregator": aggregator, "workers_by_role": by_role,
                "physical_generations": len(fresh), "reused_node_occurrences_excluded": len(all_records) - len(fresh),
                "all_node_occurrences": len(all_records), "gate_config": dict(gate),
                "format_definition": "Complete raw JSON schema; finite numeric answer canonicalization is accepted, malformed/incomplete JSON is invalid",
                "denominator": "physical_generations_only; cached node occurrences excluded; paired repeat calls are not independent samples"}

    return message_check(records)


def qualify(backend, question, config: GraphConfig, seed: int, run_dir: Path,
            message_gate_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Independent execution and message qualification on one label-free question."""
    checks = []
    all_round_records = []
    for rounds in (2, 3):
        executor = Executor(backend, dataclasses.replace(config, rounds=rounds))
        full_mask = [1] * 12
        toggled_mask = [0] + [1] * 11
        first = executor.execute(question, full_mask, seed)
        repeated = executor.execute(question, full_mask, seed)
        changed_full = executor.execute(question, toggled_mask, seed)
        changed_closure = executor.execute(question, toggled_mask, seed, base=first)
        empty = executor.execute(question, [0] * 12, seed)
        runs = {"full_first": first, "full_repeat": repeated,
                "switch_independent_full": changed_full, "switch_closure": changed_closure, "empty_graph": empty}
        for name, result in runs.items():
            write_json(run_dir / "qualification_executions" / f"r{rounds}_{name}.json", result.to_dict())
        repeated_equal = all(first.nodes[n].messages == repeated.nodes[n].messages
                             and first.nodes[n].generation.text == repeated.nodes[n].generation.text
                             and first.nodes[n].wire_message == repeated.nodes[n].wire_message for n in first.nodes)
        replay_equal = all(changed_full.nodes[n].messages == changed_closure.nodes[n].messages
                           and changed_full.nodes[n].generation.text == changed_closure.nodes[n].generation.text
                           and changed_full.nodes[n].wire_message == changed_closure.nodes[n].wire_message for n in changed_full.nodes)
        expected_calls = len(changed_closure.closure)
        empty_safe = all(node.received_message_tokens == 0 for name, node in empty.nodes.items() if name != "aggregate")
        format_counts = {name: {"nodes": len(result.nodes),
                                "format_ok": sum(n.message_metadata["format_ok"] for n in result.nodes.values()),
                                "trimmed": sum(n.message_metadata["message_trimmed"] for n in result.nodes.values()),
                                "length_finishes": sum(n.generation.finish_reason == "length" for n in result.nodes.values())}
                         for name, result in runs.items()}
        round_records = [(node_name, node) for execution in runs.values() for node_name, node in execution.nodes.items()]
        all_round_records.extend(round_records)
        messages = evaluate_message_records(round_records, message_gate_config)
        execution_passed = repeated_equal and replay_equal and expected_calls == changed_closure.cost()["calls"] and empty_safe
        passed = execution_passed and messages["status"] == "PASS"
        checks.append({"rounds": rounds, "same_input_seed_full_repeat_equal": repeated_equal,
                       "closure_independent_full_equal": replay_equal, "closure": changed_closure.closure,
                       "closure_expected_physical_calls": expected_calls,
                       "closure_actual_physical_calls": changed_closure.cost()["calls"],
                       "empty_graph_no_external_worker_messages": empty_safe,
                       "format_counts": format_counts, "physical_costs": {n: r.cost() for n, r in runs.items()},
                       "protocol_hash": executor.protocol_hash,
                       "execution_passed": execution_passed, "message_gate": messages,
                       "message_passed": messages["status"] == "PASS", "passed": passed})
    combined_messages = evaluate_message_records(all_round_records, message_gate_config)
    execution_status = "PASS" if all(c["execution_passed"] for c in checks) else "FAIL"
    message_status = "PASS" if all(c["message_passed"] for c in checks) and combined_messages["status"] == "PASS" else "FAIL"
    passed = execution_status == "PASS" and message_status == "PASS"
    result = {"schema": "real_hf_execution_qualification_v2", "status": "PASS" if passed else "FAIL",
              "question_id": question["id"], "seed": seed, "checks": checks,
              "execution_gate": {"status": execution_status},
              "message_gate": {**combined_messages, "status": message_status,
                               "all_rounds_must_pass": True,
                               "round_failures": [{"rounds": c["rounds"], "reasons": c["message_gate"]["reasons"]}
                                                  for c in checks if not c["message_passed"]]},
              "eligible_for_G1": passed,
              "label_access": "none; qualification uses no reference answers",
              "qualification_question_count": 1,
              "broader_sanity_recommendation": {"questions": 16, "labels_required": False,
                                                "status": "NOT_ESTABLISHED_BY_THIS_SINGLE_QUESTION_CHECK",
                                                "purpose": "check schema/length rates across held-out qualification questions before relying on population compliance"},
              "raw_generated_outputs_retained": True, "scientific_completion": False,
              "boundary": "Single-question execution and message-schema validity only; no population compliance estimate, graph-credit effect or paper-level benefit"}
    write_json(run_dir / "QUALIFICATION.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", default="0", help="Visible CUDA indices; each shard uses one index")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Qualification-only global question limit")
    parser.add_argument("--qualification-only", action="store_true", help="Real 2/3-round repeat/replay/empty-graph checks only")
    parser.add_argument("--scorer", default=None, help="module:function(answer, label, task=...)->quality")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid shard index/count")
    devices = [int(s.strip()) for s in args.devices.split(",")]
    if not devices or min(devices) < 0:
        parser.error("Explicit nonnegative CUDA indices required")
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    graph_values = dict(cfg.get("graph", cfg.get("executor", {})))
    allowed_fields = {f.name for f in dataclasses.fields(GraphConfig)}
    unknown = set(graph_values) - allowed_fields
    if unknown:
        raise ValueError(f"Unknown executor settings: {sorted(unknown)}")
    if "role_prompts" in graph_values:
        graph_values["role_prompts"] = tuple(graph_values["role_prompts"])
    graph_config = GraphConfig(**graph_values)
    seeds = cfg.get("seeds", [11, 23, 37, 53, 71])
    if not seeds or any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique explicit integers")
    pairs = cfg.get("edge_pairs", [[0, 1], [3, 6], [8, 11]])
    if any(len(p) != 2 or any(type(e) is not int or e not in range(12) for e in p) or p[0] == p[1] for p in pairs):
        raise ValueError("Each edge pair must specify two distinct logical edge indices")
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate intervention pairs")
    contexts = cfg.get("base_masks", default_contexts())
    if len({c["name"] for c in contexts}) != len(contexts):
        raise ValueError("Duplicate graph context names")
    from graph_credit.executor import validate_mask
    for context in contexts:
        validate_mask(context["mask"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", context["name"]) or context["name"] in {".", ".."}:
            raise ValueError("Graph context name must be a simple filename component")
    questions = read_jsonl(args.data)
    for question in questions:
        Executor.question_payload(question)
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Question limit must be positive")
        questions = questions[:args.limit]
    if not questions:
        raise ValueError("Empty question file")
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("pilot_%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in {".", ".."}:
        raise ValueError("Run ID must be a simple directory name")
    run_dir = args.root / "results" / "pilot" / run_id
    if args.num_shards > 1:
        run_dir /= f"shard_{args.shard_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "preflight_plan.json").exists():
        raise FileExistsError("Run directory is already initialized; use a new run ID (no silent overwrite/retry)")
    locked_sources = source_manifest(Path(__file__).resolve().parents[1])
    locked_scorer = scorer_identity(args.scorer)
    plan = {
        "schema": "graph_credit_pilot_plan_v1", "config": cfg,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "question_ids": [q["id"] for q in questions],
        "question_payload_sha256": [digest(Executor.question_payload(q)) for q in questions],
        "seeds": seeds, "edge_order": logical_edges(), "edge_pairs": pairs, "base_masks": contexts,
        "cell_execution_order": ["11", "01", "00", "10"],
        "sampling_pairing": "same_question_seed_node_across_all_graphs",
        "feedback_timing": "all_assigned_outputs_frozen_before_labels_opened",
        "num_shards": args.num_shards,
        "intended_shards": args.num_shards,
        "label_path_identifier": str(args.labels) if args.labels else None,
        "scientific_completion": False,
        "qualification_only": args.qualification_only or args.limit is not None or len(questions) < 200 or len(seeds) < 5,
        "scorer": args.scorer or "graph_credit.scoring:score_item",
        "scorer_identity": locked_scorer,
        "source_manifest": locked_sources,
        "source_manifest_sha256": digest(locked_sources),
    }
    plan_sha = write_json(run_dir / "preflight_plan.json", plan)
    assigned = [q for i, q in enumerate(questions) if i % args.num_shards == args.shard_index]
    runtime = {"run_id": run_id, "shard_index": args.shard_index, "num_shards": args.num_shards,
               "assigned_questions": len(assigned), "frozen_plan_sha256": plan_sha,
               "phase": "LOADING_MODEL", "scientific_completion": False}
    write_json(run_dir / "STATUS.json", runtime)
    backend = None
    raw_rows: list[dict[str, Any]] = []
    try:
        from graph_credit.backend import HFBackend
        backend = HFBackend(args.model, f"cuda:{devices[args.shard_index % len(devices)]}",
                            run_dir / "physical_calls.jsonl")
        executor = Executor(backend, graph_config)
        plan.update(protocol_hash=executor.protocol_hash, model_manifest=backend.identity["model_files"],
                    model_identity=backend.identity)
        plan_sha = write_json(run_dir / "frozen_plan.json", plan)
        runtime["frozen_plan_sha256"] = plan_sha
        write_json(run_dir / "run_manifest.json", {
            "schema": "graph_credit_pilot_run_v1", "protocol_hash": executor.protocol_hash,
            "model_identity": backend.identity, "runtime": backend.runtime,
            "executor_config": dataclasses.asdict(graph_config), "frozen_plan_sha256": plan_sha,
            "config_hash": digest(cfg), "scientific_completion": False,
            "source_manifest": locked_sources, "source_manifest_sha256": digest(locked_sources),
            "scorer_identity": locked_scorer,
        })
        runtime.update(phase="GENERATING", protocol_hash=executor.protocol_hash)
        write_json(run_dir / "STATUS.json", runtime)
        if args.qualification_only:
            if not assigned:
                raise ValueError("No assigned qualification question")
            qualification = qualify(backend, assigned[0], graph_config, seeds[0], run_dir,
                                    cfg.get("qualification_message_gate"))
            runtime.update(phase="QUALIFICATION_" + qualification["status"], generation_complete=True,
                           workflow_complete=False, scientific_completion=False,
                           successful_physical_calls=len(backend.physical_calls))
            write_json(run_dir / "STATUS.json", runtime)
            print(canonical(runtime), flush=True)
            return 0 if qualification["status"] == "PASS" else 2
        for question in assigned:
            qkey = hashlib.sha256(question["id"].encode()).hexdigest()[:20]
            for seed in seeds:
                base = None
                for context in contexts:
                    for edge_e, edge_f in pairs:
                        for cell in ("11", "01", "00", "10"):
                            graph = list(context["mask"])
                            graph[edge_e], graph[edge_f] = int(cell[0]), int(cell[1])
                            result = executor.execute(question, graph, seed, base=base)
                            base = result
                            filename = f"{qkey}_{seed}_{context['name']}_{edge_e}_{edge_f}_{cell}.json"
                            # Graph context names enter filenames only after validation.
                            if not re.fullmatch(r"[A-Za-z0-9_.-]+", context["name"]):
                                raise ValueError("Graph context name must be a simple filename component")
                            execution_path = Path("executions") / filename
                            execution_sha = write_json(run_dir / execution_path, result.to_dict())
                            row = {
                                "question_id": question["id"], "task": question.get("task", "gsm8k"),
                                "split": question.get("split", "dev"),
                                "context_id": context["name"] + "_" + digest(context["mask"])[:12],
                                "context_family": context["name"], "edge_e": edge_e, "edge_f": edge_f,
                                "seed": seed, "cell": cell, "mask_e": int(cell[0]), "mask_f": int(cell[1]),
                                "graph_mask": graph, "protocol_hash": executor.protocol_hash,
                                "frozen_plan_sha256": plan_sha, "quality": None,
                                "answer": result.answer, "execution_id": result.execution_id,
                                "service_output": result.nodes["aggregate"].generation.text,
                                "service_message_metadata": result.nodes["aggregate"].message_metadata,
                                "execution_path": str(execution_path).replace("\\", "/"),
                                "execution_sha256": execution_sha,
                                "physical_cost": result.cost(), "logical_service_cost": result.cost(False),
                            }
                            append_json(run_dir / "raw_executions.jsonl", row)
                            raw_rows.append(row)
                            runtime.update(phase="GENERATING", outputs_frozen=len(raw_rows),
                                           current_question_id=question["id"], current_seed=seed,
                                           successful_physical_calls=len(backend.physical_calls))
                            write_json(run_dir / "STATUS.json", runtime)
            print(canonical({"question_completed": question["id"], "outputs_frozen": len(raw_rows)}), flush=True)
        frozen_outputs_sha = hashlib.sha256((run_dir / "raw_executions.jsonl").read_bytes()).hexdigest() if raw_rows else None
        write_json(run_dir / "OUTPUTS_FROZEN.json", {"outputs": len(raw_rows), "raw_executions_sha256": frozen_outputs_sha,
                                                     "labels_opened": False, "frozen_plan_sha256": plan_sha})
        if args.labels:
            if source_manifest(Path(__file__).resolve().parents[1]) != locked_sources or scorer_identity(args.scorer) != locked_scorer:
                raise RuntimeError("Source/config/scorer bytes changed after plan freeze; refuse inconsistent scoring")
            runtime.update(phase="SCORING_FROZEN_OUTPUTS")
            write_json(run_dir / "STATUS.json", runtime)
            labels = read_jsonl(args.labels)  # First label access: strictly after OUTPUTS_FROZEN.
            if len({label["id"] for label in labels}) != len(labels):
                raise ValueError("Duplicate label IDs")
            label_by_id = {label["id"]: label for label in labels}
            scorer = load_scorer(args.scorer)
            for row in raw_rows:
                label = label_by_id.get(row["question_id"])
                if label is None:
                    raise ValueError(f"Missing purchased diagnostic label: {row['question_id']}")
                if scorer:
                    quality = float(scorer(row["service_output"], label, task=row["task"]))
                elif row["task"].lower() in {"gsm8k", "gsm"}:
                    from graph_credit.scoring import score_item
                    quality = score_item(row["service_output"], label, task="gsm8k")
                else:
                    raise ValueError("Non-numerical tasks require a frozen explicit --scorer")
                if not 0 <= quality <= 1:
                    raise ValueError("Scorer returned a quality outside [0,1]")
                append_json(run_dir / "observations.jsonl", {**row, "quality": quality})
            write_json(run_dir / "SCORING.json", {"labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                                                  "diagnostic_scoring_count": len(raw_rows), "natural_feedback_count": 0,
                                                  "scorer": args.scorer or "graph_credit.scoring:score_item",
                                                  "scorer_identity": locked_scorer,
                                                  "source_manifest_sha256": digest(locked_sources),
                                                  "raw_executions_sha256": frozen_outputs_sha,
                                                  "labels_used_to_change_service_outputs": False})
        runtime.update(phase="PILOT_ONLY" if plan["qualification_only"] else "GENERATION_COMPLETE_AWAITING_INDEPENDENT_GATE",
                       generation_complete=True, scored=bool(args.labels), workflow_complete=False,
                       outputs_frozen=len(raw_rows), scientific_completion=False)
        write_json(run_dir / "STATUS.json", runtime)
        print(canonical(runtime), flush=True)
        return 0
    except Exception as exc:
        runtime.update(phase="FAILED", error_type=type(exc).__name__, error=str(exc),
                       outputs_frozen=len(raw_rows), scientific_completion=False)
        write_json(run_dir / "STATUS.json", runtime)
        write_json(run_dir / "failure.json", {"error_type": type(exc).__name__, "error": str(exc),
                                               "traceback": traceback.format_exc(),
                                               "physical_calls_recorded": len(backend.physical_calls) if backend else 0,
                                               "silent_inference_retries": 0})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
