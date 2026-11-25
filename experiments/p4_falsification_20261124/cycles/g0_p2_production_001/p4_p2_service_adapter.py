"""G0-only adapter that routes the real P2 renderer/model calls through P4 broker.

This is a bounded development integration probe. It does not inspect gold and
does not run any candidate comparison. P2's prompt renderer is imported from
the locked production runner; all model calls in history, online, and
post-decision full-graph audit stages are observed by actual model hooks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
P2_DIR = WORKSPACE / "experiments" / "p2_multihop_20261124"
if str(P2_DIR) not in sys.path:
    sys.path.insert(0, str(P2_DIR))
from p2_service_runner import encode, prompts, summary
from p2_logic import terminal_answer

from p4_online_runtime import (
    CallEvent, CallResult, OnlineBroker, RuntimeNode, event_ledger_status,
    input_key, run_policy_step,
)


def distribution(value: str) -> np.ndarray:
    arr = np.asarray(json.loads(value), dtype=np.float32)
    if arr.shape != (4,) or not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError("invalid serialized P2 probability vector")
    if not np.isclose(float(arr.sum()), 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("P2 probability vector does not sum to one")
    return arr


def exact_terminal_weights(outputs: Mapping[str, str]) -> tuple[dict[str, int], int]:
    source = distribution(outputs["source"])
    relays = [distribution(outputs[f"relay_{i}"]) for i in range(4)]
    fractions: dict[str, Fraction] = {}
    for i in range(4):
        for j in range(4):
            fractions[f"terminal_{i}_{j}"] = Fraction.from_float(float(source[i])) * Fraction.from_float(float(relays[i][j]))
    denominator = reduce(lambda a, b: (a // gcd(a, b)) * b, (value.denominator for value in fractions.values()), 1)
    weights = {node: value.numerator * (denominator // value.denominator) for node, value in fractions.items()}
    if not any(weight > 0 for weight in weights.values()):
        raise ValueError("all P2 terminal masses are zero")
    return weights, denominator


def _terminal_name(i: int, j: int) -> str:
    return f"terminal_{i}_{j}"


def _event_dict(event: CallEvent) -> dict[str, Any]:
    return {field: getattr(event, field) for field in (
        "call_id", "node_id", "input_digest", "input_tokens", "output_digest",
        "paid_cost", "model_version", "tool_version", "raw_output_digest", "parse_status",
    )}


class P2TorchHookBackend:
    """Use P2's exact logits-to-probability/action semantics with actual hooks."""
    def __init__(self, model, device, source_ids, label_ids, model_version, tool_version, cap_seconds=180):
        self.model = model
        self.device = device
        self.source_ids = list(source_ids)
        self.label_ids = [list(value) for value in label_ids]
        self.model_version = model_version
        self.tool_version = tool_version
        self.started = time.perf_counter()
        self.cap_seconds = cap_seconds
        self.run_nonce = uuid.uuid4().hex[:12]
        self.active = None
        self.actual_ids = None
        self.pre_count = 0
        self.post_count = 0
        self.pre_handle = model.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self.post_handle = model.register_forward_hook(self._post_hook, with_kwargs=True)

    def close(self):
        self.pre_handle.remove()
        self.post_handle.remove()

    def _pre_hook(self, _module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None:
            raise RuntimeError("P2 production hook did not observe input_ids")
        actual = ids.detach().to("cpu").numpy().astype(np.int64, copy=False)
        if actual.ndim != 2 or actual.shape[0] != 1:
            raise RuntimeError("G0 adapter requires one unpadded input per model call")
        self.actual_ids = [int(token) for token in actual[0].tolist()]
        self.pre_count += 1

    def _parse(self, node_id: str, logits) -> str:
        if node_id.startswith("terminal_"):
            if any(len(ids) != 1 for ids in self.label_ids):
                raise ValueError("locked P2 action labels must each be one token")
            selected = [logits[ids[0]] for ids in self.label_ids]
            return chr(65 + int(torch.stack(selected).argmax().item()))
        token_ids = torch.tensor(self.source_ids, dtype=torch.long)
        probabilities = logits[token_ids].softmax(0).numpy().astype(np.float32, copy=False)
        return json.dumps([float(value) for value in probabilities], separators=(",", ":"), allow_nan=False)

    def _post_hook(self, _module, _args, _kwargs, output):
        self.post_count += 1
        if self.active is None or self.actual_ids is None:
            raise RuntimeError("unmatched P2 model hook event")
        node_id, expected_ids, model_version, tool_version, paid_cost, sink = self.active
        logits = output.logits[0, -1].detach().float().cpu().contiguous()
        raw_digest = hashlib.sha256(logits.numpy().tobytes()).hexdigest()
        parse_status = "OK"
        try:
            text = self._parse(node_id, logits)
        except Exception:
            text = ""
            parse_status = "PARSE_ERROR"
        event = CallEvent(
            call_id=f"hook-{self.run_nonce}-{self.pre_count}-{hashlib.sha256((node_id + raw_digest).encode()).hexdigest()[:16]}",
            node_id=node_id,
            input_digest=input_key(self.actual_ids, self.model_version, self.tool_version),
            input_tokens=len(self.actual_ids), output_digest=hashlib.sha256(text.encode()).hexdigest() if parse_status == "OK" else "",
            paid_cost=1 + len(self.actual_ids), model_version=self.model_version, tool_version=self.tool_version,
            raw_output_digest=raw_digest, parse_status=parse_status,
        )
        sink(event)
        if (self.actual_ids != list(expected_ids) or model_version != self.model_version
                or tool_version != self.tool_version or paid_cost != event.paid_cost):
            self.result = RuntimeError("actual P2 hook input/cost differs from broker request")
        elif parse_status != "OK":
            self.result = RuntimeError("P2 hook parser failed")
        else:
            self.result = CallResult(text, event)

    def __call__(self, node_id, ids, model_version, tool_version, paid_cost, hook_sink):
        if time.perf_counter() - self.started > self.cap_seconds:
            raise TimeoutError("P2 production G0 wall-time cap reached")
        if self.active is not None:
            raise RuntimeError("nested P2 model calls are forbidden")
        self.active = (node_id, list(ids), model_version, tool_version, paid_cost, hook_sink)
        self.actual_ids = None
        self.result = None
        pre_before, post_before = self.pre_count, self.post_count
        input_ids = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        mask = torch.ones_like(input_ids)
        positions = mask.long().cumsum(1) - 1
        try:
            with torch.inference_mode():
                self.model(input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=False)
        except Exception:
            if self.pre_count == pre_before + 1 and self.post_count == post_before:
                actual = self.actual_ids or list(ids)
                hook_sink(CallEvent(
                    call_id=f"hook-error-{self.pre_count}-{node_id}", node_id=node_id,
                    input_digest=input_key(actual, self.model_version, self.tool_version), input_tokens=len(actual),
                    output_digest="", paid_cost=1 + len(actual), model_version=self.model_version,
                    tool_version=self.tool_version, raw_output_digest="", parse_status="FORWARD_ERROR",
                ))
            raise
        finally:
            self.active = None
        if self.result is None:
            raise RuntimeError("P2 model forward did not produce a result")
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def run_row(row: dict[str, Any], model, tokenizer, device: str, model_version: str, tool_version: str, worker: Path, cap_seconds: int = 180, started_at: float | None = None) -> dict[str, Any]:
    started = time.perf_counter() if started_at is None else started_at
    old_facts = [row["second_fact"]] * 4
    old_interfaces = [""] * 4
    new_facts = row.get("second_facts_new", [row.get("second_fact_new") or row["second_fact"]] * 4)
    new_interfaces = row.get("new_relay_interface", [""] * 4)
    if len(new_facts) != 4 or len(new_interfaces) != 4:
        raise ValueError("locked P2 row must contain four relay fact/interface slots")
    source_ids = [tokenizer.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(4)]
    label_ids = [tokenizer.encode(chr(65 + i), add_special_tokens=False) for i in range(8)]
    if any(len(value) != 1 for value in label_ids):
        raise ValueError("P2 label output is not one token under the locked tokenizer")
    backend = P2TorchHookBackend(model, device, source_ids, label_ids, model_version, tool_version, cap_seconds)

    history_events: list[CallEvent] = []
    old_input_ids: dict[str, list[int]] = {}
    old_outputs: dict[str, str] = {}

    def history_call(node_id: str, prompt: str) -> str:
        ids = encode(tokenizer, prompt)
        old_input_ids[node_id] = ids
        cost = 1 + len(ids)
        return backend(node_id, ids, model_version, tool_version, cost, history_events.append).output

    p_old = prompts(row, "pending", ["pending"] * 4, old_facts, old_interfaces)
    old_outputs["source"] = history_call("source", p_old["source"])
    old_source_summary = summary(distribution(old_outputs["source"]))
    p_old_relay = prompts(row, old_source_summary, ["pending"] * 4, old_facts, old_interfaces)
    for i in range(4):
        old_outputs[f"relay_{i}"] = history_call(f"relay_{i}", p_old_relay["relay"][i])
    old_relay_summaries = [summary(distribution(old_outputs[f"relay_{i}"])) for i in range(4)]
    p_old_terminal = prompts(row, old_source_summary, old_relay_summaries, old_facts, old_interfaces)
    for i in range(4):
        for j in range(4):
            node_id = _terminal_name(i, j)
            old_outputs[node_id] = history_call(node_id, p_old_terminal["terminal"][(i, j)])
    if len(history_events) != 21:
        raise RuntimeError("P2 production history must contain 1 source + 4 relay + 16 terminal calls")

    def ids(text: str) -> list[int]:
        return encode(tokenizer, text)

    nodes: list[RuntimeNode] = []
    p_new_source = prompts(row, "pending", ["pending"] * 4, new_facts, new_interfaces)
    nodes.append(RuntimeNode(
        "source", (), lambda _parents: ids(p_new_source["source"]),
        input_key(old_input_ids["source"], model_version, tool_version), old_outputs["source"], model_version, tool_version,
    ))
    for i in range(4):
        def render_relay(parents: Mapping[str, str], i=i) -> list[int]:
            p = prompts(row, summary(distribution(parents["source"])), ["pending"] * 4, new_facts, new_interfaces)
            return ids(p["relay"][i])
        nodes.append(RuntimeNode(
            f"relay_{i}", ("source",), render_relay,
            input_key(old_input_ids[f"relay_{i}"], model_version, tool_version), old_outputs[f"relay_{i}"], model_version, tool_version,
        ))
    for i in range(4):
        for j in range(4):
            parent_ids = ("source", f"relay_{i}") if i == j else ("source", f"relay_{i}", f"relay_{j}")
            def render_terminal(parents: Mapping[str, str], i=i, j=j) -> list[int]:
                relay_summaries = ["pending"] * 4
                relay_summaries[i] = summary(distribution(parents[f"relay_{i}"]))
                relay_summaries[j] = summary(distribution(parents[f"relay_{j}"]))
                p = prompts(row, summary(distribution(parents["source"])), relay_summaries, new_facts, new_interfaces)
                return ids(p["terminal"][(i, j)])
            node_id = _terminal_name(i, j)
            nodes.append(RuntimeNode(
                node_id, parent_ids, render_terminal,
                input_key(old_input_ids[node_id], model_version, tool_version), old_outputs[node_id],
                model_version, tool_version, terminal=True,
            ))
    terminal_ids = [_terminal_name(i, j) for i in range(4) for j in range(4)]
    # Provisional positive masses retain the whole DAG while the actual P2
    # source/relay outputs are resolved. They are replaced before any terminal
    # feedback is queried or exposed to the policy.
    broker = OnlineBroker(tuple(chr(65 + i) for i in range(8)), nodes, {node_id: 1 for node_id in terminal_ids}, 10**9, backend)
    initial_known = set(broker.outputs)
    upstream_calls = []
    while True:
        legal = broker.public_state()["legal_actions"]
        ready_upstream = [node for node in legal if node == "source" or node.startswith("relay_")]
        if not ready_upstream:
            break
        for node_id in ready_upstream:
            obs = broker.query(node_id)
            upstream_calls.append({"node_id": node_id, "call_id": obs.call_id, "input_digest": obs.input_digest, "paid_cost": obs.paid_cost})
        if time.perf_counter() - started > cap_seconds:
            raise TimeoutError("P2 adapter exceeded locked wall-time cap during upstream validation")
    if any(node_id not in broker.outputs for node_id in ("source", *(f"relay_{i}" for i in range(4)))):
        raise RuntimeError("P2 source/relay stage did not resolve")
    terminal_weights, weight_scale = exact_terminal_weights(broker.outputs)
    broker.set_terminal_weights(terminal_weights)
    history_input_digest = {event.node_id: event.input_digest for event in history_events}
    pre_policy_cached_terminals = {node_id for node_id in broker.outputs if node_id.startswith("terminal_")}
    current_terminal_digest: dict[str, str] = {}
    for node_id in terminal_ids:
        node = broker.nodes[node_id]
        parent_values = {parent: broker.outputs[parent] for parent in node.parents}
        rendered_ids = list(node.render(parent_values))
        current_terminal_digest[node_id] = input_key(rendered_ids, model_version, tool_version)
    terminal_cache_equivalence = {
        node_id: (current_terminal_digest[node_id] == history_input_digest[node_id]) == (node_id in pre_policy_cached_terminals)
        for node_id in terminal_ids if terminal_weights[node_id] > 0
    }
    relay_input_changed = {}
    for i in range(4):
        node = broker.nodes[f"relay_{i}"]
        rendered_ids = list(node.render({"source": broker.outputs["source"]}))
        relay_input_changed[i] = input_key(rendered_ids, model_version, tool_version) != history_input_digest[f"relay_{i}"]
    second_parent_invalidation = {
        j: any(
            i != j and relay_input_changed[j]
            and current_terminal_digest[_terminal_name(i, j)] != history_input_digest[_terminal_name(i, j)]
            for i in range(4)
        )
        for j in range(4)
    }

    trace = []
    for _ in range(40):
        if time.perf_counter() - started > cap_seconds:
            raise TimeoutError("P2 adapter exceeded locked wall-time cap during policy decisions")
        action = run_policy_step(worker, broker.public_state())
        decision = broker.accept_policy_action(action)
        trace.append({"action": {key: value for key, value in action.items() if key != "planner_wall_seconds"}, "decision": decision})
        if decision["status"] != "RUNNING":
            break
    else:
        raise RuntimeError("P2 policy exceeded the bounded action-step count")
    online_status = broker.status
    online_action = trace[-1]["decision"].get("action") if trace else None
    online_event_ids_at_stop = {event.call_id for event in broker.events}
    policy_state_before_target = broker.public_state()
    policy_state_digest_before_target = hashlib.sha256(
        json.dumps(policy_state_before_target, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()

    # Full-new-graph target audit occurs only after the online policy has
    # stopped. It is recorded separately and never returned to the policy.
    target_events: list[CallEvent] = []
    target_labels: dict[tuple[int, int], int] = {}
    for i in range(4):
        for j in range(4):
            node_id = _terminal_name(i, j)
            if terminal_weights[node_id] == 0:
                continue
            if node_id in broker.outputs:
                label = broker.outputs[node_id]
            else:
                node = broker.nodes[node_id]
                parents = {parent: broker.outputs[parent] for parent in node.parents}
                current_ids = list(node.render(parents))
                result = backend(node_id, current_ids, model_version, tool_version, 1 + len(current_ids), target_events.append)
                label = result.output
            if label not in tuple(chr(65 + k) for k in range(8)):
                raise RuntimeError("full-graph audit encountered an invalid action label")
            target_labels[(i, j)] = ord(label) - 65
    source = distribution(broker.outputs["source"])
    relay = [distribution(broker.outputs[f"relay_{i}"]) for i in range(4)]
    target_index = terminal_answer(source, relay, target_labels)
    target_action = chr(65 + target_index)
    policy_state_after_target = broker.public_state()
    policy_state_digest_after_target = hashlib.sha256(
        json.dumps(policy_state_after_target, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()

    history_cost = sum(event.paid_cost for event in history_events)
    online_cost = sum(event.paid_cost for event in broker.events)
    target_cost = sum(event.paid_cost for event in target_events)
    history_ledger = event_ledger_status(history_events, history_cost, len(history_events))
    online_ledger = event_ledger_status(broker.events, broker.spent, len(broker.events))
    target_ledger = event_ledger_status(target_events, target_cost, len(target_events))
    history_keys = {event.node_id: event.input_digest for event in history_events}
    online_nodes = [event.node_id for event in broker.events]
    history_event_by_node = {event.node_id: event for event in history_events}
    comparisons = {
        "p2_renderer_history_inputs_match_actual_hook_ids": len(history_event_by_node) == 21 and all(
            history_event_by_node[node_id].input_digest == input_key(old_input_ids[node_id], model_version, tool_version)
            for node_id in old_input_ids
        ),
        "history_has_exact_21_call_shape": len(history_events) == 21,
        "unchanged_source_cached": "source" in initial_known and "source" not in online_nodes,
        "every_positive_terminal_cache_hit_matches_complete_actual_prompt_ids": all(terminal_cache_equivalence.values()),
        "actual_second_parent_changes_invalidate_terminal_cache": all(
            second_parent_invalidation[j] for j in range(4) if relay_input_changed[j]
        ) and any(relay_input_changed.values()),
        "all_new_relay_prompts_are_checked_against_exact_old_input": all(
            (node_id in online_nodes) == (input_key(list(broker.nodes[node_id].render({parent: broker.outputs[parent] for parent in broker.nodes[node_id].parents})), model_version, tool_version) != history_keys[node_id])
            for node_id in (f"relay_{i}" for i in range(4))
        ),
        "target_audit_happened_after_stop_without_changing_policy_state": (
            online_status in {"FULL_RESOLVED", "CERTIFIED"}
            and bool(trace)
            and trace[-1]["action"].get("type") == "stop"
            and len(online_event_ids_at_stop) == len(broker.events)
            and not (online_event_ids_at_stop & {event.call_id for event in target_events})
            and policy_state_digest_before_target == policy_state_digest_after_target
        ),
        "online_action_matches_post_stop_full_new_graph_action": online_action == target_action,
        "post_edit_ledger_valid": online_ledger[0] == "VALID_DEVELOPMENT_LEDGER",
        "history_ledger_valid": history_ledger[0] == "VALID_DEVELOPMENT_LEDGER",
        "target_audit_ledger_valid": target_ledger[0] == "VALID_DEVELOPMENT_LEDGER",
        "hook_counts_match_all_actual_calls": backend.pre_count == backend.post_count == len(history_events) + len(broker.events) + len(target_events),
    }
    # This adapter has no deployable predictor or hidden-generator mutation
    # probe. T13 therefore stays open even when the broker and ledger checks
    # below pass.
    status = "PASS_T01_T02_T03_T04_T05_T07_T11_BOUNDED_DEV" if all(comparisons.values()) else "FAIL_BOUNDED_DEV"
    backend.close()
    return {
        "schema": "mab2_p4_p2_production_adapter_result_v1",
        "classification": "BOUNDED_G0_DEVELOPMENT_ONLY_NO_GOLD",
        "status": status,
        "row_key_sha256": hashlib.sha256(row["key"].encode()).hexdigest(),
        "p2_runner_sha256": hashlib.sha256((P2_DIR / "p2_service_runner.py").read_bytes()).hexdigest(),
        "p2_logic_sha256": hashlib.sha256((P2_DIR / "p2_logic.py").read_bytes()).hexdigest(),
        "gold_or_target_loaded_before_online_stop": False,
        "target_audit_runs_after_online_stop": True,
        "post_edit_action": online_action,
        "post_edit_status": online_status,
        "policy_state_digest_before_target_audit": policy_state_digest_before_target,
        "policy_state_digest_after_target_audit": policy_state_digest_after_target,
        "full_new_graph_audit_action": target_action,
        "terminal_weight_scale_denominator": weight_scale,
        "terminal_weight_positive_count": sum(weight > 0 for weight in terminal_weights.values()),
        "initial_exact_cache_nodes": sorted(initial_known),
        "pre_policy_cached_terminal_nodes": sorted(pre_policy_cached_terminals),
        "current_terminal_input_digests": current_terminal_digest,
        "terminal_cache_equivalence": terminal_cache_equivalence,
        "relay_input_changed": relay_input_changed,
        "second_parent_terminal_invalidation": second_parent_invalidation,
        "upstream_broker_calls": upstream_calls,
        "query_trace": trace,
        "actual_call_counts": {"history": len(history_events), "online": len(broker.events), "post_stop_audit": len(target_events)},
        "actual_token_counts": {
            "history": sum(event.input_tokens for event in history_events),
            "online": sum(event.input_tokens for event in broker.events),
            "post_stop_audit": sum(event.input_tokens for event in target_events),
        },
        "actual_cost_units": {"history": history_cost, "online": online_cost, "post_stop_audit": target_cost},
        "ledger_checks": {"history": history_ledger, "online": online_ledger, "post_stop_audit": target_ledger},
        "hook_calls": {"pre": backend.pre_count, "post": backend.post_count},
        "comparisons": comparisons,
        "wall_seconds_including_history_and_policy": time.perf_counter() - started,
        "disclosure": "Uses production P2 prompts and Qwen model on one exposed development row. No gold is loaded. The full-new-graph audit runs only after online stop and is isolated from the policy's PublicState. This is G0 integration evidence only, not a candidate comparison, quality result, external blind audit, or G0 closure.",
        "event_rows": {
            "history": [_event_dict(event) for event in history_events],
            "online": [_event_dict(event) for event in broker.events],
            "post_stop_audit": [_event_dict(event) for event in target_events],
        },
    }


def execution_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    def normalized_events(stage: str) -> list[dict[str, Any]]:
        return [
            {key: event[key] for key in ("node_id", "input_digest", "input_tokens", "output_digest", "paid_cost", "model_version", "tool_version", "raw_output_digest", "parse_status")}
            for event in result["event_rows"][stage]
        ]
    trace = []
    for row in result["query_trace"]:
        decision = dict(row["decision"])
        observation = decision.get("observation")
        if isinstance(observation, dict):
            decision["observation"] = {key: observation[key] for key in ("node_id", "input_digest", "paid_cost")}
        trace.append({"action": row["action"], "decision": decision})
    return {
        "post_edit_action": result["post_edit_action"],
        "post_edit_status": result["post_edit_status"],
        "full_new_graph_audit_action": result["full_new_graph_audit_action"],
        "initial_exact_cache_nodes": result["initial_exact_cache_nodes"],
        "current_terminal_input_digests": result["current_terminal_input_digests"],
        "terminal_weights": [result["terminal_weight_scale_denominator"], result["terminal_weight_positive_count"]],
        "query_trace": trace,
        "history_events": normalized_events("history"),
        "online_events": normalized_events("online"),
        "post_stop_audit_events": normalized_events("post_stop_audit"),
    }


def compare_repeat_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(runs) < 2:
        return {"status": "NOT_RUN", "repeat_count": len(runs), "semantic_match": False}
    reference = execution_signature(runs[0])
    matches = [execution_signature(result) == reference for result in runs[1:]]
    return {"status": "PASS" if all(matches) else "FAIL", "repeat_count": len(runs), "semantic_match_each": matches}


def combine_repeat_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one locked repeat run is required")
    result = copy.deepcopy(runs[0])
    repeat_check = compare_repeat_runs(runs)
    if len(runs) > 1:
        for stage in ("history", "online", "post_stop_audit"):
            result["event_rows"][stage] = [event for run in runs for event in run["event_rows"][stage]]
            result["actual_call_counts"][stage] = sum(run["actual_call_counts"][stage] for run in runs)
            result["actual_token_counts"][stage] = sum(run["actual_token_counts"][stage] for run in runs)
            result["actual_cost_units"][stage] = sum(run["actual_cost_units"][stage] for run in runs)
            events = [CallEvent(**event) for run in runs for event in run["event_rows"][stage]]
            result["ledger_checks"][stage] = event_ledger_status(events, result["actual_cost_units"][stage], len(events))
        result["hook_calls"] = {
            "pre": sum(run["hook_calls"]["pre"] for run in runs),
            "post": sum(run["hook_calls"]["post"] for run in runs),
        }
        result["comparisons"]["repeat_execution_semantics_match"] = repeat_check["status"] == "PASS"
        result["repeat_runs"] = [{
            "status": run["status"], "post_edit_action": run["post_edit_action"],
            "full_new_graph_audit_action": run["full_new_graph_audit_action"],
            "actual_call_counts": run["actual_call_counts"], "actual_cost_units": run["actual_cost_units"],
        } for run in runs]
        result["repeat_check"] = repeat_check
        if repeat_check["status"] != "PASS":
            result["status"] = "FAIL_REPEAT_SEMANTICS_BOUNDED_DEV"
    else:
        result["repeat_check"] = repeat_check
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=HERE / "p4_policy_worker.py")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cap-seconds", type=int, default=180)
    args = parser.parse_args()
    started = time.perf_counter()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != lock["input_sha256"]:
        raise RuntimeError("P2 input does not match frozen integration lock")
    for filename, expected in lock["checkpoint_sha256"].items():
        if hashlib.sha256((args.model / filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError("model checkpoint does not match frozen lock: " + filename)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not bind the selected physical GPU")
    rowset = json.loads(args.input.read_text(encoding="utf-8"))
    if not 0 <= args.row < len(rowset):
        raise IndexError("row index outside exposed development data")
    row = rowset[args.row]
    if any("gold" in key.lower() or "target" in key.lower() for key in row):
        raise RuntimeError("target-bearing row refused")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True).to("cuda:0").eval()
    if time.perf_counter() - started >= args.cap_seconds:
        raise TimeoutError("model load exceeded locked P2 G0 wall-time cap")
    model_version = "qwen05b:" + lock["checkpoint_sha256"]["model.safetensors"]
    tool_version = "p2-renderer:" + lock["p2_files_sha256"]["p2_service_runner.py"] + ":" + lock["p2_files_sha256"]["p2_logic.py"] + ":" + lock["tokenizer_sha256"]
    if args.repeat_count < 1 or args.repeat_count > 3:
        raise ValueError("G0 repetition count must be in [1, 3]")
    runs = [run_row(row, model, tokenizer, "cuda:0", model_version, tool_version, args.worker, args.cap_seconds, started_at=started) for _ in range(args.repeat_count)]
    result = combine_repeat_runs(runs)
    result["wall_seconds_including_history_and_policy"] = time.perf_counter() - started
    result.update({"physical_gpu": args.gpu, "model_path": str(args.model), "input_sha256": lock["input_sha256"], "checkpoint_sha256": lock["checkpoint_sha256"], "row_index": args.row, "repeat_count": args.repeat_count, "gpu_hours_upper_bound": result["wall_seconds_including_history_and_policy"] / 3600.0, "time_cap_seconds": args.cap_seconds})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "event_rows"}, indent=2))
    return 0 if result["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
