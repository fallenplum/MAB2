"""Twelve independent worker samples plus one fixed aggregator; staging only.

No inference is launched by importing this module. This is not the graph's
no-edge treatment: each sample gets the unchanged round-one prompt only.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import graph_credit.executor as frozen_executor
import routed_executor as frozen_router
from graph_credit.executor import (
    AGGREGATOR_PROMPT, ROLE_PROMPTS, Executor, Generation, GraphConfig,
    NodeRecord, canonical, derive_seed, digest, normalize_message,
)
from routed_executor import RoutedBackend


CONTROL_ID = "independent_4roles_x3samples_plus_aggregate_v2_cost_reconciliation_staging"
WORKER_ROUTE = ("qwen", "mistral", "qwen", "mistral")
SAMPLES_PER_ROLE = 3
WORKER_NODES = tuple(f"r{sample}:w{role}" for role in range(4) for sample in range(1, 4))
AGGREGATE_INPUT_PREFIX = "\n\nIndependent worker samples (fixed role-major, sample-minor order):\n"
AGGREGATE_FORMAT_SUFFIX = (
    '\n\nRequired final format: {"answer":"<final answer>","uncertainty":"","evidence":"<brief check>"}. '
    'Include exactly all 3 keys in that order. Never omit uncertainty, even when empty. '
    'Return only the complete single-line JSON object.'
)
DEPENDENCY_PINS = {
    "executor": "85acab26b3b713900e8099a5d77197af698525f57c2d9d07d1193c68b2c383b1",
    "router": "0273aebcfba546fd7112c82937b8f27447d0eee87243fbd9c00137d107938625",
}
PHYSICAL_EVENT_FIELDS = (
    "call_id", "status", "seed", "started_unix", "input_tokens", "output_tokens",
    "elapsed_seconds", "gpu_seconds", "gpu_peak_allocated_bytes", "finish_reason",
    "prompt_sha256", "error_type",
)
COST_METRICS = {
    "input_tokens": "input_tokens", "output_tokens": "output_tokens",
    "model_elapsed_seconds": "elapsed_seconds", "gpu_seconds": "gpu_seconds",
    "peak_allocated_bytes": "gpu_peak_allocated_bytes",
}


def _request_cost_evidence(slot):
    """Reconcile states of one backend request without choosing a preferred row.

    A call_id identifies states, not repeated generation. Missing/conflicting
    metric reports remain unknown even if one report happens to be complete.
    Observed ranges below describe evidence, not bounds on the unknown truth.
    """
    reports = []
    if slot.generation is not None:
        generation = slot.generation
        reports.append({"source": "returned_generation", "call_id": generation.call_id,
                        "seed": slot.seed, **{field: getattr(generation, field) for field in COST_METRICS.values()}})
    reports.extend({"source": f"physical_event_{i}", **copy.deepcopy(row)} for i, row in enumerate(slot.physical_events))
    if not reports:
        reports = [{"source": "no_returned_generation_or_physical_record"}]
    groups = {}
    for i, row in enumerate(reports):
        call_id = row.get("call_id")
        key = ("identified", call_id) if isinstance(call_id, str) and call_id.strip() else ("unidentified", i)
        groups.setdefault(key, []).append(row)
    request_issues = []
    if any(key[0] == "unidentified" for key in groups):
        request_issues.append("missing_call_id_cannot_establish_event_identity")
    if len(groups) > 1:
        request_issues.append("multiple_or_unidentified_call_id_groups_for_one_backend_request")
    reconciled = []
    for key, rows in groups.items():
        issues = list(request_issues)
        if any(type(row.get("seed")) is not int for row in rows):
            issues.append("missing_or_invalid_seed_identity")
        elif any(row["seed"] != slot.seed for row in rows):
            issues.append("seed_identity_conflicts_with_request")
        for field in ("seed", "prompt_sha256", "started_unix"):
            present = [row[field] for row in rows if row.get(field) is not None]
            if present and any(value != present[0] for value in present[1:]):
                issues.append(f"conflicting_identity_field:{field}")
        metrics = {}
        for key_name, field in COST_METRICS.items():
            missing = sum(row.get(field) is None for row in rows)
            raw = [row[field] for row in rows if row.get(field) is not None]
            valid = [value for value in raw if type(value) in (int, float) and math.isfinite(value) and value >= 0]
            invalid = len(raw) - len(valid)
            conflict = bool(valid and any(value != valid[0] for value in valid[1:]))
            resolved = valid[0] if valid and not (issues or missing or invalid or conflict) else None
            metrics[key_name] = {"resolved": resolved, "reported_count": len(valid), "missing_count": missing,
                "invalid_count": invalid, "conflict": conflict,
                "observed_min": min(valid) if valid else None, "observed_max": max(valid) if valid else None,
                "observed_range_is_not_a_truth_bound": True}
        reconciled.append({"call_id": key[1] if key[0] == "identified" else None,
                           "record_sources": [row["source"] for row in rows],
                           "event_statuses": [row.get("status") for row in rows if row.get("status") is not None],
                           "identity_issues": issues, "metrics": metrics})
    return {"slot_index": slot.index, "node": slot.node, "model": slot.model,
            "physical_event_count": len(slot.physical_events), "groups": reconciled}


@dataclass
class SampleSlot:
    index: int
    node: str
    role: int | None
    sample: int | None
    model: str
    seed: int
    status: str = "NOT_ATTEMPTED"
    attempted_backend_request: bool = False
    messages: list[dict[str, str]] | None = None
    input_fingerprint: str | None = None
    received_message_tokens: int | None = None
    self_state_tokens: int = 0
    generation: Generation | None = None
    record: NodeRecord | None = None
    error_type: str | None = None
    route_events: list[dict[str, Any]] = field(default_factory=list)
    physical_events: list[dict[str, Any]] = field(default_factory=list)


def _success_cost(slots: list[SampleSlot]) -> dict[str, int | float]:
    generated = [s for s in slots if s.generation is not None]
    return {
        "calls": len(generated),
        "input_tokens": sum(s.generation.input_tokens for s in generated),
        "output_tokens": sum(s.generation.output_tokens for s in generated),
        "received_message_tokens": sum(s.received_message_tokens or 0 for s in generated),
        "self_state_tokens": 0,
        "model_elapsed_seconds": sum(s.generation.elapsed_seconds for s in generated),
        "gpu_seconds": sum(s.generation.gpu_seconds for s in generated),
        "peak_allocated_bytes": max((s.generation.gpu_peak_allocated_bytes for s in generated), default=0),
    }


@dataclass
class Independent12Result:
    execution_id: str
    question_id: str
    question_fingerprint: str
    seed: int
    protocol_hash: str
    slots: list[SampleSlot]
    status: str = "RUNNING"
    answer: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def nodes(self) -> dict[str, NodeRecord]:
        return {s.node: s.record for s in self.slots if s.record is not None}

    def cost(self) -> dict[str, Any]:
        """All attempted requests, preserving unknown failure metrics as null.

        With no cache, performed logical and physical request ledgers are equal.
        On abort, planned-but-unattempted slots are not billed as full service.
        CUDA event time is not a resident-GPU occupancy ledger.
        """
        attempted = [s for s in self.slots if s.attempted_backend_request]
        request_evidence = [_request_cost_evidence(slot) for slot in attempted]
        identities = {}
        for request in request_evidence:
            for group in request["groups"]:
                if group["call_id"] is not None:
                    identities.setdefault((request["model"], group["call_id"]), []).append(group)
        for groups in identities.values():
            if len(groups) > 1:
                for group in groups:
                    group["identity_issues"].append("call_id_reused_across_backend_requests")
                    for metric in group["metrics"].values():
                        metric["resolved"] = None
        values = {key: [group["metrics"][key]["resolved"] for request in request_evidence
                        for group in request["groups"]] for key in COST_METRICS}
        unknown_failure_requests = sum(slot.generation is None and not slot.physical_events for slot in attempted)
        totals, known = {}, {}
        for key, entries in values.items():
            present = [x for x in entries if x is not None]
            reduction = max(present, default=0) if key == "peak_allocated_bytes" else sum(present)
            known[key] = reduction
            totals[key] = None if any(x is None for x in entries) else reduction
        return {
            "calls": len(attempted), "call_count_scope": "attempted_backend_requests_including_rejections",
            "planned_calls": 13, "unattempted_calls": 13 - len(attempted),
            "returned_generations": sum(s.generation is not None for s in self.slots),
            "failed_requests_without_physical_record": unknown_failure_requests,
            **totals,
            "received_message_tokens": sum(s.received_message_tokens or 0 for s in attempted),
            "self_state_tokens": 0,
            "known_cost_components": known,
            "known_cost_component_scope": "Only unambiguous reconciled components; conflicting or incomplete groups are excluded",
            "known_cost_ranges": {key: {"lower": known[key], "upper": totals[key],
                "scope": "nonnegative subtotal lower bound; null upper means unresolved"} for key in totals},
            "request_cost_evidence": request_evidence,
            "cost_reconciliation": "same_call_id_states_once_if_complete_and_equal_else_unknown_v2",
            "event_status_is_not_generation_multiplicity": True,
            "returned_generation_cost": _success_cost(self.slots),
            "returned_generation_cost_scope": "Raw returned Generation reports, not reconciled totals when evidence conflicts",
            "cost_fields_complete": all(value is not None for value in totals.values()),
            "service_complete": self.status == "COMPLETE",
            "resident_GPU_hours": None,
        }

    def to_dict(self) -> dict[str, Any]:
        costs = self.cost()
        return {
            **dataclasses.asdict(self), "control_id": CONTROL_ID,
            "physical_cost": costs, "logical_service_cost": copy.deepcopy(costs),
            "cache_reuse_count": 0,
            "format_failures": sum(s.record is not None and not s.record.message_metadata["format_ok"] for s in self.slots),
            "scientific_admission": False,
        }


class Independent12ExecutionError(RuntimeError):
    def __init__(self, node: str, partial_result: Independent12Result):
        super().__init__(f"Independent12 execution aborted at {node}; inspect partial_result")
        self.partial_result = partial_result


class Independent12Executor(Executor):
    """CPU-safe constructor; execute invokes the explicitly supplied backend.

    r1/r2/r3 node spellings retain the frozen model router and seed streams.
    Here their r index means independent sample, never conversational round.
    There is no mask, base result, retry, selection, or cache argument.
    """

    def __init__(self, backend: RoutedBackend, config: GraphConfig | None = None):
        if not isinstance(backend, RoutedBackend):
            raise TypeError("Independent12Executor requires the frozen RoutedBackend")
        config = config or GraphConfig(rounds=3)
        if config.rounds != 3 or tuple(config.role_prompts) != ROLE_PROMPTS:
            raise ValueError("This contract requires three samples and the four frozen role prompts")
        if tuple(backend.worker_models) != WORKER_ROUTE or backend.aggregator_model != "mistral":
            raise ValueError("This contract requires workers [qwen,mistral,qwen,mistral] and Mistral aggregation")
        dependency_hashes = {
            "executor": hashlib.sha256(Path(frozen_executor.__file__).read_bytes()).hexdigest(),
            "router": hashlib.sha256(Path(frozen_router.__file__).read_bytes()).hexdigest(),
        }
        if dependency_hashes != DEPENDENCY_PINS:
            raise RuntimeError("Frozen executor/router source bytes do not match the qualified dependency pins")
        super().__init__(backend, config)
        self.protocol_identity = {
            "control_id": CONTROL_ID, "config": dataclasses.asdict(config),
            "backend": copy.deepcopy(backend.identity), "dependency_sha256": dependency_hashes,
            "control_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "samples_per_role": SAMPLES_PER_ROLE, "worker_node_order": list(WORKER_NODES),
            "sample_node_semantics": "r_is_independent_sample_with_no_previous_state",
            "seed_policy": "derive_seed(question_id,execution_seed,r{sample}:w{role}); unchanged aggregate node",
            "worker_prompt_policy": "exact_frozen_round1_messages_for_each_role_no_sample_marker",
            "aggregator_system_prompt": AGGREGATOR_PROMPT,
            "aggregator_input_prefix": AGGREGATE_INPUT_PREFIX,
            "aggregator_format_suffix": AGGREGATE_FORMAT_SUFFIX,
            "aggregation_slot_fields": ["role", "sample", "node", "message"],
            "failure_policy": "abort_no_retry_preserve_all_13_planned_slots_and_unknown_costs",
            "cost_reconciliation_policy": "same_call_id_states_once_if_complete_and_equal_else_unknown_v2",
            "format_failure_policy": "retain_paid_FORMAT_ERROR_wire_in_original_slot_and_continue",
            "cache_policy": "disabled", "serialization": "canonical_json_v1",
        }
        self.protocol_hash = digest(self.protocol_identity)

    def _worker_messages(self, payload: Mapping[str, Any], role: int) -> list[dict[str, str]]:
        # Calling the original round-one builder cannot read a predecessor.
        messages, received, self_state = Executor._messages(self, payload, f"r1:w{role}", (0,) * 12, {})
        assert received == self_state == 0
        return messages

    def _aggregate_messages(self, payload: Mapping[str, Any], slots: list[SampleSlot]):
        workers = slots[:12]
        if len(workers) != 12 or [s.node for s in workers] != list(WORKER_NODES) or any(s.record is None for s in workers):
            raise RuntimeError("All twelve fixed worker slots, including format errors, are required")
        inbound = [{"role": s.role, "sample": s.sample, "node": s.node, "message": s.record.wire_message} for s in workers]
        text = payload["question"]
        if payload.get("context"):
            text += "\n\nSupplied context:\n" + payload["context"]
        messages = [{"role": "system", "content": AGGREGATOR_PROMPT},
                    {"role": "user", "content": text + AGGREGATE_INPUT_PREFIX + canonical(inbound) + AGGREGATE_FORMAT_SUFFIX}]
        received = sum(len(self.backend.encode(item["message"])) for item in inbound)
        return messages, received

    def _execute_slot(self, payload, slot: SampleSlot, result: Independent12Result):
        self.backend.select_node(slot.node)
        underlying = self.backend.backends[slot.model]
        physical = getattr(underlying, "physical_calls", [])
        physical_start, route_start = len(physical), len(self.backend.route_events)
        try:
            if slot.node == "aggregate":
                slot.messages, slot.received_message_tokens = self._aggregate_messages(payload, result.slots)
            else:
                slot.messages = self._worker_messages(payload, slot.role)
                slot.received_message_tokens = 0
            cap = self.config.aggregate_tokens if slot.node == "aggregate" else self.config.message_tokens
            slot.input_fingerprint = digest({"question": result.question_fingerprint, "node": slot.node,
                "messages": slot.messages, "seed": slot.seed, "protocol": self.protocol_hash, "max_new_tokens": cap})
            slot.attempted_backend_request = True
            generation = self.backend.generate(slot.messages, slot.seed, cap, self.config)
            slot.generation = generation
            if generation.input_tokens > self.config.max_input_tokens:
                raise ValueError("Backend executed an over-limit input; backend must reject before generation")
            if generation.output_tokens > cap:
                raise ValueError("Backend violated the frozen output token bound")
            wire, parsed, metadata = normalize_message(generation.text, self.backend, cap)
            record = NodeRecord(slot.node, slot.input_fingerprint, "", slot.seed, slot.messages,
                                generation, wire, parsed, metadata, slot.received_message_tokens, 0, False, None)
            slot.record = dataclasses.replace(record, output_digest=digest(record.integrity_value()))
            slot.status = "SUCCESS" if metadata["format_ok"] else "FORMAT_ERROR"
        finally:
            slot.route_events = copy.deepcopy(self.backend.route_events[route_start:])
            slot.physical_events = [{key: value for key, value in row.items() if key in PHYSICAL_EVENT_FIELDS}
                                    for row in physical[physical_start:]]

    def execute(self, question: Mapping[str, Any], seed: int) -> Independent12Result:
        payload = self.question_payload(question)
        if digest(self.protocol_identity) != self.protocol_hash or self.protocol_identity["backend"] != self.backend.identity:
            raise RuntimeError("Frozen independent-control or backend identity was modified")
        if dataclasses.asdict(self.config) != self.protocol_identity["config"]:
            raise RuntimeError("Frozen configuration was modified")
        if tuple(self.backend.worker_models) != WORKER_ROUTE or self.backend.aggregator_model != "mistral":
            raise RuntimeError("Frozen model route was modified")
        slots = [SampleSlot(i, node, int(node[-1]), int(node[1]), self.backend.model_for_node(node),
                            derive_seed(payload["id"], seed, node)) for i, node in enumerate(WORKER_NODES)]
        slots.append(SampleSlot(12, "aggregate", None, None, "mistral", derive_seed(payload["id"], seed, "aggregate")))
        result = Independent12Result(uuid.uuid4().hex, payload["id"], digest(payload), seed, self.protocol_hash, slots)
        if not self.backend._execution_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent use of a routed backend is forbidden")
        started = time.perf_counter()
        try:
            for slot in slots:
                try:
                    self._execute_slot(payload, slot, result)
                except Exception as error:
                    slot.status = ("POSTPROCESS_ERROR" if slot.generation is not None else
                                   "BACKEND_ERROR" if slot.attempted_backend_request else "PREPARATION_ERROR")
                    slot.error_type = type(error).__name__
                    result.status = "FAILED"
                    raise Independent12ExecutionError(slot.node, result) from error
            result.status = "COMPLETE"
            result.answer = slots[-1].record.parsed["answer"]
            return result
        finally:
            result.elapsed_seconds = time.perf_counter() - started
            self.backend.clear_node()
            self.backend._execution_lock.release()
