"""Broker-controlled online runtime for P4 development integration tests.

The broker owns renderers, old transcript entries, live model calls and costs.
The policy receives a JSON PublicState in a separate subprocess. This is a
logical isolation harness, not an OS security boundary or external blind test.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

_CORE_DIR = Path(__file__).resolve().parent
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))
from p4_core import Certificate, strict_certificate


def input_key(input_ids: Sequence[int], model_version: str, tool_version: str = "none") -> str:
    if not model_version or not tool_version:
        raise ValueError("missing model/tool version must fail closed")
    if not input_ids or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in input_ids):
        raise ValueError("missing or invalid actual input_ids must fail closed")
    payload = json.dumps(
        {"input_ids": list(input_ids), "model_version": model_version, "tool_version": tool_version},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RuntimeNode:
    node_id: str
    parents: tuple[str, ...]
    render: Callable[[Mapping[str, str]], Sequence[int]]
    old_input_digest: str
    old_output: str
    model_version: str
    tool_version: str = "none"
    fixed_cost: int = 1
    terminal: bool = False


@dataclass(frozen=True)
class CallEvent:
    call_id: str
    node_id: str
    input_digest: str
    input_tokens: int
    output_digest: str
    paid_cost: int
    model_version: str
    tool_version: str
    raw_output_digest: str = ""
    parse_status: str = "OK"


@dataclass(frozen=True)
class CallResult:
    output: str
    event: CallEvent


@dataclass(frozen=True)
class Observation:
    node_id: str
    output: str
    input_digest: str
    paid_cost: int
    call_id: str


class OnlineBroker:
    """Exact effective-input cache closure and broker-owned call accounting."""

    def __init__(
        self,
        actions: Sequence[str],
        nodes: Sequence[RuntimeNode],
        terminal_weights: Mapping[str, int],
        budget: int,
        model_call: Callable[[str, Sequence[int], str, str, int, Callable[[CallEvent], None]], CallResult],
        predictions: Mapping[str, str] | None = None,
    ) -> None:
        if len(actions) < 2 or len(set(actions)) != len(actions):
            raise ValueError("actions must be a unique finite set")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        if not terminal_weights or any(isinstance(w, bool) or not isinstance(w, int) or w < 0 for w in terminal_weights.values()):
            raise ValueError("terminal weights must be nonnegative integers")
        if not any(w > 0 for w in terminal_weights.values()):
            raise ValueError("at least one terminal must have positive weight")
        by_id = {n.node_id: n for n in nodes}
        if len(by_id) != len(nodes) or set(terminal_weights) - set(by_id):
            raise ValueError("duplicate node or unknown weighted terminal")
        seen: set[str] = set()
        for node in nodes:
            if not node.node_id or any(parent not in seen for parent in node.parents):
                raise ValueError("nodes must be uniquely named and topologically ordered")
            if isinstance(node.fixed_cost, bool) or not isinstance(node.fixed_cost, int) or node.fixed_cost <= 0:
                raise ValueError("node cost must be a positive integer")
            if not node.old_input_digest or not node.model_version or not node.tool_version:
                raise ValueError("history fingerprints and execution versions are required")
            seen.add(node.node_id)
        if any(not by_id[n].terminal for n in terminal_weights):
            raise ValueError("every weighted node must be marked terminal")
        self.actions = tuple(actions)
        self.nodes = by_id
        self.order = tuple(n.node_id for n in nodes)
        self.weights = dict(terminal_weights)
        self.predictions = dict(predictions or {})
        self.budget = budget
        self.call = model_call
        self.spent = 0
        self.outputs: dict[str, str] = {}
        self.current_keys: dict[str, str] = {}
        self.events: list[CallEvent] = []
        self.observations: list[Observation] = []
        self.status = "RUNNING"
        self._relevant = self._ancestors({node for node, weight in self.weights.items() if weight > 0})
        self._close_exact_cache()

    def set_terminal_weights(self, weights: Mapping[str, int]) -> None:
        """Replace provisional terminal masses before any terminal is observed."""
        if set(weights) != set(self.weights):
            raise ValueError("terminal weight domain changed")
        if any(isinstance(weight, bool) or not isinstance(weight, int) or weight < 0 for weight in weights.values()):
            raise ValueError("terminal weights must be nonnegative integers")
        if not any(weight > 0 for weight in weights.values()):
            raise ValueError("at least one terminal must have positive weight")
        if any(event.node_id in self.weights for event in self.events):
            raise ValueError("terminal weights are frozen after a paid terminal observation")
        self.weights = dict(weights)
        self._relevant = self._ancestors({node for node, weight in self.weights.items() if weight > 0})
        self._close_exact_cache()

    def _ancestors(self, roots: set[str]) -> set[str]:
        relevant = set(roots)
        stack = list(roots)
        while stack:
            node = self.nodes[stack.pop()]
            for parent in node.parents:
                if parent not in relevant:
                    relevant.add(parent)
                    stack.append(parent)
        return relevant

    def _effective_input(self, node: RuntimeNode) -> tuple[list[int], str]:
        parents = {parent: self.outputs[parent] for parent in node.parents}
        ids = list(node.render(parents))
        return ids, input_key(ids, node.model_version, node.tool_version)

    def _close_exact_cache(self) -> None:
        changed = True
        while changed:
            changed = False
            for node_id in self.order:
                node = self.nodes[node_id]
                if node_id not in self._relevant or node_id in self.outputs:
                    continue
                if not all(parent in self.outputs for parent in node.parents):
                    continue
                ids, new_key = self._effective_input(node)
                self.current_keys[node_id] = new_key
                if new_key == node.old_input_digest:
                    self.outputs[node_id] = node.old_output
                    changed = True

    def public_state(self) -> dict:
        legal: list[str] = []
        costs: dict[str, int] = {}
        for node_id in self.order:
            node = self.nodes[node_id]
            if node_id not in self._relevant or node_id in self.outputs:
                continue
            if all(parent in self.outputs for parent in node.parents):
                ids, key = self._effective_input(node)
                self.current_keys[node_id] = key
                cost = node.fixed_cost + len(ids)
                if self.spent + cost <= self.budget:
                    legal.append(node_id)
                    costs[node_id] = cost
        known_terminal = {node: label for node, label in self.outputs.items() if node in self.weights and self.weights[node] > 0}
        certificate = strict_certificate(self.actions, self.weights, known_terminal)
        return {
            "actions": list(self.actions),
            "graph": {node_id: list(self.nodes[node_id].parents) for node_id in self.order if node_id in self._relevant},
            "predictions": dict(self.predictions),
            "weights": dict(self.weights),
            "known_labels": known_terminal,
            "legal_actions": legal,
            "costs": costs,
            "budget_remaining": self.budget - self.spent,
            "certificate": {
                "status": certificate.status,
                "action": certificate.action,
                "known_gap": certificate.known_gap,
                "unknown_weight": certificate.unknown_weight,
            },
        }

    def _record_hook_event(self, event: CallEvent) -> None:
        if not isinstance(event, CallEvent):
            raise TypeError("only model-hook CallEvent records are accepted")
        if not event.call_id or any(item.call_id == event.call_id for item in self.events):
            raise ValueError("duplicate or empty hook call_id")
        self.events.append(event)

    def query(self, node_id: str) -> Observation:
        state = self.public_state()
        if node_id not in state["legal_actions"]:
            raise ValueError("illegal query")
        node = self.nodes[node_id]
        ids, digest = self._effective_input(node)
        cost = node.fixed_cost + len(ids)
        if self.spent + cost > self.budget:
            raise ValueError("budget exceeded")
        before = len(self.events)
        try:
            result = self.call(node_id, ids, node.model_version, node.tool_version, cost, self._record_hook_event)
        except Exception:
            observed = self.events[before:]
            self.spent += sum(max(0, int(event.paid_cost)) for event in observed)
            self.status = "INVALID"
            raise
        observed = self.events[before:]
        self.spent += sum(max(0, int(event.paid_cost)) for event in observed)
        if len(observed) != 1:
            self.status = "INVALID"
            raise ValueError("model-call hook event count does not match broker request")
        event = observed[0]
        if (event.node_id != node_id or event.input_digest != digest or event.input_tokens != len(ids)
                or event.paid_cost != cost or event.model_version != node.model_version
                or event.tool_version != node.tool_version
                or event.output_digest != hashlib.sha256(result.output.encode("utf-8")).hexdigest()
                or not event.raw_output_digest or event.parse_status != "OK"
                or event != result.event):
            self.status = "INVALID"
            raise ValueError("model-call hook event does not match broker request")
        self.outputs[node_id] = result.output
        observation = Observation(node_id, result.output, digest, cost, event.call_id)
        self.observations.append(observation)
        self._close_exact_cache()
        if node.terminal and result.output not in self.actions:
            self.status = "INVALID"
        return observation

    def accept_policy_action(self, action: dict) -> dict:
        if self.status == "INVALID":
            return {"status": "INVALID", "reason": "BROKER_INVALID_STATE"}
        if not isinstance(action, dict) or action.get("type") not in {"query", "stop", "abstain"}:
            self.status = "INVALID"
            return {"status": self.status, "reason": "MALFORMED_POLICY_ACTION"}
        state = self.public_state()
        cert = state["certificate"]
        if action["type"] == "stop":
            if cert["status"] not in {"CERTIFIED", "FULL_RESOLVED"} or action.get("action") != cert["action"]:
                self.status = "INVALID"
                return {"status": self.status, "reason": "UNPROVEN_STOP"}
            self.status = cert["status"]
            return {"status": self.status, "action": cert["action"]}
        if action["type"] == "abstain":
            self.status = "BUDGET_EXHAUSTED" if not state["legal_actions"] and self._has_unaffordable_ready_action() else "UNRESOLVED"
            return {"status": self.status, "reason": str(action.get("reason", "POLICY_ABSTAIN"))}
        try:
            observation = self.query(action.get("node_id"))
            return {"status": "RUNNING", "observation": observation.__dict__}
        except Exception as exc:
            self.status = "INVALID"
            return {"status": self.status, "reason": type(exc).__name__ + ":" + str(exc)}

    def _has_unaffordable_ready_action(self) -> bool:
        for node_id in self.order:
            node = self.nodes[node_id]
            if node_id not in self._relevant or node_id in self.outputs:
                continue
            if all(parent in self.outputs for parent in node.parents):
                ids, _ = self._effective_input(node)
                if node.fixed_cost + len(ids) > self.budget - self.spent:
                    return True
        return False


def run_policy_step(policy_worker: Path, state: dict, timeout_seconds: float = 5.0) -> dict:
    """Call the policy with PublicState only; use a mount/network namespace on POSIX."""
    clean_env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP") if key in os.environ}
    clean_env["PYTHONIOENCODING"] = "utf-8"
    start = time.perf_counter()
    if os.name == "posix":
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise RuntimeError("OS policy isolation unavailable: bwrap is required on POSIX")
        worker = policy_worker.resolve()
        command = [
            bwrap,
            "--unshare-user", "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts",
            "--die-with-parent",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--dir", "/policy",
            "--ro-bind", str(worker), "/policy/worker.py",
            "--chdir", "/tmp",
            sys.executable, "/policy/worker.py", "--once",
        ]
        cwd = "/"
        isolation = "BWRAP_USER_PID_NET_IPC_UTS_MOUNT"
    else:
        command = [sys.executable, str(policy_worker), "--once"]
        cwd = str(policy_worker.parent)
        isolation = "SUBPROCESS_LOGICAL_ONLY"
    proc = subprocess.run(
        command,
        input=json.dumps(state, ensure_ascii=False, allow_nan=False),
        capture_output=True, text=True, encoding="utf-8", timeout=timeout_seconds,
        cwd=cwd, env=clean_env, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"policy subprocess failed: {proc.returncode}: {proc.stderr[-1000:]}")
    result = json.loads(proc.stdout)
    if not isinstance(result, dict):
        raise ValueError("policy returned non-object action")
    result["planner_wall_seconds"] = time.perf_counter() - start
    result["policy_isolation"] = isolation
    return result


def event_ledger_status(
    events: Sequence[CallEvent], expected_spent: int, expected_calls: int,
    expected_manifest: Sequence[Mapping[str, object]] | None = None,
) -> tuple[str, str]:
    ids: set[str] = set()
    total = 0
    for event in events:
        if not event.call_id or event.call_id in ids:
            return "INVALID", "DUPLICATE_OR_EMPTY_CALL_ID"
        ids.add(event.call_id)
        if (event.paid_cost <= 0 or event.input_tokens <= 0 or not event.input_digest
                or not event.model_version or not event.tool_version
                or event.parse_status not in {"OK", "PARSE_ERROR", "FORWARD_ERROR"}
                or (event.parse_status == "OK" and (not event.output_digest or not event.raw_output_digest))
                or (event.parse_status == "PARSE_ERROR" and (event.output_digest or not event.raw_output_digest))
                or (event.parse_status == "FORWARD_ERROR" and (event.output_digest or event.raw_output_digest))):
            return "INVALID", "INVALID_EVENT_FIELDS"
        total += event.paid_cost
    if len(events) != expected_calls:
        return "INVALID", "CALL_COUNT_MISMATCH"
    if total != expected_spent:
        return "INVALID", "COST_MISMATCH"
    if expected_manifest is not None:
        if len(expected_manifest) != len(events):
            return "INVALID", "MANIFEST_COUNT_MISMATCH"
        for event, expected in zip(events, expected_manifest):
            for field in ("call_id", "node_id", "input_digest", "input_tokens", "output_digest", "paid_cost", "model_version", "tool_version", "raw_output_digest", "parse_status"):
                if getattr(event, field) != expected.get(field):
                    return "INVALID", "MANIFEST_MISMATCH"
    return "VALID_DEVELOPMENT_LEDGER", "OK"


def coverage_status(expected: Sequence[tuple[str, str, int]], observed: Sequence[dict], budget: int) -> tuple[str, str]:
    """Require exactly one result for every locked case-method-seed tuple."""
    expected_set = set(expected)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        return "INVALID", "INVALID_BUDGET"
    if len(expected_set) != len(expected):
        return "INVALID", "DUPLICATE_EXPECTED_KEY"
    keys: list[tuple[str, str, int]] = []
    total_cost = 0
    for row in observed:
        try:
            case_id, method_id, seed, cost = row["case_id"], row["method_id"], row["seed"], row["cost"]
        except (KeyError, TypeError, ValueError):
            return "INVALID", "MALFORMED_RESULT"
        if not isinstance(case_id, str) or not case_id or not isinstance(method_id, str) or not method_id:
            return "INVALID", "MALFORMED_RESULT"
        if isinstance(seed, bool) or not isinstance(seed, int) or isinstance(cost, bool) or not isinstance(cost, int):
            return "INVALID", "MALFORMED_RESULT"
        key = (case_id, method_id, seed)
        if cost < 0:
            return "INVALID", "NEGATIVE_COST"
        keys.append(key)
        total_cost += cost
    if len(keys) != len(set(keys)):
        return "INVALID", "DUPLICATE_RESULT_KEY"
    if set(keys) != expected_set:
        return "INVALID", "MISSING_OR_EXTRA_RESULT"
    if total_cost > budget:
        return "INVALID", "BUDGET_EXCEEDED"
    return "VALID_DEVELOPMENT_COVERAGE", "OK"
