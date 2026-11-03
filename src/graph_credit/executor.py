"""Closed-input, time-unrolled graph execution and conservative counterfactual replay.

Logical edges are source-major. Fixed self-state and final aggregation edges are
always present. Reuse is forbidden throughout the union-graph affected closure,
even when an affected node happens to serialize to the same prompt.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


ROLE_PROMPTS = (
    "Solve the question independently; state a candidate answer and the key reasoning.",
    "Check evidence and arithmetic carefully; identify unsupported claims before answering.",
    "Check the logical reasoning step by step; correct any inconsistent inference.",
    "Look for counterexamples and alternative interpretations; give your best supported answer.",
)
MESSAGE_INSTRUCTION = (
    'Return only one JSON object with exactly these string fields: "answer", '
    '"evidence", "uncertainty". Put the short final answer in "answer"; '
    'put concise reasoning or checks in "evidence"; put unresolved issues in '
    '"uncertainty" (empty string if none). Stay within the output token limit. '
    'For a numerical question put only the final number in "answer".'
)
AGGREGATOR_PROMPT = (
    "You are the fixed answer aggregator. Independently verify the candidates "
    "against the question and any supplied evidence, resolve disagreements, "
    "and return the best final answer. " + MESSAGE_INSTRUCTION
)


@dataclass(frozen=True)
class GraphConfig:
    rounds: int = 2
    n_workers: int = 4
    message_tokens: int = 128
    aggregate_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    role_prompts: tuple[str, ...] = ROLE_PROMPTS
    max_input_tokens: int = 8192
    executor_version: str = "unrolled_closed_input_v1"

    def __post_init__(self) -> None:
        if self.rounds not in (2, 3):
            raise ValueError("Only preregistered 2- or 3-round execution is supported")
        if self.n_workers != 4 or len(self.role_prompts) != 4:
            raise ValueError("Pilot requires four fixed worker roles")
        if self.message_tokens < 32 or self.aggregate_tokens < 32:
            raise ValueError("Token limits must allow the fixed three-field message schema")
        if self.do_sample and (self.temperature <= 0 or not 0 < self.top_p <= 1):
            raise ValueError("Invalid explicit sampling settings")
        if self.max_input_tokens < 128:
            raise ValueError("Invalid maximum prompt token count")


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    gpu_seconds: float = 0.0
    gpu_peak_allocated_bytes: int = 0
    finish_reason: str = "unknown"
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class Backend(Protocol):
    identity: Mapping[str, Any]

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: Sequence[int]) -> str: ...
    def generate(self, messages: list[dict[str, str]], seed: int,
                 max_new_tokens: int, config: GraphConfig) -> Generation: ...


class CacheIntegrityError(RuntimeError):
    pass


def logical_edges(n: int = 4) -> tuple[tuple[int, int], ...]:
    return tuple((src, dst) for src in range(n) for dst in range(n) if src != dst)


def validate_mask(mask: Sequence[int], n: int = 4) -> tuple[int, ...]:
    values = tuple(mask)
    if len(values) != n * (n - 1) or any(type(x) not in (int, bool) or x not in (0, 1) for x in values):
        raise ValueError("Graph mask must contain exactly 12 explicit binary values")
    return tuple(int(x) for x in values)


def derive_seed(question_id: str, execution_seed: int, node: str) -> int:
    """Same question/node stream across graph masks; independent across seed IDs."""
    return int(digest([question_id, execution_seed, node])[:16], 16) % (2**31 - 1)


def unrolled_dependencies(mask: Sequence[int], rounds: int = 2) -> dict[str, set[str]]:
    graph = validate_mask(mask)
    parents = {f"r{r}:w{w}": set() for r in range(1, rounds + 1) for w in range(4)}
    edges = logical_edges()
    for r in range(2, rounds + 1):
        for w in range(4):
            parents[f"r{r}:w{w}"].add(f"r{r-1}:w{w}")
        for i, (src, dst) in enumerate(edges):
            if graph[i]:
                parents[f"r{r}:w{dst}"].add(f"r{r-1}:w{src}")
    parents["aggregate"] = {f"r{rounds}:w{w}" for w in range(4)}
    return parents


def affected_closure(old_mask: Sequence[int], new_mask: Sequence[int], rounds: int = 2) -> set[str]:
    old, new = validate_mask(old_mask), validate_mask(new_mask)
    changed = [i for i in range(12) if old[i] != new[i]]
    affected = {f"r{r}:w{logical_edges()[i][1]}" for i in changed for r in range(2, rounds + 1)}
    deps_old, deps_new = unrolled_dependencies(old, rounds), unrolled_dependencies(new, rounds)
    while True:
        descendants = {node for node in deps_old if (deps_old[node] | deps_new[node]) & affected}
        updated = affected | descendants
        if updated == affected:
            return affected
        affected = updated


def normalize_message(text: str, backend: Backend, token_limit: int) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Deterministic, non-model repair; malformed output and trimming remain visible."""
    fields = ("answer", "evidence", "uncertainty")
    parsed = None
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and all(isinstance(candidate.get(k), str) for k in fields):
            parsed = {k: candidate[k] for k in fields}
            break
    format_ok = parsed is not None
    if parsed is None:
        parsed = {"answer": text.strip(), "evidence": "", "uncertainty": "FORMAT_ERROR"}
    serialized = canonical(parsed)
    trimmed = False
    # Trim evidence/uncertainty before the actual answer. Count the wire message,
    # including field names and JSON escaping, rather than generated text alone.
    for key in ("evidence", "uncertainty", "answer"):
        while len(backend.encode(serialized)) > token_limit and parsed[key]:
            tokens = backend.encode(parsed[key])
            overshoot = len(backend.encode(serialized)) - token_limit
            tokens = tokens[:max(0, len(tokens) - max(1, overshoot))]
            parsed[key] = backend.decode(tokens)
            serialized = canonical(parsed)
            trimmed = True
    if len(backend.encode(serialized)) > token_limit:
        raise ValueError("Tokenizer cannot encode the empty fixed message schema within the cap")
    return serialized, parsed, {"format_ok": format_ok, "message_trimmed": trimmed,
                                 "message_tokens": len(backend.encode(serialized))}


@dataclass(frozen=True)
class NodeRecord:
    node: str
    input_fingerprint: str
    output_digest: str
    seed: int
    messages: list[dict[str, str]]
    generation: Generation
    wire_message: str
    parsed: dict[str, str]
    message_metadata: dict[str, Any]
    received_message_tokens: int
    self_state_tokens: int
    reused: bool = False
    source_execution_id: str | None = None

    def integrity_value(self) -> dict[str, Any]:
        return {"node": self.node, "input_fingerprint": self.input_fingerprint,
                "seed": self.seed, "messages": self.messages,
                "generation": dataclasses.asdict(self.generation), "wire_message": self.wire_message,
                "parsed": self.parsed, "message_metadata": self.message_metadata,
                "received_message_tokens": self.received_message_tokens, "self_state_tokens": self.self_state_tokens}


@dataclass
class ExecutionResult:
    execution_id: str
    question_id: str
    question_fingerprint: str
    mask: tuple[int, ...]
    seed: int
    protocol_hash: str
    nodes: dict[str, NodeRecord]
    closure: list[str]
    answer: str
    elapsed_seconds: float

    def cost(self, physical_only: bool = True) -> dict[str, int | float]:
        records = [n for n in self.nodes.values() if not physical_only or not n.reused]
        return {
            "calls": len(records),
            "input_tokens": sum(n.generation.input_tokens for n in records),
            "output_tokens": sum(n.generation.output_tokens for n in records),
            "received_message_tokens": sum(n.received_message_tokens for n in records),
            "self_state_tokens": sum(n.self_state_tokens for n in records),
            "model_elapsed_seconds": sum(n.generation.elapsed_seconds for n in records),
            "gpu_seconds": sum(n.generation.gpu_seconds for n in records),
            "peak_allocated_bytes": max((n.generation.gpu_peak_allocated_bytes for n in records), default=0),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "physical_cost": self.cost(), "logical_service_cost": self.cost(False),
                "cache_reuse_count": sum(n.reused for n in self.nodes.values())}


class Executor:
    def __init__(self, backend: Backend, config: GraphConfig | None = None,
                 model_identity: Mapping[str, Any] | None = None) -> None:
        self.backend = backend
        self.config = config or GraphConfig()
        # All prompts, serialization rules, model/tokenizer identity, decoding
        # settings and software versions must be fixed before the first call.
        self.model_identity = dict(model_identity or backend.identity)
        self.protocol_hash = digest({"config": dataclasses.asdict(self.config), "backend": self.model_identity,
                                     "executor_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     "message_instruction": MESSAGE_INSTRUCTION,
                                     "aggregator_prompt": AGGREGATOR_PROMPT,
                                     "edge_order": logical_edges(), "serialization": "canonical_json_v1"})

    @staticmethod
    def question_payload(question: Mapping[str, Any]) -> dict[str, Any]:
        if any(k in question for k in ("answer", "label", "gold", "target", "solution", "quality")):
            raise ValueError("Gold/quality fields cannot enter the execution interface")
        if not isinstance(question.get("id"), str) or not isinstance(question.get("question"), str):
            raise ValueError("Question requires string id and question")
        if not isinstance(question.get("context", ""), str):
            raise ValueError("Supplied context must be an explicit string, not an unvalidated metadata object")
        # Metadata not explicitly used by the model remains excluded from prompts.
        return {"id": question["id"], "question": question["question"], "context": question.get("context", "")}

    def _messages(self, question: Mapping[str, Any], node: str, graph: tuple[int, ...],
                  outputs: dict[str, NodeRecord]) -> tuple[list[dict[str, str]], int, int]:
        qtext = question["question"]
        if question.get("context"):
            qtext += "\n\nSupplied context:\n" + str(question["context"])
        if node == "aggregate":
            inbound = [{"worker": w, "message": outputs[f"r{self.config.rounds}:w{w}"].wire_message} for w in range(4)]
            received = sum(len(self.backend.encode(i["message"])) for i in inbound)
            return [{"role": "system", "content": AGGREGATOR_PROMPT},
                    {"role": "user", "content": qtext + "\n\nFinal worker messages:\n" + canonical(inbound)}], received, 0
        r, w = (int(s[1:]) for s in node.split(":"))
        role = self.config.role_prompts[w]
        prompt = qtext
        received, self_tokens = 0, 0
        if r == 1:
            prompt += "\n\nRound 1: work independently. No external messages are available."
        else:
            own = outputs[f"r{r-1}:w{w}"].wire_message
            self_tokens = len(self.backend.encode(own))
            inbound = [{"worker": src, "message": outputs[f"r{r-1}:w{src}"].wire_message}
                       for i, (src, dst) in enumerate(logical_edges()) if graph[i] and dst == w]
            received = sum(len(self.backend.encode(i["message"])) for i in inbound)
            prompt += f"\n\nRound {r}. Your previous state:\n{own}"
            prompt += "\n\nExternal messages (ordered by source worker):\n" + (canonical(inbound) if inbound else "[] (no external messages)")
            prompt += "\nRecheck your candidate using the question, your previous state and these messages."
        return [{"role": "system", "content": role + " " + MESSAGE_INSTRUCTION},
                {"role": "user", "content": prompt}], received, self_tokens

    def execute(self, question: Mapping[str, Any], mask: Sequence[int], seed: int,
                base: ExecutionResult | None = None) -> ExecutionResult:
        started = time.perf_counter()
        payload, graph = self.question_payload(question), validate_mask(mask)
        qfingerprint = digest(payload)
        qid = payload["id"]
        execution_id = uuid.uuid4().hex
        can_compare = (base is not None and base.question_fingerprint == qfingerprint
                       and base.seed == seed and base.protocol_hash == self.protocol_hash)
        closure = affected_closure(base.mask, graph, self.config.rounds) if can_compare else set(unrolled_dependencies(graph, self.config.rounds))
        outputs: dict[str, NodeRecord] = {}
        ordering = [f"r{r}:w{w}" for r in range(1, self.config.rounds + 1) for w in range(4)] + ["aggregate"]
        for node in ordering:
            messages, received, self_tokens = self._messages(payload, node, graph, outputs)
            node_seed = derive_seed(qid, seed, node)
            max_new_tokens = self.config.aggregate_tokens if node == "aggregate" else self.config.message_tokens
            fingerprint = digest({"question": qfingerprint, "node": node, "messages": messages,
                                  "seed": node_seed, "protocol": self.protocol_hash,
                                  "max_new_tokens": max_new_tokens})
            cached = base.nodes.get(node) if can_compare and node not in closure else None
            if cached is not None and digest(cached.integrity_value()) != cached.output_digest:
                raise CacheIntegrityError(f"Cached node output was modified: {node}")
            if cached is not None and cached.input_fingerprint == fingerprint:
                outputs[node] = dataclasses.replace(cached, reused=True, source_execution_id=base.execution_id)
                continue
            generation = self.backend.generate(messages, node_seed, max_new_tokens, self.config)
            if generation.input_tokens > self.config.max_input_tokens:
                raise ValueError("Backend executed an over-limit input; backend must reject before generation")
            if generation.output_tokens > max_new_tokens:
                raise ValueError("Backend violated the frozen output token bound")
            wire, parsed, metadata = normalize_message(generation.text, self.backend, max_new_tokens)
            record = NodeRecord(node, fingerprint, "", node_seed, messages, generation, wire, parsed, metadata,
                                received, self_tokens, False, None)
            record = dataclasses.replace(record, output_digest=digest(record.integrity_value()))
            outputs[node] = record
        return ExecutionResult(execution_id, qid, qfingerprint, graph, seed, self.protocol_hash, outputs,
                               sorted(closure), outputs["aggregate"].parsed["answer"], time.perf_counter() - started)
