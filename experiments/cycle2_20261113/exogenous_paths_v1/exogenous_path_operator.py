"""CPU-prepared four-call DAG; no model/data loading, scoring, or admission.

Two preconstructed qualified HFBackend-compatible objects are lent by the caller.
Existing DurableBackend wrappers are supported without changing their identity,
RNG, Generation, or native cost records. Journals/returned records are PRIVATE.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid
from dataclasses import dataclass


OPERATOR_ID = "exogenous_path_four_call_dag_v1_cpu_staging"
NODES = ("U", "V", "R", "M")
GRAPH_CONDITIONS = ("G0", "G0_MINUS_EU", "G0_MINUS_EV", "G1", "G1_MINUS_EU", "G1_MINUS_EV")
SCHEDULE = GRAPH_CONDITIONS + ("INDEPENDENT3_M", "SINGLE_M", "SINGLE_Q")
G0_EDGES = (("U", "R"), ("V", "R"), ("R", "M"), ("U", "M"), ("V", "M"))
MODEL_ROUTE = {"U": "qwen", "V": "qwen", "R": "mistral", "M": "mistral"}
SOURCE_INSTRUCTION = (
    "Read the question and supplied documents. Give a concise evidence note with "
    "a candidate answer and the relevant document titles. Use only the supplied "
    "evidence; state uncertainty briefly when it is insufficient. Output plain text, "
    "not JSON. Do not repeat the documents."
)
RELAY_INSTRUCTION = (
    "Read the question and received evidence notes. Check their consistency and "
    "combine the relevant evidence into one concise note with a candidate answer. "
    "Do not assume access to documents that are not supplied. State uncertainty "
    "briefly when evidence is insufficient. Output plain text, not JSON."
)
FINAL_INSTRUCTION = (
    "Answer the question using the supplied documents and evidence notes, if any. "
    "Resolve disagreements internally. Output only the short final answer, with "
    "no explanation, answer label, JSON, Markdown, or surrounding quotation marks."
)
FINAL_TEXT_RULE = "Entire Generation.text.strip(); no extraction, parsing, gold access or fallback"
PARTITION_RULE = "Raw title UTF8 SHA256 sort; original index tie; alternating U/V; each shard original input order"
COST_FIELDS = ("calls", "input_tokens", "output_tokens", "model_elapsed_seconds", "gpu_seconds")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PathConfig:
    message_tokens: int = 128
    final_tokens: int = 256
    max_input_tokens: int = 8192
    do_sample: bool = True
    temperature: float = .7
    top_p: float = .9

    def __post_init__(self):
        if (type(self.message_tokens) is not int or self.message_tokens < 1
                or type(self.final_tokens) is not int or self.final_tokens < 1
                or type(self.max_input_tokens) is not int or self.max_input_tokens < 1):
            raise ValueError("Positive integer token caps required")
        if type(self.do_sample) is not bool or not math.isfinite(self.temperature) or not math.isfinite(self.top_p):
            raise ValueError("Explicit finite sampling settings required")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid sampling settings")


def validate_payload(payload):
    """Only the input-only Hotpot schema; preserve every title/sentence exactly."""
    if not isinstance(payload, dict) or set(payload) != {"id", "question", "context"}:
        raise ValueError("Payload permits exactly id/question/context; labels and metadata forbidden")
    if any(type(payload[k]) is not str or not payload[k].strip() for k in ("id", "question")):
        raise ValueError("Nonempty string id and question required")
    context = payload["context"]
    if not isinstance(context, dict) or set(context) != {"title", "sentences"}:
        raise ValueError("Context must be the explicit title/sentences dictionary")
    titles, sentences = context["title"], context["sentences"]
    if type(titles) is not list or type(sentences) is not list or len(titles) != 10 or len(sentences) != 10:
        raise ValueError("Exactly ten complete documents required")
    if any(type(t) is not str or not t for t in titles):
        raise ValueError("Document titles must be nonempty strings")
    if any(type(s) is not list or any(type(x) is not str for x in s) for s in sentences):
        raise ValueError("Every document must retain its complete string sentence list")
    canonical(payload)
    return copy.deepcopy(payload)


def partition_document_indices(payload):
    """No title normalization, labels, type, level, model or quality access."""
    p = validate_payload(payload)
    title_hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in p["context"]["title"]]
    ordered = sorted(range(10), key=lambda i: (title_hashes[i], i))
    return {"U": sorted(ordered[::2]), "V": sorted(ordered[1::2]),
            "raw_title_sha256": title_hashes, "rule": PARTITION_RULE}


def _documents(payload, indices):
    return [{"title": payload["context"]["title"][i],
             "sentences": copy.deepcopy(payload["context"]["sentences"][i])} for i in indices]


def _messages(payload, documents, inbound, instruction):
    return [{"role": "system", "content": instruction},
            {"role": "user", "content": canonical({"question": payload["question"],
                "documents": documents, "evidence_notes": inbound})}]


def build_full_context_messages(payload, mode, peer_messages=None):
    """Input-only capacity probe, or actual full-context control prompt.

    peer_messages=None omits future generated messages; it is NOT a dynamic
    capacity guarantee. Only INDEPENDENT3_M may receive U/V/R text messages.
    """
    p = validate_payload(payload)
    modes = {"SINGLE_Q", "SINGLE_M", "INDEPENDENT3_U", "INDEPENDENT3_V", "INDEPENDENT3_R", "INDEPENDENT3_M"}
    if mode not in modes:
        raise ValueError("Unknown full-context prompt mode")
    if peer_messages is not None and mode != "INDEPENDENT3_M":
        raise ValueError("Independent workers and single models cannot receive peer messages")
    inbound = []
    if peer_messages is not None:
        if set(peer_messages) != {"U", "V", "R"} or any(type(v) is not str for v in peer_messages.values()):
            raise ValueError("Three exact string notes U/V/R required")
        inbound = [{"source": n, "text": peer_messages[n]} for n in ("U", "V", "R")]
    final = mode in {"SINGLE_Q", "SINGLE_M", "INDEPENDENT3_M"}
    return _messages(p, _documents(p, range(10)), inbound, FINAL_INSTRUCTION if final else SOURCE_INSTRUCTION)


def input_only_messages(payload, config=None):
    """Exact static framing for external tokenizer screening; no labels/readers."""
    config = config or PathConfig()
    result = {}
    for mode in ("SINGLE_Q", "SINGLE_M", "INDEPENDENT3_U", "INDEPENDENT3_V", "INDEPENDENT3_R", "INDEPENDENT3_M"):
        model = "qwen" if mode in {"SINGLE_Q", "INDEPENDENT3_U", "INDEPENDENT3_V"} else "mistral"
        final = mode in {"SINGLE_Q", "SINGLE_M", "INDEPENDENT3_M"}
        result[mode] = {"model": model, "messages": build_full_context_messages(payload, mode),
                        "max_new_tokens": config.final_tokens if final else config.message_tokens,
                        "future_generated_notes_included": False}
    return result


def condition_edges(condition, removed_upstream):
    if condition not in GRAPH_CONDITIONS or removed_upstream not in {"U", "V"}:
        raise ValueError("Invalid graph condition or exogenous upstream deletion")
    removed = set()
    if condition.startswith("G1"):
        removed.add((removed_upstream, "R"))
    if condition.endswith("_MINUS_EU"):
        removed.add(("U", "M"))
    if condition.endswith("_MINUS_EV"):
        removed.add(("V", "M"))
    return tuple(edge for edge in G0_EDGES if edge not in removed)


def node_seed(question_id, execution_seed, node):
    return int(digest(["exogenous_path_rng_v1", question_id, execution_seed, node])[:16], 16) % (2**31 - 1)


class InputRejected(ValueError):
    def __init__(self, reason, measured_input_tokens, max_input_tokens, model_position_limit, max_new_tokens):
        super().__init__(reason)
        self.details = {"reason": reason, "input_tokens": measured_input_tokens,
            "max_input_tokens": max_input_tokens, "model_position_limit": model_position_limit,
            "max_new_tokens": max_new_tokens, "truncation_performed": False}


class PathExecutionError(RuntimeError):
    def __init__(self, partial_result):
        super().__init__("Path matrix incomplete; no retry or score imputation permitted")
        self.partial_result = partial_result


def _token_count(backend, messages):
    tokens = backend.tokenizer.apply_chat_template(copy.deepcopy(messages), tokenize=True, add_generation_prompt=True)
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    if hasattr(tokens, "shape"):
        shape = tuple(tokens.shape)
        if len(shape) == 1:
            return int(shape[0])
        if len(shape) == 2 and shape[0] == 1:
            return int(shape[1])
        raise ValueError("Unexpected tokenizer tensor shape")
    if type(tokens) is list and all(type(t) is int for t in tokens):
        return len(tokens)
    if type(tokens) is list and len(tokens) == 1 and type(tokens[0]) is list and all(type(t) is int for t in tokens[0]):
        return len(tokens[0])
    raise ValueError("Tokenizer did not return one explicit token sequence")


def validate_request_capacity(backend, messages, config, max_new_tokens):
    """Receiver chat-template count, no truncation; include output reservation."""
    count = _token_count(backend, messages)
    model_config = getattr(getattr(backend, "model", None), "config", None)
    limit = getattr(model_config, "max_position_embeddings", config.max_input_tokens + max_new_tokens)
    if type(limit) is not int or limit < 1:
        raise ValueError("Invalid model position limit")
    if count > config.max_input_tokens:
        raise InputRejected("INPUT_TOKEN_LIMIT", count, config.max_input_tokens, limit, max_new_tokens)
    if count + max_new_tokens > limit:
        raise InputRejected("MODEL_POSITION_LIMIT", count, config.max_input_tokens, limit, max_new_tokens)
    return {"input_tokens": count, "max_input_tokens": config.max_input_tokens,
            "model_position_limit": limit, "max_new_tokens": max_new_tokens}


def _generation_dict(generation):
    if not dataclasses.is_dataclass(generation):
        raise ValueError("Backend must return the qualified dataclass Generation interface")
    g = dataclasses.asdict(generation)
    expected = {"text", "input_tokens", "output_tokens", "elapsed_seconds", "gpu_seconds",
                "gpu_peak_allocated_bytes", "finish_reason", "call_id"}
    if set(g) != expected or type(g["text"]) is not str or type(g["call_id"]) is not str or not g["call_id"]:
        raise ValueError("Invalid Generation schema")
    if g["finish_reason"] not in {"eos", "length"}:
        raise ValueError("Unqualified finish reason")
    for k in ("input_tokens", "output_tokens", "gpu_peak_allocated_bytes"):
        if type(g[k]) is not int or g[k] < 0:
            raise ValueError("Invalid Generation integer cost")
    for k in ("elapsed_seconds", "gpu_seconds"):
        if type(g[k]) not in (int, float) or not math.isfinite(g[k]) or g[k] < 0:
            raise ValueError("Invalid Generation elapsed cost")
    return g


def _logical_cost(records):
    records = list(records)
    out = {}
    for model in ("qwen", "mistral"):
        selected = [r for r in records if r["model"] == model]
        out[model] = {"calls": len(selected), "input_tokens": sum(r["generation"]["input_tokens"] for r in selected),
            "output_tokens": sum(r["generation"]["output_tokens"] for r in selected),
            "model_elapsed_seconds": sum(r["generation"]["elapsed_seconds"] for r in selected),
            "gpu_seconds": sum(r["generation"]["gpu_seconds"] for r in selected),
            "peak_allocated_bytes": max((r["generation"]["gpu_peak_allocated_bytes"] for r in selected), default=0),
            "received_message_tokens": sum(r["received_message_tokens"] for r in selected)}
    return out


def _physical_cost(native, traces):
    """Native records counted once; failures/missing fields are never zero-filled."""
    output = {}
    for model in ("qwen", "mistral"):
        rows = native[model]
        unknown_requests = sum(t["model"] == model and t.get("backend_invoked") is True
                               and not t.get("native_events") for t in traces)
        keys = {"input_tokens": "input_tokens", "output_tokens": "output_tokens",
                "model_elapsed_seconds": "elapsed_seconds", "gpu_seconds": "gpu_seconds"}
        known, totals = {}, {"calls": None if unknown_requests else len(rows)}
        for key, native_key in keys.items():
            values = [r.get(native_key) for r in rows]
            known[key] = sum(v for v in values if type(v) in (int, float) and math.isfinite(v))
            totals[key] = None if unknown_requests or any(v is None for v in values) else known[key]
        output[model] = {"totals": totals, "known_subtotals": known,
            "native_successes": sum(r.get("status") == "SUCCESS" for r in rows),
            "native_failures": sum(r.get("status") == "FAILED" for r in rows),
            "unknown_backend_requests": unknown_requests, "resident_GPU_hours": None}
    return output


class PathOperator:
    def __init__(self, qwen, mistral, config=None, before_request=None):
        self.backends = {"qwen": qwen, "mistral": mistral}
        if qwen is mistral:
            raise ValueError("Two distinct model backends required")
        self.config = config or PathConfig()
        if not isinstance(self.config, PathConfig):
            raise ValueError("Explicit new PathConfig required; old graph protocol is not accepted")
        self.before_request = before_request
        self.identities = {k: copy.deepcopy(b.identity) for k, b in self.backends.items()}
        self.identity_hashes = {k: digest(v) for k, v in self.identities.items()}
        if self.identity_hashes["qwen"] == self.identity_hashes["mistral"]:
            raise ValueError("Model identities must distinguish Qwen and Mistral")
        for b in self.backends.values():
            if not isinstance(b.physical_calls, list):
                raise ValueError("Append-only native physical_calls required")
            if before_request is not None and getattr(b, "before_request", None) is not None:
                raise ValueError("Admission hook must not be installed twice in operator and durable wrapper")
        self.protocol = {"operator_id": OPERATOR_ID, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config": dataclasses.asdict(self.config), "model_identities": self.identities,
            "G0_edges": G0_EDGES, "schedule": SCHEDULE, "route": MODEL_ROUTE,
            "source_instruction": SOURCE_INSTRUCTION, "relay_instruction": RELAY_INSTRUCTION,
            "final_instruction": FINAL_INSTRUCTION, "partition_rule": PARTITION_RULE,
            "wire_rule": "Entire intermediate Generation.text.strip(); no parsing/fallback/truncation",
            "final_text_rule": FINAL_TEXT_RULE, "serialization": "sorted compact JSON UTF8, unmodified document strings",
            "seed_rule": "SHA256 canonical [exogenous_path_rng_v1,question_id,execution_seed,node], first16hex mod(2^31-1)",
            "cache_rule": "per-matrix exact request+model identity+seed+config, all baselines share cache",
            "baseline_independent_M_has_full_context": True, "admission_granted": False}
        self.protocol_hash = digest(self.protocol)
        self._lock = threading.Lock()
        self._poisoned = False
        self._active_request = None

    @property
    def active_request(self):
        return copy.deepcopy(self._active_request)

    def _identity_check(self):
        if any(digest(b.identity) != self.identity_hashes[k] for k, b in self.backends.items()):
            raise ValueError("Backend model/tokenizer/revision identity changed")

    def _request(self, payload, execution_seed, condition, node, model, messages, received, final):
        self._identity_check()
        derived = node_seed(payload["id"], execution_seed, node)
        cap = self.config.final_tokens if final else self.config.message_tokens
        request = {"question_fingerprint": digest(payload), "node": node, "model": model,
            "model_identity_sha256": self.identity_hashes[model], "messages": copy.deepcopy(messages),
            "seed": derived, "max_new_tokens": cap, "config": dataclasses.asdict(self.config),
            "protocol_sha256": self.protocol_hash}
        fingerprint = digest(request)
        if fingerprint in self._cache:
            base = self._cache[fingerprint]
            if digest(base["value"]) != base["integrity_sha256"] or base["value"]["request"] != request:
                raise ValueError("Exact cache integrity mismatch")
            record = copy.deepcopy(base["value"])
            record.update(reused=True, source_request_id=record["request_id"])
            return record
        request_id = uuid.uuid4().hex
        trace = {"request_id": request_id, "condition": condition, "execution_seed": execution_seed,
                 "node": node, "model": model, "request_fingerprint": fingerprint,
                 "request": request, "status": "PREFLIGHT_STARTED", "backend_invoked": False,
                 "before_request_invoked": False, "generation": None, "native_events": []}
        self._traces.append(trace)
        backend = self.backends[model]
        native_start = len(backend.physical_calls)
        trace["native_start"] = native_start
        self._active_request = {"matrix_id": self._matrix_id, "condition": condition, "node": node,
            "model": model, "execution_seed": execution_seed, "seed": derived,
            "question_fingerprint": request["question_fingerprint"], "request_id": request_id,
            "request_fingerprint": fingerprint, "protocol_sha256": self.protocol_hash,
            "max_new_tokens": cap}
        try:
            trace["capacity"] = validate_request_capacity(backend, messages, self.config, cap)
            if self.before_request is not None:
                trace["before_request_invoked"] = True
                self.before_request(copy.deepcopy({**self._active_request, "capacity": trace["capacity"]}))
            self._identity_check()
            trace["status"] = "BACKEND_STARTED"
            trace["backend_invoked"] = True
            generation = backend.generate(copy.deepcopy(messages), derived, cap, self.config)
            raw = _generation_dict(generation)
            trace["generation"] = raw
            self._identity_check()
            trace["native_events"] = copy.deepcopy(backend.physical_calls[native_start:])
            if raw["input_tokens"] != trace["capacity"]["input_tokens"] or raw["output_tokens"] > cap:
                raise ValueError("Generation token accounting differs from exact preflight/cap")
            events = trace["native_events"]
            if len(events) != 1 or events[0].get("status") != "SUCCESS":
                raise ValueError("One matching native SUCCESS required for returned Generation")
            native = events[0]
            expected = {"call_id": raw["call_id"], "seed": derived, "prompt_sha256": digest(messages),
                "input_tokens": raw["input_tokens"], "output_tokens": raw["output_tokens"],
                "elapsed_seconds": raw["elapsed_seconds"], "gpu_seconds": raw["gpu_seconds"],
                "gpu_peak_allocated_bytes": raw["gpu_peak_allocated_bytes"], "finish_reason": raw["finish_reason"]}
            if any(native.get(k) != v for k, v in expected.items()):
                raise ValueError("Native/Generation/request mismatch")
            if raw["call_id"] in self._call_ids:
                raise ValueError("Repeated native call_id across fresh requests")
            self._call_ids.add(raw["call_id"])
            trace["status"] = "GENERATION_RETURNED"
            record = {"node": node, "model": model, "request_id": request_id,
                "request_fingerprint": fingerprint, "request": request,
                "generation": raw, "wire_message": raw["text"].strip(),
                "received_message_tokens": received, "transport_truncated": False,
                "json_parse_attempted": False, "is_final": final,
                "reused": False, "source_request_id": None}
            self._cache[fingerprint] = {"value": copy.deepcopy(record), "integrity_sha256": digest(record)}
            return record
        except Exception as error:
            trace["native_events"] = copy.deepcopy(backend.physical_calls[native_start:])
            trace["status"] = "INPUT_REJECTED" if isinstance(error, InputRejected) else "FAILED"
            trace["error_type"] = type(error).__name__
            if isinstance(error, InputRejected):
                trace["input_rejection"] = copy.deepcopy(error.details)
            # Durable journal failures can expose returned RAM Generation. Keep
            # it as emergency evidence, never fabricate a successful condition.
            current, seen = error, set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                value = getattr(current, "returned_generation", None)
                if value is not None and dataclasses.is_dataclass(value):
                    trace["emergency_returned_generation"] = dataclasses.asdict(value)
                current = current.__cause__
            raise
        finally:
            trace["native_end"] = len(backend.physical_calls)
            self._active_request = None

    def _condition(self, payload, execution_seed, condition, removed_upstream, shards):
        outputs = {}
        edges = condition_edges(condition, removed_upstream) if condition in GRAPH_CONDITIONS else ()
        order = ("M",) if condition in {"SINGLE_Q", "SINGLE_M"} else NODES
        row = {"key": f"{execution_seed}/{condition}", "condition": condition,
            "execution_seed": execution_seed, "edges": [list(x) for x in edges],
            "status": "STARTED", "nodes": outputs}
        self._conditions.append(row)
        for node in order:
            final = node == "M"
            model = "qwen" if condition == "SINGLE_Q" else MODEL_ROUTE[node]
            inbound = []
            if condition in GRAPH_CONDITIONS:
                parents = [source for source in NODES if (source, node) in edges]
                inbound = [{"source": source, "text": outputs[source]["wire_message"]} for source in parents]
                docs = _documents(payload, shards[node]) if node in {"U", "V"} else []
                prompt = FINAL_INSTRUCTION if final else SOURCE_INSTRUCTION if node in {"U", "V"} else RELAY_INSTRUCTION
                messages = _messages(payload, docs, inbound, prompt)
            elif condition == "INDEPENDENT3_M":
                notes = {source: outputs[source]["wire_message"] for source in ("U", "V", "R")} if final else None
                messages = build_full_context_messages(payload, "INDEPENDENT3_" + node, notes)
                inbound = [] if notes is None else [{"source": n, "text": notes[n]} for n in ("U", "V", "R")]
            else:
                messages = build_full_context_messages(payload, condition)
            received = sum(len(self.backends[model].encode(message["text"])) for message in inbound)
            outputs[node] = self._request(payload, execution_seed, condition, node, model, messages, received, final)
        row.update(status="COMPLETE", answer=outputs["M"]["generation"]["text"].strip(),
                   logical_service_cost_by_model=_logical_cost(outputs.values()),
                   cache_reuse_count=sum(r["reused"] for r in outputs.values()))

    def _snapshot(self, payload, seeds, removed_upstream, shards, started, status):
        native = {name: copy.deepcopy(b.physical_calls[self._native_starts[name]:]) for name, b in self.backends.items()}
        return {"status": status, "matrix_id": self._matrix_id, "question_fingerprint": digest(payload),
            "execution_seeds": list(seeds), "removed_upstream": removed_upstream,
            "document_partition": copy.deepcopy(shards), "protocol_sha256": self.protocol_hash,
            "protocol": copy.deepcopy(self.protocol), "conditions": copy.deepcopy(self._conditions),
            "request_trace": copy.deepcopy(self._traces), "native_events_by_model": native,
            "physical_cost_by_model": _physical_cost(native, self._traces),
            "logical_completed_node_cost_by_model": _logical_cost(r for c in self._conditions for r in c["nodes"].values()),
            "planned_terminal_conditions": 18, "complete_terminal_conditions": sum(c["status"] == "COMPLETE" for c in self._conditions),
            "planned_logical_calls": 60, "elapsed_seconds": time.perf_counter() - started,
            "resident_GPU_hours": None, "scores_computed": False, "scientific_admission": False,
            "GPU_admission": False, "automatic_retry": False, "scoring_rule": FINAL_TEXT_RULE}

    def run_matrix(self, payload, seeds, removed_upstream):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent use of a backend matrix is forbidden")
        started = time.perf_counter()
        try:
            if self._poisoned:
                raise RuntimeError("Operator failed previously; no silent retry, create an explicitly reviewed new attempt")
            p = validate_payload(payload)
            if not isinstance(seeds, (tuple, list)) or len(seeds) != 2 or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != 2:
                raise ValueError("Two distinct explicit nonnegative generation seeds required")
            if removed_upstream not in {"U", "V"}:
                raise ValueError("Exogenous upstream choice must be supplied, never selected from outcomes")
            if len({node_seed(p["id"], seed, node) for seed in seeds for node in NODES}) != 8:
                raise ValueError("Derived RNG stream collision")
            shards = partition_document_indices(p)
            self._matrix_id = uuid.uuid4().hex
            self._cache, self._call_ids, self._traces, self._conditions = {}, set(), [], []
            self._native_starts = {name: len(b.physical_calls) for name, b in self.backends.items()}
            try:
                for seed in seeds:
                    for condition in SCHEDULE:
                        self._condition(p, seed, condition, removed_upstream, shards)
                result = self._snapshot(p, seeds, removed_upstream, shards, started, "COMPLETE_UNSCORED")
                if result["complete_terminal_conditions"] != 18:
                    raise ValueError("Incomplete terminal coverage")
                if sum(c["calls"] for c in result["logical_completed_node_cost_by_model"].values()) != 60:
                    raise ValueError("Logical call accounting mismatch")
                return result
            except Exception as error:
                self._poisoned = True
                if self._conditions and self._conditions[-1]["status"] != "COMPLETE":
                    self._conditions[-1]["status"] = "INPUT_REJECTED" if isinstance(error, InputRejected) else "FAILED"
                partial = self._snapshot(p, seeds, removed_upstream, shards, started,
                                         "INPUT_REJECTED" if isinstance(error, InputRejected) else "INCOMPLETE_OR_INVALID")
                raise PathExecutionError(partial) from error
        finally:
            self._lock.release()
