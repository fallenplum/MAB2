"""New-schema, label-free MAB2 batch wrapper around the unchanged frozen executor.

Public modes: preflight, supervise, audit. Only supervise launches a CUDA child.
The private worker mode requires a live supervisor receipt/token. No scoring,
label-loading, downloads, auto-retry, or foreign-process control is implemented.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unicodedata
import uuid

sys.dont_write_bytecode = True
SCHEMA = "mab2_cycle2_collection_plan_v1"
EDITABLE = (0, 3, 4, 6, 8, 9)
REQUIRED_SOURCES = ("src/graph_credit/__init__.py", "src/graph_credit/executor.py", "src/graph_credit/backend.py")
EXCLUSION_KINDS = ("G1", "G2", "qualification", "literature_exposure", "official_test_ids")
SHA = re.compile(r"[0-9a-f]{64}\Z")


class ProtocolError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise ProtocolError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _unique_object(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, "Duplicate JSON member")
        result[k] = v
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            require(line.endswith("\n") and line.strip(), "Incomplete/blank durable ledger record")
            rows.append(json.loads(line, object_pairs_hook=_unique_object))
    return rows


def append_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(canonical(value)+"\n"); f.flush(); os.fsync(f.fileno())


def atomic_json(path, value, *, immutable=False):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+"."+uuid.uuid4().hex+".tmp")
    raw = (canonical(value)+"\n").encode("utf-8")
    try:
        with tmp.open("xb") as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        # Caller holds the only worker lease for this path. Immutable artifacts
        # cannot be replaced even by their original worker.
        require(not immutable or not p.exists(), "Refuse overwrite of immutable artifact")
        os.replace(tmp, p)
    finally:
        if tmp.exists(): tmp.unlink()  # only this function's own incomplete temp
    return hashlib.sha256(raw).hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative, "Expected project-relative POSIX path")
    parts = relative.split("/")
    require(all(p not in ("", ".", "..") for p in parts), "Unsafe path components")
    root = Path(root).resolve()
    p = root.joinpath(*parts)
    require(p.resolve().is_relative_to(root), "Path escapes declared root")
    cur = root
    for part in parts:
        cur = cur/part
        require(not cur.is_symlink(), "Symlink evidence paths are forbidden")
        if cur.exists():
            require(not getattr(cur.stat(), "st_file_attributes", 0) & 0x400, "Reparse evidence paths are forbidden")
    return p


def checked(root, descriptor):
    p = safe_path(root, descriptor["path"])
    require(SHA.fullmatch(descriptor["sha256"]) is not None and file_sha(p) == descriptor["sha256"], "Pinned artifact hash mismatch")
    return p


def question_text_sha(question):
    text = " ".join(unicodedata.normalize("NFKC", question).lower().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_menu():
    """A/B stay+six toggles. A->B cache uses only legal same-q same-seed state."""
    a = [1]*12
    for e in EDITABLE: a[e] = int(e in (0, 3, 6))
    b = list(a)
    for e in (1, 5, 10): b[e] = 0
    requests = []
    for name, mask in (("A", a), ("B", b)):
        requests.append({"name": name+"_stay", "mask": mask, "base": None if name == "A" else "A_stay", "background": name, "action": "stay"})
        for e in EDITABLE:
            flipped = list(mask); flipped[e] ^= 1
            requests.append({"name": name+"_toggle_"+str(e), "mask": flipped, "base": name+"_stay", "background": name, "action": "toggle", "edge": e})
    return requests


def episode_menu(permutation, complement):
    """Relabel template nodes; optionally complement only the six editable bits."""
    require(sorted(permutation) == [0, 1, 2, 3] and type(complement) is bool, "Invalid episode transform")
    edges = [(i, j) for i in range(4) for j in range(4) if i != j]
    mapping = {e: edges.index((permutation[i], permutation[j])) for e, (i, j) in enumerate(edges)}
    result = []
    for request in default_menu():
        row = dict(request); row["mask"] = [0]*12
        for e, bit in enumerate(request["mask"]):
            row["mask"][mapping[e]] = bit ^ int(complement and e in EDITABLE)
        # Stable names retain template edge indices; edge field is the actual transformed edge.
        if "edge" in row: row["edge"] = mapping[row["edge"]]
        result.append(row)
    return result


def unit_menu(plan, unit):
    if unit["phase"] == "qualification": return plan["qualification_menu"]
    episode = next(e for e in plan["episodes"] if e["episode"] == unit["episode"])
    return episode["menu"][:7] if unit["phase"] == "I" else episode["menu"]


def load_operator(source_root, source_manifest):
    require(all(p in source_manifest for p in REQUIRED_SOURCES), "Incomplete frozen operator sources")
    for rel, pin in source_manifest.items():
        checked(source_root, {"path": rel, "sha256": pin})
    package = safe_path(source_root, "src/graph_credit")
    name = "cycle2_frozen_graph_credit_"+digest(source_manifest)[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, package/"__init__.py", submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name+".executor"), importlib.import_module(name+".backend").HFBackend


def validate_menu(menu, module, *, qualification=False):
    names, normal_masks = set(), set()
    require(isinstance(menu, list) and menu, "Empty execution menu")
    for request in menu:
        name = request["name"]
        require(re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None and name not in names, "Invalid/duplicate request name")
        mask = module.validate_mask(request["mask"])
        require(request.get("base") is None or request["base"] in names, "Cache base must precede its request")
        if request.get("replay_of") is not None:
            require(qualification and request["replay_of"] in names and request.get("base") is None,
                    "Independent replay is qualification-only and must execute fully")
            original = next(r for r in menu if r["name"] == request["replay_of"])
            require(request["mask"] == original["mask"], "Replay mask differs from original")
        else:
            require(mask not in normal_masks, "Duplicate non-replay mask would change the information design")
            normal_masks.add(mask)
        names.add(name)


def estimated_calls(menu, module, rounds=2):
    masks = {r["name"]: r["mask"] for r in menu}
    return sum(4*rounds+1 if r.get("base") is None else len(module.affected_closure(masks[r["base"]], r["mask"], rounds)) for r in menu)


def validate_exclusions(questions, exclusion):
    require(exclusion.get("schema") == "mab2_cycle2_exclusion_v1", "Wrong exclusion schema")
    coverage = exclusion.get("coverage", {})
    require(all(coverage.get(k) == "ALL_KNOWN_RECORDED" for k in EXCLUSION_KINDS if k != "literature_exposure"), "Historical exclusion coverage is incomplete")
    require(coverage.get("literature_exposure") in ("ALL_KNOWN_RECORDED", "KNOWN_RECORDS_REVIEWED")
            and exclusion.get("unknown_exposure_beyond_recorded") is True, "Literature records must be reviewed while unknown exposure remains explicitly unverified")
    ids = set(exclusion["excluded_ids"])
    hashes = set(exclusion.get("excluded_normalized_question_sha256", []))
    prefixes = set(exclusion.get("excluded_normalized_question_sha256_prefix20", []))
    seen = set()
    for q in questions.values():
        h = question_text_sha(q["question"])
        require(q["id"] not in ids and h not in hashes and h[:20] not in prefixes, "New batch overlaps excluded exposure")
        require(h not in seen, "Duplicate normalized question under different IDs")
        seen.add(h)
        require(q.get("source_split") == "train", "New development batch must come from official train")
        require(q["id"] == "gsm8k_"+h[:20], "Question ID does not bind normalized source text")
    return {"new_unique_questions": len(questions), "excluded_id_count": len(ids), "overlap": 0,
            "exclusion_scope": "all known recorded exposures, not unknown external exposure", "sealed_test_content_read": False}


def preflight(source_root, data_root, plan_path, plan_sha, *, verify_runner=True):
    require(SHA.fullmatch(plan_sha) is not None and file_sha(plan_path) == plan_sha, "Plan is not the exact supplied frozen bytes")
    plan = read_json(plan_path)
    require(plan.get("schema") == SCHEMA and plan.get("status") == "FROZEN" and plan.get("labels_read") is False, "New cycle2 frozen plan required; G2 plans are rejected")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", plan["run_id"]) is not None, "Invalid run ID")
    if verify_runner: require(file_sha(__file__) == plan["runner_sha256"], "Runner source differs from frozen plan")
    source_root, data_root = Path(source_root).resolve(), Path(data_root).resolve()
    module, backend_class = load_operator(source_root, plan["source_manifest"])
    cfg = dict(plan["graph"])
    if "role_prompts" in cfg: cfg["role_prompts"] = tuple(cfg["role_prompts"])
    config = module.GraphConfig(**cfg)
    require(config.rounds == 2, "This declared A/B batch is two-round only")
    expected = module.Executor(SimpleNamespace(identity=plan["model"]["identity"]), config)
    require(expected.protocol_hash == plan["executor_protocol_hash"], "Operator/model/config hash differs from frozen contract")
    episodes = plan["episodes"]
    require([e["episode"] for e in episodes] == list(range(8)), "Expected eight ordered episodes")
    transforms = set()
    for ep in episodes:
        require(ep["seed"] == 3701+ep["episode"], "Wrong episode seed namespace")
        require(ep["permutation"] in [[(j+k)%4 for j in range(4)] for k in range(4)], "Expected cyclic node permutation")
        require(ep["menu"] == episode_menu(ep["permutation"], ep["complement"]), "Episode intervention differs from frozen transform")
        transforms.add((tuple(ep["permutation"]), ep["complement"]))
        validate_menu(ep["menu"], module)
    require(len(transforms) == 8, "Each cyclic permutation/complement pair must occur once")
    qmenu = plan["qualification_menu"]
    validate_menu(qmenu, module, qualification=True)
    require(any(r.get("replay_of") for r in qmenu), "Qualification requires explicit independent full replay")
    workers = plan["workers"]
    require(workers and [w["index"] for w in workers] == list(range(len(workers))), "Worker indices must be contiguous")
    require(len({w["physical_gpu"] for w in workers}) == len(workers), "One declared process per distinct physical GPU")
    require(all(type(w["physical_gpu"]) is int and w["physical_gpu"] >= 0 and w["resident_wall_cap_seconds"] > 15 and w["max_call_attempts"] > 0 for w in workers), "Invalid worker resource cap")
    require(sum(w["resident_wall_cap_seconds"] for w in workers) <= plan["budget"]["resident_wall_gpu_seconds"] <= 24*3600,
            "Worker reservations exceed 24 GPU-hour first-candidate cap")
    require(sum(w["max_call_attempts"] for w in workers) <= plan["budget"]["max_call_attempts"], "Per-worker call reservations exceed global cap")
    require(type(plan["max_unit_attempts"]) is int and 1 <= plan["max_unit_attempts"] <= 2, "At most one explicitly requested failed-unit retry")
    units, keys, qroles = plan["units"], set(), {}
    for u in units:
        require(re.fullmatch(r"[A-Za-z0-9_-]+", u["unit_id"]) is not None and u["unit_id"] not in keys, "Invalid/duplicate unit ID")
        keys.add(u["unit_id"])
        require(u["phase"] in ("qualification", "I", "C", "S") and 0 <= u["worker_index"] < len(workers), "Invalid unit role/worker")
        require(type(u["seed"]) is int and (u["seed"] == 3799 if u["phase"] == "qualification" else u["seed"] in range(3701, 3709)), "Wrong new seed namespace")
        role = (u["phase"], u.get("episode"))
        if u["question_id"] in qroles:
            require(u["phase"] == "qualification" and qroles[u["question_id"]][0] == "qualification", "Development questions cannot repeat across roles/episodes")
        qroles[u["question_id"]] = role
        if u["phase"] != "qualification":
            require(type(u.get("episode")) is int and 0 <= u["episode"] < 8 and u["seed"] == episodes[u["episode"]]["seed"], "Episode/seed mapping differs from frozen design")
        require("menu" not in u or u["menu"] == unit_menu(plan, u), "Per-unit menu disagrees with declared episode/phase")
    require(units and any(u["phase"] == "qualification" for u in units), "No qualification units")
    # Keep engineering fixtures possible through their own tests; production
    # must exactly match the 8*(I12,C12,S24) fresh development allocation.
    counts = {(ep, role): sum(u.get("episode") == ep and u["phase"] == role for u in units) for ep in range(8) for role in ("I", "C", "S")}
    require(all(counts[(ep, role)] == n for ep in range(8) for role, n in (("I", 12), ("C", 12), ("S", 24))), "Formal episode allocation is not 8*(12,12,24)")
    require(len({u["question_id"] for u in units if u["phase"] == "qualification"}) == 6, "Exactly six independent qualification questions are required")
    questions = {}
    question_file = checked(data_root, plan["questions"])
    with question_file.open(encoding="utf-8-sig") as f:
        for line in f:
            require(line.strip(), "Blank question row")
            q = json.loads(line, object_pairs_hook=_unique_object)
            require(set(q) <= {"id", "question", "context", "task", "split", "source_split"}, "Unexpected fields in label-free question payload")
            module.Executor.question_payload(q)
            require(q["id"] not in questions, "Duplicate question ID")
            require(module.digest(module.Executor.question_payload(q)) == plan["questions"]["payload_sha256"].get(q["id"]), "Question payload does not match frozen hash")
            questions[q["id"]] = q
    require(set(questions) == set(qroles) == set(plan["questions"]["payload_sha256"]), "Question file contains missing or extra units")
    exclusion = read_json(checked(data_root, plan["exclusions"]))
    exclusion_audit = validate_exclusions(questions, exclusion)
    model_path = safe_path(data_root, plan["model"]["path"])
    output = safe_path(data_root, plan["output_dir"])
    require(plan["output_dir"] == "results/cycle2/"+plan["run_id"], "New experiment needs its own cycle2 output namespace")
    return {"plan": plan, "plan_sha256": plan_sha, "source_root": source_root, "data_root": data_root,
            "module": module, "backend_class": backend_class, "config": config, "questions": questions,
            "model_path": model_path, "output": output, "exclusion_audit": exclusion_audit}


@contextmanager
def process_lease(path):
    """OS-held lock is released on process exit; stale files do not own a lease."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if os.fstat(f.fileno()).st_size == 0: f.write(b"0"); f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ProtocolError("Worker already has a live supervisor") from error
        try: yield
        finally:
            f.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def resident_used(records):
    starts, ends = {}, {}
    for row in records:
        target = starts if row["event"] == "START" else ends if row["event"] == "END" else None
        require(target is not None and row["attempt_id"] not in target, "Invalid/duplicate resident accounting event")
        target[row["attempt_id"]] = row
    require(set(ends) <= set(starts), "Resident END without START")
    require(set(starts) == set(ends), "Unclosed resident attempt: fail closed; preserve record for explicit recovery")
    require(all(row["resident_wall_seconds"] >= 0 for row in ends.values()), "Invalid resident duration")
    return sum(row["resident_wall_seconds"] for row in ends.values())


def require_live_linux_parent(expected_pid):
    """Linux worker dies if its own supervisor dies, including uncatchable SIGKILL.

    This affects only the newly created worker, never another project/process.
    The parent PID check after prctl closes the launch/death race.
    """
    require(sys.platform.startswith("linux"), "GPU worker supervision requires Linux parent-death signalling")
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, signal.SIGTERM, 0, 0, 0) == 0, "Cannot install worker parent-death signal")
    require(expected_pid == os.getppid(), "Budget supervisor is no longer the live parent")


def interrupt_supervisor(signum, frame):
    raise KeyboardInterrupt("Supervisor received signal "+str(signum))


class JournalBackend:
    """Journal every attempted call before entering the unchanged backend."""
    def __init__(self, backend, journal_path, maximum):
        self.backend, self.identity = backend, backend.identity
        self.journal_path, self.maximum = Path(journal_path), maximum
        rows = read_jsonl(self.journal_path)
        self.count = sum(r["event"] == "CALL_START" for r in rows)
        self.starts = {r["reservation_id"]: r for r in rows if r["event"] == "CALL_START"}
        self.success = {r["call_id"]: r for r in rows if r["event"] == "CALL_SUCCESS"}
        self.context = {}

    def encode(self, text): return self.backend.encode(text)
    def decode(self, tokens): return self.backend.decode(tokens)

    def generate(self, messages, seed, max_new_tokens, config):
        require(self.count < self.maximum, "Worker call-attempt cap reached")
        reservation = uuid.uuid4().hex
        started = time.monotonic()
        start_record = {"schema": "mab2_cycle2_call_journal_v1", "event": "CALL_START", "reservation_id": reservation,
                    "seed": seed, "prompt_sha256": digest(messages), "max_new_tokens": max_new_tokens,
                    "input_token_upper_bound": config.max_input_tokens, "started_unix": time.time(), **self.context}
        append_json(self.journal_path, start_record); self.starts[reservation] = start_record
        self.count += 1
        try:
            result = self.backend.generate(messages, seed, max_new_tokens, config)
        except BaseException as error:
            append_json(self.journal_path, {"event": "CALL_FAILED", "reservation_id": reservation,
                        "elapsed_wall_seconds": time.monotonic()-started, "error_type": type(error).__name__,
                        "output_tokens": None, "gpu_seconds": None, "cost_is_zero": False, **self.context})
            raise
        success_record = {"event": "CALL_SUCCESS", "reservation_id": reservation, "call_id": result.call_id,
                    "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                    "model_elapsed_seconds": result.elapsed_seconds, "gpu_seconds": result.gpu_seconds,
                    "gpu_peak_allocated_bytes": result.gpu_peak_allocated_bytes, **self.context}
        append_json(self.journal_path, success_record); self.success[result.call_id] = success_record
        return result


def restore_execution(module, value):
    nodes = {k: module.NodeRecord(**{**v, "generation": module.Generation(**v["generation"])}) for k, v in value["nodes"].items()}
    keys = ("execution_id", "question_id", "question_fingerprint", "seed", "protocol_hash", "closure", "answer", "elapsed_seconds")
    return module.ExecutionResult(**{k: value[k] for k in keys}, mask=tuple(value["mask"]), nodes=nodes)


def execution_signature(result):
    return {name: {"input_fingerprint": node.input_fingerprint, "wire_message": node.wire_message,
                   "raw_text": node.generation.text, "seed": node.seed, "input_tokens": node.generation.input_tokens,
                   "output_tokens": node.generation.output_tokens, "finish_reason": node.generation.finish_reason}
            for name, node in result.nodes.items()}


def validate_checkpoint(module, record, executor, question, unit, request, plan_sha, prior):
    require(record.get("schema") == "mab2_cycle2_graph_checkpoint_v1" and record["plan_sha256"] == plan_sha,
            "Checkpoint schema/protocol mismatch")
    require(record["unit_id"] == unit["unit_id"] and record["request"] == request and record["labels_read"] is False,
            "Checkpoint unit/menu identity mismatch")
    value = record["execution"]
    require(digest(value) == record["execution_value_sha256"], "Checkpoint content digest mismatch")
    result = restore_execution(module, value)
    require(value["physical_cost"] == result.cost() and value["logical_service_cost"] == result.cost(False), "Checkpoint derived cost mismatch")
    payload = module.Executor.question_payload(question)
    require(result.question_id == unit["question_id"] and result.question_fingerprint == module.digest(payload)
            and result.seed == unit["seed"] and result.protocol_hash == executor.protocol_hash and list(result.mask) == request["mask"],
            "Checkpoint execution binding mismatch")
    base = prior.get(request.get("base"))
    ordering = [f"r{r}:w{w}" for r in range(1, executor.config.rounds+1) for w in range(4)]+["aggregate"]
    require(set(result.nodes) == set(ordering), "Missing/extra checkpoint nodes")
    closure = module.affected_closure(base.mask, result.mask, executor.config.rounds) if base is not None else set(ordering)
    require(result.closure == sorted(closure), "Checkpoint affected closure mismatch")
    outputs = {}
    for name in ordering:
        node = result.nodes[name]
        messages, received, self_tokens = executor._messages(payload, name, result.mask, outputs)
        cap = executor.config.aggregate_tokens if name == "aggregate" else executor.config.message_tokens
        seed = module.derive_seed(unit["question_id"], unit["seed"], name)
        fingerprint = module.digest({"question": result.question_fingerprint, "node": name, "messages": messages,
                                     "seed": seed, "protocol": executor.protocol_hash, "max_new_tokens": cap})
        require(node.input_fingerprint == fingerprint and node.seed == seed and node.messages == messages
                and node.received_message_tokens == received and node.self_state_tokens == self_tokens
                and module.digest(node.integrity_value()) == node.output_digest, "Checkpoint node prompt/seed/integrity mismatch")
        if name not in closure:
            require(node.reused and node.source_execution_id == base.execution_id
                    and node.integrity_value() == base.nodes[name].integrity_value(), "Checkpoint cache lineage mismatch")
        else:
            require(not node.reused and node.source_execution_id is None, "Affected node was improperly cached")
        outputs[name] = node
    require(result.answer == result.nodes["aggregate"].parsed["answer"], "Checkpoint answer mismatch")
    return result


def validate_execution_journal(result, journal):
    """Every generated/cached node must refer to a successful pre-journalled call."""
    starts, success = journal.starts, journal.success
    for node in result.nodes.values():
        call = success.get(node.generation.call_id)
        require(call is not None and call["reservation_id"] in starts, "Checkpoint call missing successful ledger evidence")
        start = starts[call["reservation_id"]]
        require(start["seed"] == node.seed and start["prompt_sha256"] == digest(node.messages), "Checkpoint call prompt/seed disagrees with journal")
        require(all(call[k] == getattr(node.generation, k) for k in ("input_tokens", "output_tokens", "gpu_seconds", "gpu_peak_allocated_bytes")), "Checkpoint call cost disagrees with journal")


def collect_one_unit(context, executor, journal_backend, unit, worker_dir, *, resume=False, progress=None):
    plan, module = context["plan"], context["module"]
    folder = Path(worker_dir)/"units"/unit["unit_id"]
    folder.mkdir(parents=True, exist_ok=True)
    menu = unit_menu(plan, unit)
    question = context["questions"][unit["question_id"]]
    done = folder/"UNIT_COMPLETE.json"
    if done.exists():
        receipt = read_json(done)
        require(receipt["plan_sha256"] == context["plan_sha256"] and receipt["unit"] == unit, "Completed unit identity mismatch")
        prior = {}
        for request in menu:
            path = folder/(request["name"]+".json")
            require(file_sha(path) == receipt["graphs"][request["name"]], "Completed graph bytes changed")
            prior[request["name"]] = validate_checkpoint(module, read_json(path), executor, question, unit, request, context["plan_sha256"], prior)
            validate_execution_journal(prior[request["name"]], journal_backend)
        return receipt, False
    starts = list(folder.glob("attempt_*.START.json"))
    require(not starts or resume, "Partial unit requires explicit --resume; no silent inference retry")
    require(len(starts) < plan["max_unit_attempts"], "Unit retry limit exhausted")
    attempt = uuid.uuid4().hex
    atomic_json(folder/("attempt_"+attempt+".START.json"), {"unit": unit, "plan_sha256": context["plan_sha256"],
                "started_unix": time.time(), "explicit_resume": bool(starts)}, immutable=True)
    try:
        executions, graph_hashes, replay_checks = {}, {}, []
        for request in menu:
            if (context["output"]/"STOP_REQUESTED.json").exists(): raise ProtocolError("Persistent stop request")
            path = folder/(request["name"]+".json")
            if path.exists():
                result = validate_checkpoint(module, read_json(path), executor, question, unit, request, context["plan_sha256"], executions)
            else:
                journal_backend.context = {"unit_id": unit["unit_id"], "request_name": request["name"], "plan_sha256": context["plan_sha256"]}
                result = executor.execute(question, request["mask"], unit["seed"], base=executions.get(request.get("base")))
                value = result.to_dict()
                atomic_json(path, {"schema": "mab2_cycle2_graph_checkpoint_v1", "unit_id": unit["unit_id"],
                            "plan_sha256": context["plan_sha256"], "request": request, "execution": value,
                            "execution_value_sha256": digest(value), "labels_read": False, "quality": None,
                            "committed_unix": time.time()}, immutable=True)
            executions[request["name"]] = result
            validate_execution_journal(result, journal_backend)
            graph_hashes[request["name"]] = file_sha(path)
            if request.get("replay_of"):
                equal = execution_signature(result) == execution_signature(executions[request["replay_of"]])
                replay_checks.append(equal)
                require(equal, "Qualification full replay differs from cached execution")
            if progress: progress(unit, request["name"], len(graph_hashes))
        receipt = {"schema": "mab2_cycle2_unit_complete_v1", "unit": unit, "plan_sha256": context["plan_sha256"],
                   "executor_protocol_hash": executor.protocol_hash, "graphs": graph_hashes,
                   "logical_calls": sum(r.cost(False)["calls"] for r in executions.values()),
                   "physical_calls_referenced": sum(r.cost()["calls"] for r in executions.values()),
                   "replay_checks": replay_checks, "labels_read": False, "scored": False, "completed_unix": time.time()}
        atomic_json(done, receipt, immutable=True)
        atomic_json(folder/("attempt_"+attempt+".END.json"), {"status": "COMPLETE", "plan_sha256": context["plan_sha256"],
                    "unit_complete_sha256": file_sha(done)}, immutable=True)
        return receipt, True
    except BaseException as error:
        atomic_json(folder/("attempt_"+attempt+".END.json"), {"status": "FAILED", "error_type": type(error).__name__,
                    "plan_sha256": context["plan_sha256"], "failure_unix": time.time(), "silent_retry": False}, immutable=True)
        raise


def ledger_summary(worker_dir):
    worker_dir = Path(worker_dir)
    events = read_jsonl(worker_dir/"call_journal.jsonl")
    starts = {e["reservation_id"]: e for e in events if e["event"] == "CALL_START"}
    terminal = {e["reservation_id"]: e for e in events if e["event"] != "CALL_START"}
    require(len(starts)+len(terminal) == len(events) and set(terminal) <= set(starts), "Duplicate/unbound call journal records")
    known = {e["call_id"]: e for e in terminal.values() if e["event"] == "CALL_SUCCESS"}
    native = read_jsonl(worker_dir/"physical_calls.jsonl")
    native_success = {e["call_id"]: e for e in native if e.get("status") == "SUCCESS"}
    require(len({e["call_id"] for e in native}) == len(native), "Duplicate native physical-call IDs")
    for call_id, row in native_success.items():
        if call_id in known:
            require(all(row[k] == known[call_id][k] for k in ("input_tokens", "output_tokens", "gpu_seconds")), "Native/journal cost mismatch")
        else:
            known[call_id] = row  # completed native call before wrapper/process failure
    pending = set(starts)-set(terminal)
    failures = sum(e["event"] == "CALL_FAILED" for e in terminal.values())
    return {"attempted_calls": len(starts), "successful_native_or_journal_calls": len(known), "failed_wrapper_calls": failures,
            "pending_call_reservations": len(pending), "known_input_tokens": sum(e["input_tokens"] for e in known.values()),
            "known_output_tokens": sum(e["output_tokens"] for e in known.values()), "known_gpu_seconds": sum(e["gpu_seconds"] for e in known.values()),
            "unknown_failure_output_tokens": None if failures or pending else 0,
            "unknown_failure_gpu_seconds": None if failures or pending else 0,
            "output_token_reservation_upper_bound": sum(e["max_new_tokens"] for e in starts.values()),
            "resident_wall_budget_is_separate": True}


def run_worker(context, index, phase, *, resume=False):
    plan = context["plan"]; setting = plan["workers"][index]
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(setting["physical_gpu"]), "Worker visibility must contain exactly its declared physical GPU")
    root = context["output"]; worker_dir = root/("worker_"+str(index)); worker_dir.mkdir(parents=True, exist_ok=True)
    state = {"schema": "mab2_cycle2_worker_status_v1", "plan_sha256": context["plan_sha256"], "worker_index": index,
             "pid": os.getpid(), "physical_gpu": setting["physical_gpu"], "phase": phase, "labels_read": False,
             "scientific_completion": False, "started_unix": time.time(), "completed_units": 0}
    state_lock = threading.Lock(); stop_heartbeat = threading.Event()
    def update(**fields):
        with state_lock:
            state.update(fields, updated_unix=time.time())
            atomic_json(worker_dir/"STATUS.json", state)
    def heartbeat():
        while not stop_heartbeat.wait(30): update(heartbeat_unix=time.time())
    update(status="LOADING_MODEL")
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    try:
        if phase == "development":
            approval = read_json(root/"QUALIFICATION_ACCEPTED.json")
            require(approval.get("status") == "PASS" and approval.get("plan_sha256") == context["plan_sha256"]
                    and approval.get("executor_protocol_hash") == plan["executor_protocol_hash"], "New-run qualification has not been independently accepted")
            reports = approval.get("worker_reports", {})
            require(set(reports) == {str(w["index"]) for w in plan["workers"]}, "Qualification receipt omits a worker")
            for worker_id, pin in reports.items():
                report_path = root/("worker_"+worker_id)/"QUALIFICATION_COMPLETE.json"
                require(file_sha(report_path) == pin and read_json(report_path).get("status") == "COMPLETE_UNSCORED", "Qualification report binding changed")
        backend = context["backend_class"](context["model_path"], "cuda:0", worker_dir/"physical_calls.jsonl")
        require(backend.identity == plan["model"]["identity"], "Actual model/tokenizer/runtime differs from frozen identity")
        journal = JournalBackend(backend, worker_dir/"call_journal.jsonl", setting["max_call_attempts"])
        executor = context["module"].Executor(journal, context["config"])
        require(executor.protocol_hash == plan["executor_protocol_hash"], "Loaded executor hash changed")
        atomic_json(worker_dir/("RUNTIME_"+uuid.uuid4().hex+".json"), {"model_identity": backend.identity, "runtime": backend.runtime,
                    "pid": os.getpid(), "plan_sha256": context["plan_sha256"], "labels_read": False}, immutable=True)
        units = [u for u in plan["units"] if u["worker_index"] == index and ((u["phase"] == "qualification") == (phase == "qualification"))]
        require(units, "Worker has no assigned units in this phase")
        completed = []
        for u in units:
            update(status="GENERATING", current_unit=u["unit_id"])
            receipt, fresh = collect_one_unit(context, executor, journal, u, worker_dir, resume=resume,
                                             progress=lambda unit, request, n: update(current_unit=unit["unit_id"], current_request=request, committed_graphs_in_unit=n))
            completed.append({"unit_id": u["unit_id"], "sha256": file_sha(worker_dir/"units"/u["unit_id"]/"UNIT_COMPLETE.json")})
            update(completed_units=len(completed), last_unit_was_resumed=not fresh, call_attempts=journal.count)
        for rel, pin in plan["source_manifest"].items(): checked(context["source_root"], {"path": rel, "sha256": pin})
        report = {"schema": "mab2_cycle2_phase_complete_v1", "status": "COMPLETE_UNSCORED", "worker_index": index,
                  "phase": phase, "plan_sha256": context["plan_sha256"], "executor_protocol_hash": executor.protocol_hash,
                  "completed_units": completed, "ledger_summary": ledger_summary(worker_dir), "labels_read": False,
                  "scientific_completion": False, "completed_unix": time.time()}
        marker = worker_dir/("QUALIFICATION_COMPLETE.json" if phase == "qualification" else "DEVELOPMENT_COMPLETE.json")
        atomic_json(marker, report, immutable=True)
        update(status="COMPLETE_UNSCORED", generation_complete=True)
        return 0
    except BaseException as error:
        update(status="FAILED", error_type=type(error).__name__, error=str(error), automatic_retry=False)
        raise
    finally:
        stop_heartbeat.set(); thread.join(timeout=1)


def supervise(args, context):
    plan = context["plan"]; index = args.worker_index
    require(0 <= index < len(plan["workers"]), "Unknown worker index")
    worker_dir = context["output"]/("worker_"+str(index)); worker_dir.mkdir(parents=True, exist_ok=True)
    ledger = worker_dir/"resident_wall.jsonl"
    with process_lease(worker_dir/"SUPERVISOR.lock"):
        used = resident_used(read_jsonl(ledger))
        remaining = plan["workers"][index]["resident_wall_cap_seconds"]-used
        require(remaining > 15, "Resident GPU-wall budget exhausted")
        require(not (context["output"]/"STOP_REQUESTED.json").exists(), "Persistent stop requested")
        require(not (worker_dir/("QUALIFICATION_COMPLETE.json" if args.phase == "qualification" else "DEVELOPMENT_COMPLETE.json")).exists(), "Phase is already complete; use audit, do not reload GPU")
        attempt = uuid.uuid4().hex; token = uuid.uuid4().hex
        append_json(ledger, {"event": "START", "attempt_id": attempt, "supervisor_pid": os.getpid(),
                    "phase": args.phase, "plan_sha256": context["plan_sha256"], "started_unix": time.time(),
                    "reserved_remaining_seconds": remaining, "child_token_sha256": hashlib.sha256(token.encode()).hexdigest()})
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(plan["workers"][index]["physical_gpu"])
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"; env["MAB2_CYCLE2_CHILD_TOKEN"] = token
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): env[name] = "1"
        argv = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "worker", "--source-root", str(args.source_root),
                "--data-root", str(args.data_root), "--plan", str(args.plan), "--plan-sha256", args.plan_sha256,
                "--worker-index", str(index), "--phase", args.phase, "--supervisor-pid", str(os.getpid()), "--resident-attempt", attempt]
        if args.resume: argv.append("--resume")
        child = None; started = time.monotonic(); reason = "CHILD_EXIT"
        log_path = worker_dir/("attempt_"+attempt+".log")
        try:
            with log_path.open("xb") as log:
                child = subprocess.Popen(argv, env=env, cwd=args.source_root, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    elapsed = time.monotonic()-started
                    atomic_json(worker_dir/"SUPERVISOR_STATUS.json", {"plan_sha256": args.plan_sha256, "supervisor_pid": os.getpid(),
                                "child_pid": child.pid, "physical_gpu": plan["workers"][index]["physical_gpu"],
                                "updated_unix": time.time(), "phase": args.phase, "prior_resident_seconds": used,
                                "current_resident_seconds": elapsed, "budget_remaining_seconds": remaining-elapsed})
                    if elapsed >= remaining-12 or (context["output"]/"STOP_REQUESTED.json").exists():
                        reason = "BUDGET_STOP" if elapsed >= remaining-12 else "STOP_REQUESTED"
                        child.terminate()
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
                        break
                    time.sleep(min(1, max(.01, remaining-12-elapsed)))
                return_code = child.wait()
        except BaseException:
            reason = "SUPERVISOR_EXCEPTION"
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
            duration = time.monotonic()-started
            append_json(ledger, {"event": "END", "attempt_id": attempt, "plan_sha256": args.plan_sha256,
                        "resident_wall_seconds": duration, "child_pid": child.pid if child else None,
                        "return_code": child.returncode if child else None, "reason": reason, "ended_unix": time.time(),
                        "includes_model_load_and_failures": True, "resident_cap_exceeded": duration > remaining})
        require(reason == "CHILD_EXIT" and return_code == 0, "Worker stopped/failed; no automatic retry")
        return 0


def audit_run(context):
    root = context["output"]; plan = context["plan"]
    workers = []
    for w in plan["workers"]:
        folder = root/("worker_"+str(w["index"]))
        try: resident = resident_used(read_jsonl(folder/"resident_wall.jsonl")); closed = True
        except ProtocolError: resident, closed = None, False
        workers.append({"index": w["index"], "resident_wall_seconds": resident, "resident_accounting_closed": closed,
                        "calls": ledger_summary(folder), "qualification_complete": (folder/"QUALIFICATION_COMPLETE.json").exists(),
                        "development_complete": (folder/"DEVELOPMENT_COMPLETE.json").exists()})
    return {"schema": "mab2_cycle2_execution_audit_v1", "plan_sha256": context["plan_sha256"], "workers": workers,
            "resident_wall_gpu_seconds": sum(w["resident_wall_seconds"] for w in workers) if all(w["resident_accounting_closed"] for w in workers) else None,
            "labels_read": False, "scientific_completion": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "supervise", "worker", "audit"), required=True)
    p.add_argument("--source-root", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--plan-sha256", required=True)
    p.add_argument("--worker-index", type=int, default=0); p.add_argument("--phase", choices=("qualification", "development"), default="qualification")
    p.add_argument("--resume", action="store_true"); p.add_argument("--supervisor-pid", type=int)
    p.add_argument("--resident-attempt")
    args = p.parse_args()
    if args.mode == "supervise":
        signal.signal(signal.SIGTERM, interrupt_supervisor)
        signal.signal(signal.SIGINT, interrupt_supervisor)
    if args.mode == "worker": require_live_linux_parent(args.supervisor_pid)
    context = preflight(args.source_root, args.data_root, args.plan, args.plan_sha256)
    if args.mode == "preflight":
        print(canonical({"status": "PREFLIGHT_PASS_NO_MODEL_LOADED", "plan_sha256": args.plan_sha256,
                         "exclusions": context["exclusion_audit"], "units": len(context["plan"]["units"]),
                         "planned_fresh_calls": sum(estimated_calls(unit_menu(context["plan"], u), context["module"]) for u in context["plan"]["units"]), "labels_read": False}))
        return 0
    if args.mode == "audit": print(canonical(audit_run(context))); return 0
    if args.mode == "supervise": return supervise(args, context)
    require(args.supervisor_pid == os.getppid(), "Direct worker invocation bypasses the resident budget supervisor")
    records = read_jsonl(context["output"]/("worker_"+str(args.worker_index))/"resident_wall.jsonl")
    start = [r for r in records if r["event"] == "START" and r["attempt_id"] == args.resident_attempt]
    token = os.environ.pop("MAB2_CYCLE2_CHILD_TOKEN", "")
    require(len(start) == 1 and start[0]["supervisor_pid"] == args.supervisor_pid
            and start[0]["plan_sha256"] == args.plan_sha256 and hashlib.sha256(token.encode()).hexdigest() == start[0]["child_token_sha256"],
            "Missing live resident supervisor binding")
    return run_worker(context, args.worker_index, args.phase, resume=args.resume)


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(canonical({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "labels_read": False}), file=sys.stderr)
        raise
