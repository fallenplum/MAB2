"""CPU-only independent review. Reads only named qualification inputs.

No project code imports, model loading, inference, benchmark or original-gold reads.
Private locations are CLI arguments and are never written to the receipt.
"""
import argparse
import ast
import collections
import hashlib
import json
import math
import subprocess
from pathlib import Path


COMMIT = "b7434719eebb93eb41770bbc9af63222e3a47380"
PLAN_PIN = "279bec01c0a3516f3293990c8464deda78a240a6aed7bafb9ceafcd36016089c"
PREFIX = "experiments/cycle2_20261113/heterogeneous_qualification_v2/"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha(canonical(value).encode())


def readj(path):
    return json.loads(path.read_bytes())


def readl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source(repo, name):
    return subprocess.check_output(["git", "-c", "safe.directory=" + repo.as_posix(), "-C", str(repo), "show", COMMIT + ":" + name])


def literal_globals(raw):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(raw).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in {"ROLE_PROMPTS", "MESSAGE_INSTRUCTION", "AGGREGATOR_PROMPT"}}


def independent_closure(old, new, rounds):
    edges = [(s, t) for s in range(4) for t in range(4) if s != t]
    changed_dest = {edges[i][1] for i in range(12) if old[i] != new[i]}
    affected = {f"r{r}:w{w}" for r in range(2, rounds + 1) for w in changed_dest}
    for r in range(2, rounds + 1):
        for w in range(4):
            parents = {f"r{r-1}:w{w}"}
            parents |= {f"r{r-1}:w{s}" for i, (s, t) in enumerate(edges) if t == w and (old[i] or new[i])}
            if parents & affected:
                affected.add(f"r{r}:w{w}")
    if any(f"r{rounds}:w{w}" in affected for w in range(4)):
        affected.add("aggregate")
    return sorted(affected)


def cost(records):
    gens = [r["generation"] for r in records]
    return {"calls": len(records), "input_tokens": sum(g["input_tokens"] for g in gens),
            "output_tokens": sum(g["output_tokens"] for g in gens),
            "received_message_tokens": sum(r["received_message_tokens"] for r in records),
            "self_state_tokens": sum(r["self_state_tokens"] for r in records),
            "model_elapsed_seconds": sum(g["elapsed_seconds"] for g in gens),
            "gpu_seconds": sum(g["gpu_seconds"] for g in gens),
            "peak_allocated_bytes": max((g["gpu_peak_allocated_bytes"] for g in gens), default=0)}


def reconstruct_messages(question, node, records, mask, constants):
    text = question["question"]
    if question.get("context"):
        text += "\n\nSupplied context:\n" + question["context"]
    if node == "aggregate":
        incoming = [{"worker": w, "message": records[f"r3:w{w}"]["wire_message"]} for w in range(4)]
        return [{"role": "system", "content": constants["AGGREGATOR_PROMPT"]},
                {"role": "user", "content": text + "\n\nFinal worker messages:\n" + canonical(incoming)
                 + '\n\nRequired final format: {"answer":"<final answer>","uncertainty":"","evidence":"<brief check>"}. '
                 + 'Include exactly all 3 keys in that order. Never omit uncertainty, even when empty. '
                 + 'Return only the complete single-line JSON object.'}]
    rnd, worker = int(node[1]), int(node[-1])
    if rnd == 1:
        text += "\n\nRound 1: work independently. No external messages are available."
    else:
        own = records[f"r{rnd-1}:w{worker}"]["wire_message"]
        edges = [(s, t) for s in range(4) for t in range(4) if s != t]
        incoming = [{"worker": s, "message": records[f"r{rnd-1}:w{s}"]["wire_message"]}
                    for i, (s, t) in enumerate(edges) if t == worker and mask[i]]
        text += f"\n\nRound {rnd}. Your previous state:\n{own}"
        text += "\n\nExternal messages (ordered by source worker):\n" + (canonical(incoming) if incoming else "[] (no external messages)")
        text += "\nRecheck your candidate using the question, your previous state and these messages."
    return [{"role": "system", "content": constants["ROLE_PROMPTS"][worker] + " " + constants["MESSAGE_INSTRUCTION"]},
            {"role": "user", "content": text}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--qwen-tokenizer", type=Path)
    p.add_argument("--mistral-tokenizer", type=Path)
    a = p.parse_args()
    checks = []
    def check(name, condition, count=1):
        checks.append({"check": name, "pass": bool(condition), "count": count})
        if not condition:
            raise AssertionError(name)

    plan_path = a.private / "PRIVATE_PLAN_V2B.json"
    plan = readj(plan_path)
    check("private_plan_exact_sha256", sha(plan_path.read_bytes()) == PLAN_PIN)
    check("source_commit", plan["source_commit"] == COMMIT)
    git_bytes = {name: source(a.repo, name) for name in plan["source_manifest"]}
    source_bytes = {name: (a.repo / name).read_bytes() for name in plan["source_manifest"]}
    check("deployed_source_byte_pins_match_local_qualification_files", all(sha(raw) == plan["source_manifest"][name] for name, raw in source_bytes.items()), len(source_bytes))
    line_ending_mapping = [{"file": name, "git_blob_sha256": sha(git_bytes[name]), "deployed_sha256": sha(raw),
                           "git_blob_LF_normalized_sha256": sha(git_bytes[name].replace(b"\r\n", b"\n")),
                           "deployed_LF_normalized_sha256": sha(raw.replace(b"\r\n", b"\n")),
                           "git_blob_bytes": len(git_bytes[name]), "deployed_bytes": len(raw),
                           "git_blob_CRLF_count": git_bytes[name].count(b"\r\n"), "deployed_CRLF_count": raw.count(b"\r\n"),
                           "byte_identity": "exact" if raw == git_bytes[name] else "LF_CRLF_ONLY"}
                          for name, raw in source_bytes.items()]
    check("frozen_git_sources_match_after_only_CRLF_to_LF_normalization", all(raw.replace(b"\r\n", b"\n") == git_bytes[name].replace(b"\r\n", b"\n") for name, raw in source_bytes.items()), len(source_bytes))
    deployment = readj(a.private / "DEPLOY_RECEIPT_V2B.json")
    uploaded = {row["remote"].removeprefix(plan["code_root"] + "/"): row for row in deployment["uploads"]}
    check("deployed_source_receipt_pins_and_lengths", all(uploaded[name]["sha256"] == sha(raw) and uploaded[name]["bytes"] == len(raw) for name, raw in source_bytes.items()), len(source_bytes))
    check("deployed_plan_receipt_pin", uploaded["PRIVATE_PLAN_V2B.json"]["sha256"] == PLAN_PIN)
    protocol = json.loads(source_bytes[PREFIX + "PROTOCOL.json"])
    constants = literal_globals(source_bytes["src/graph_credit/executor.py"])
    run = a.private / "server_closed_v2"
    sync = readj(a.private / "CLOSED_SYNC_RECEIPT.json")
    prior_replay = readj(a.private / "INDEPENDENT_OUTPUT_REPLAY.json")
    closure_receipt = readj(a.private / "ROOT_CLOSURE_RECEIPT.json")
    check("closed_sync_exact_file_hashes_and_lengths", all((run / row["name"]).stat().st_size == row["bytes"] and sha((run / row["name"]).read_bytes()) == row["sha256"] for row in sync["files"]), len(sync["files"]))
    check("root_replay_input_pins", all(sha((run / name).read_bytes()) == pin for name, pin in prior_replay["source_pins"].items()), len(prior_replay["source_pins"]))
    ident = readj(run / "IDENTITY.json")
    identity = ident["backend"]
    check("worker_and_aggregator_route_matches_protocol", identity["worker_models"] == protocol["model_route"]["workers"] and identity["aggregator_model"] == protocol["model_route"]["aggregate"])
    model_ids = identity["model_identities"]
    for name, spec in plan["models"].items():
        actual = model_ids[name]["model_files"]
        check("model_identity_manifest_" + name, all(actual[k] == v for k, v in spec["model_files"].items()) and set(actual) - set(spec["model_files"]) <= set(spec["allowed_metadata_files"]), len(actual))
        check("backend_source_identity_" + name, model_ids[name]["backend_code_sha256"] == plan["source_manifest"]["src/graph_credit/backend.py"])
    check("routing_adapter_source_identity", identity["routing_adapter_code_sha256"] == plan["source_manifest"][PREFIX + "routed_executor.py"])
    config = {"rounds": 2, "n_workers": 4, "message_tokens": 128, "aggregate_tokens": 256,
              "do_sample": True, "temperature": .7, "top_p": .9, "role_prompts": constants["ROLE_PROMPTS"],
              "max_input_tokens": 8192, "executor_version": "unrolled_closed_input_v3"}
    config.update(protocol["graph"])
    edge_order = [(s, t) for s in range(4) for t in range(4) if s != t]
    phash = digest({"config": config, "backend": identity,
                    "executor_code_sha256": plan["source_manifest"]["src/graph_credit/executor.py"],
                    "message_instruction": constants["MESSAGE_INSTRUCTION"],
                    "aggregator_prompt": constants["AGGREGATOR_PROMPT"],
                    "edge_order": edge_order, "serialization": "canonical_json_v1"})
    check("independently_reconstructed_protocol_hash", phash == ident["protocol_hash"])
    routes = readl(run / "route_calls.jsonl")
    physical = {name: readl(run / (name + "_physical_calls.jsonl")) for name in model_ids}
    all_physical = [r for rows in physical.values() for r in rows]
    check("64_unique_successful_physical_calls", len(all_physical) == 64 and len({r["call_id"] for r in all_physical}) == 64 and all(r["status"] == "SUCCESS" for r in all_physical), 64)
    route_by_call = {r["call_id"]: r for r in routes}
    physical_by_call = {r["call_id"]: (name, r) for name, rows in physical.items() for r in rows}
    check("route_physical_bijection", len(routes) == 64 and len(route_by_call) == 64 and set(route_by_call) == set(physical_by_call), 64)
    expected_nodes = [f"r{r}:w{w}" for r in range(1, 4) for w in range(4)] + ["aggregate"]
    model_for = lambda n: identity["aggregator_model"] if n == "aggregate" else identity["worker_models"][int(n[-1])]
    tokens = {}
    if a.qwen_tokenizer and a.mistral_tokenizer:
        import tokenizers
        for name, path in (("qwen", a.qwen_tokenizer), ("mistral", a.mistral_tokenizer)):
            check("local_tokenizer_json_pin_" + name, sha(path.read_bytes()) == model_ids[name]["model_files"]["tokenizer.json"]["sha256"])
            tokens[name] = tokenizers.Tokenizer.from_file(str(path))
        tokenizer_version = tokenizers.__version__
    else:
        tokenizer_version = None
    token_len = lambda model, text: len(tokens[model].encode(text, add_special_tokens=False).ids)
    executed = []
    billed_ids = []
    semantic_equalities = 0
    integrity_checks = 0
    count_direction = 0
    source_sum_mismatches = 0
    wire_max = collections.defaultdict(int)
    inputs_max = collections.defaultdict(int)
    field_keys = ["node", "input_fingerprint", "seed", "messages", "generation", "wire_message", "parsed", "message_metadata", "received_message_tokens", "self_state_tokens"]
    closure = independent_closure(protocol["old_mask"], protocol["new_mask"], 3)
    for question in protocol["synthetic_inputs"]:
        qid = question["id"]
        qfp = digest({"id": qid, "question": question["question"], "context": question.get("context", "")})
        items = {tag: readj(run / (qid + "_" + tag + ".json")) for tag in ("base", "changed", "fresh", "cache")}
        for tag, item in items.items():
            check("execution_header_" + qid + "_" + tag, set(item["nodes"]) == set(expected_nodes) and item["question_fingerprint"] == qfp and item["protocol_hash"] == phash and item["seed"] == protocol["seed"])
            expected_mask = protocol["old_mask"] if tag == "base" else protocol["new_mask"]
            expected_closure = closure if tag == "changed" else ([] if tag == "cache" else sorted(expected_nodes))
            check("mask_and_closure_" + qid + "_" + tag, item["mask"] == expected_mask and item["closure"] == expected_closure)
            check("reuse_set_" + qid + "_" + tag, sorted(n for n, r in item["nodes"].items() if not r["reused"]) == expected_closure)
            records = list(item["nodes"].values())
            check("physical_cost_" + qid + "_" + tag, cost([r for r in records if not r["reused"]]) == item["physical_cost"])
            check("logical_service_cost_" + qid + "_" + tag, cost(records) == item["logical_service_cost"] and item["logical_service_cost"]["calls"] == 13)
            check("cache_count_" + qid + "_" + tag, item["cache_reuse_count"] == sum(r["reused"] for r in records))
            check("aggregate_answer_link_" + qid + "_" + tag, item["answer"] == item["nodes"]["aggregate"]["parsed"]["answer"])
            for node, rec in item["nodes"].items():
                assert rec["messages"] == reconstruct_messages(question, node, item["nodes"], item["mask"], constants)
                seed = int(digest([qid, protocol["seed"], node])[:16], 16) % (2**31 - 1)
                cap = config["aggregate_tokens"] if node == "aggregate" else config["message_tokens"]
                fp = digest({"question": qfp, "node": node, "messages": rec["messages"], "seed": seed, "protocol": phash, "max_new_tokens": cap})
                assert rec["input_fingerprint"] == fp and rec["seed"] == seed and rec["node"] == node
                assert digest({k: rec[k] for k in field_keys}) == rec["output_digest"]
                gen = rec["generation"]
                # All observed outputs are untrimmed valid JSON, so no production normalizer is imported.
                parsed = json.loads(gen["text"])
                assert set(parsed) == {"answer", "evidence", "uncertainty"}
                assert all(isinstance(v, str) for v in parsed.values())
                assert parsed == rec["parsed"] and canonical(parsed) == rec["wire_message"]
                assert rec["message_metadata"]["format_ok"] and not rec["message_metadata"]["message_trimmed"]
                assert rec["message_metadata"]["format_error"] is None
                assert 0 <= gen["input_tokens"] <= config["max_input_tokens"] and 0 <= gen["output_tokens"] <= cap
                assert 0 <= rec["message_metadata"]["message_tokens"] <= cap
                name, physical_call = physical_by_call[gen["call_id"]]
                route = route_by_call[gen["call_id"]]
                assert name == model_for(node) == route["model"] and route["node"] == node
                assert route["model_identity_sha256"] == digest(model_ids[name])
                assert route["status"] == "SUCCESS" and route["seed"] == seed and route["max_new_tokens"] == cap
                assert route["prompt_sha256"] == physical_call["prompt_sha256"] == digest(rec["messages"])
                assert physical_call["generation_config"]["max_new_tokens"] == cap
                assert all(physical_call[k] == gen[k] for k in ("call_id", "input_tokens", "output_tokens", "elapsed_seconds", "gpu_seconds", "gpu_peak_allocated_bytes", "finish_reason"))
                assert all(route[k] == gen[k] for k in ("call_id", "input_tokens", "output_tokens"))
                if rec["reused"]:
                    parent = items["base"] if tag == "changed" else items["changed"]
                    assert rec["source_execution_id"] == parent["execution_id"]
                    assert rec["output_digest"] == parent["nodes"][node]["output_digest"]
                else:
                    assert rec["source_execution_id"] is None
                    billed_ids.append(gen["call_id"])
                if node == "aggregate":
                    sources = [f"r3:w{w}" for w in range(4)]
                    own = None
                else:
                    rnd, worker = int(node[1]), int(node[-1])
                    sources = [f"r{rnd-1}:w{s}" for i, (s, t) in enumerate(edge_order) if t == worker and item["mask"][i]] if rnd > 1 else []
                    own = f"r{rnd-1}:w{worker}" if rnd > 1 else None
                sender_sum = sum(item["nodes"][s]["message_metadata"]["message_tokens"] for s in sources)
                if sender_sum != rec["received_message_tokens"]:
                    source_sum_mismatches += 1
                if tokens:
                    assert token_len(name, rec["wire_message"]) == rec["message_metadata"]["message_tokens"]
                    assert sum(token_len(name, item["nodes"][s]["wire_message"]) for s in sources) == rec["received_message_tokens"]
                    assert (token_len(name, item["nodes"][own]["wire_message"]) if own else 0) == rec["self_state_tokens"]
                    count_direction += 3
                wire_max[name] = max(wire_max[name], rec["message_metadata"]["message_tokens"])
                inputs_max[name] = max(inputs_max[name], gen["input_tokens"])
                integrity_checks += 1
            executed.append({"case": qid.removeprefix("synthetic_hetero_"), "variant": tag, "physical_calls": item["physical_cost"]["calls"], "logical_calls": item["logical_service_cost"]["calls"], "cache_reused": item["cache_reuse_count"]})
        semantic_fields = ("messages", "input_fingerprint", "seed", "wire_message", "parsed", "message_metadata", "received_message_tokens", "self_state_tokens")
        for n in expected_nodes:
            changed, fresh = items["changed"]["nodes"][n], items["fresh"]["nodes"][n]
            assert all(changed[k] == fresh[k] for k in semantic_fields)
            assert changed["generation"]["text"] == fresh["generation"]["text"]
            semantic_equalities += len(semantic_fields) + 1
    check("all_node_integrity_seed_fingerprint_raw_output_physical_route_checks", integrity_checks == 104, integrity_checks)
    check("closed_input_prompts_reconstructed_from_only_synthetic_contract_and_permitted_messages", integrity_checks == 104, integrity_checks)
    check("all_physical_generation_ids_billed_once", len(billed_ids) == 64 and len(set(billed_ids)) == 64 and set(billed_ids) == set(physical_by_call), 64)
    check("changed_vs_fresh_semantic_equalities", semantic_equalities == 234, semantic_equalities)
    if tokens:
        check("independent_low_level_tokenizer_sender_receiver_self_state", count_direction == 312, count_direction)
    resident = readj(run / "RESIDENT_STATUS.json")
    qualification = readj(run / "QUALIFICATION.json")
    check("closed_resident_receipt", resident["status"] == "CLOSED" and resident["returncode"] == 0 and resident["reason"] == "CHILD_EXIT" and not resident["reservation_exceeded"] and resident["plan_sha256"] == PLAN_PIN)
    check("GPU_occupancy_cost_recomputed", resident["gpu_hours"] == resident["resident_seconds"] / 3600 and resident["resident_seconds"] <= 1770)
    check("closed_cost_agreement", resident == sync["cost"] and resident["gpu_hours"] == closure_receipt["GPU_hours"] == prior_replay["resident_gpu_hours"])
    check("qualification_only_64_calls_0_new_formal_rounds", qualification["inference_calls"] == 64 and qualification["synthetic_only"] and qualification["benchmark_questions_read"] == 0 and qualification["formal_candidate_rounds_started"] == 0 and qualification["no_accuracy_or_novelty_claim"])
    total_input = sum(r["input_tokens"] for r in all_physical)
    total_output = sum(r["output_tokens"] for r in all_physical)
    total_gpu = sum(r["gpu_seconds"] for r in all_physical)
    check("root_replay_aggregate_cost_agreement", total_input == prior_replay["input_tokens"] and total_output == prior_replay["output_tokens"] and math.isclose(total_gpu, prior_replay["CUDA_event_seconds"], abs_tol=1e-9))
    receipt = {"status": "PASS_SCOPED_SYNTHETIC_ENGINEERING_OUTPUT_AUDIT", "reviewer": "separate_codex_agent_independent_stdlib_replay", "source_commit": COMMIT,
        "private_plan_sha256": PLAN_PIN, "review_script_sha256": sha(Path(__file__).read_bytes()),
        "source_pins": plan["source_manifest"], "git_to_deployed_line_ending_mapping": line_ending_mapping, "input_artifact_pins": {row["name"]: row["sha256"] for row in sync["files"]},
        "input_receipt_pins": {n: sha((a.private / n).read_bytes()) for n in ("CLOSED_SYNC_RECEIPT.json", "INDEPENDENT_OUTPUT_REPLAY.json", "ROOT_CLOSURE_RECEIPT.json", "DEPLOY_RECEIPT_V2B.json")},
        "checks": checks, "node_records": integrity_checks, "execution_summary": executed,
        "physical_calls_by_model": {k: len(v) for k, v in physical.items()}, "input_tokens": total_input, "output_tokens": total_output,
        "CUDA_event_seconds": total_gpu, "resident_seconds": resident["resident_seconds"], "resident_GPU_hours": resident["gpu_hours"],
        "sender_wire_max_observed_by_model": dict(wire_max), "prompt_tokens_max_observed_by_model": dict(inputs_max),
        "receiver_count_differs_from_sender_sum_records": source_sum_mismatches,
        "independent_tokenizer_json_recount": bool(tokens), "tokenizers_version_for_CPU_recount": tokenizer_version,
        "recount_scope": "Tokenizer JSON exact bytes; encode(add_special_tokens=False); no original HF chat-template or model generation replay" if tokens else "Static source audit plus existing count ledger only; no independent tokenization",
        "new_GPU_calls": 0, "model_weight_files_opened": 0, "benchmark_or_original_gold_reads": 0, "formal_rounds_increment": 0,
        "reviewer_correction": "Initial audit assumed Git blob bytes equalled deployed checkout bytes and failed source pins. Corrected only after checking exact deployment and working-file pins plus equality under CRLF-to-LF normalization; mapping retained. No production artifacts changed.",
        "limitations": ["Two synthetic tasks, one seed, one fixed route and one edge deletion only.", "Full HF chat-template input count and generated-token ID count are cross-linked to physical logs, not independently reconstructed from token IDs.", "Model file identity verified through frozen manifests and hash chain; weights deliberately not reread.", "Root closure formal_rounds is cumulative project state; qualification increments zero.", "Does not establish real-task quality, long-context capacity, stability under failures, heterogeneous improvement or scientific admission."]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "INDEPENDENT_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("status", "node_records", "physical_calls_by_model", "input_tokens", "output_tokens", "resident_GPU_hours", "independent_tokenizer_json_recount", "receiver_count_differs_from_sender_sum_records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
