"""Actual model-forward service benchmark for graph-credit revalidation.

The benchmark measures model forward calls, input/output tokens, and elapsed
wall time for matched stale, TCR, answer-region, and cold-restart policies.
Cached source/base/terminal outputs are reused by warm policies.  Cold restart
recomputes source, new relay, and terminal logits from the same model.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_region_only import choose as answer_choose
from answer_region_only import parse_rows, radii
from answer_safe_arcc import values


def encode(tok, text):
    if getattr(tok, "chat_template", None):
        encoded = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        encoded = tok.encode(text, add_special_tokens=True)
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    return list(encoded)


def forward_one(model, tok, ids, pad_id, device):
    input_ids = torch.as_tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = attention_mask.long().cumsum(1) - 1
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    return out.logits[0, -1].float().cpu()


def build_prompts(row, n_options):
    menu = "\n".join(chr(65 + i) + ". " + v for i, v in enumerate(row["options"]))
    problem = row["stem"] + "\n" + menu
    group_menu = "\n".join(
        chr(65 + i) + ". " + (
            row["options"][2 * i] + " OR " + row["options"][2 * i + 1]
            if n_options == 8
            else row["options"][i]
        )
        for i in range(4)
    )
    source_problem = row["stem"] + "\nHypotheses:\n" + group_menu
    source = [
        source_problem
        + ("\nEvidence: " + fact if fact else "")
        + "\nChoose the best hypothesis. Return A, B, C or D."
        for fact in (row.get("first_fact", ""), "", row.get("control_fact", ""))
    ]
    old_fact = row.get("second_fact", "")
    new_fact = row.get("second_fact_new", old_fact)
    new_facts = row.get("second_facts_new")
    if not isinstance(new_facts, list) or len(new_facts) != 4:
        new_facts = [old_fact, new_fact, old_fact, old_fact]
    interfaces = row.get("new_relay_interface")
    if not isinstance(interfaces, list) or len(interfaces) != 4:
        interfaces = ["", "", "", ""]
    new_relay = [
        source_problem
        + "\nEvidence: "
        + new_facts[index]
        + interfaces[index]
        + ("\nFirst researcher proposes hypothesis " + chr(65 + i) + "." if i >= 0 else "")
        + "\nChoose the best hypothesis. Return A, B, C or D."
        for index, i in enumerate((-1, 0, 1, 2))
    ]
    terminals = [
        problem
        + "\nFirst researcher proposes hypothesis "
        + chr(65 + i)
        + ".\nSecond researcher proposes hypothesis "
        + chr(65 + j)
        + ".\nReturn only the best answer letter."
        for i, j in itertools.product(range(4), repeat=2)
    ]
    return source, new_relay, terminals


def probs_from_logits(logits, source_ids):
    return logits[source_ids].softmax(0).numpy()


def labels_from_logits(logits, label_ids):
    return int(torch.stack([logits[v[0]] for v in label_ids]).argmax().item())


def score_action(row, source, base, relay, labels, gold):
    pred = values(source, base, relay, labels)
    truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
    return {
        "decision_exact": float(np.array_equal(pred > 0.01, truth > 0.01)),
        "answer_action_exact": float(int(pred.argmax()) == int(truth.argmax())),
        "answer_accuracy": float(int(pred.argmax()) == int(gold)),
    }


def run_method(name, rows, raw_by_key, tok, model, pad_id, device, radii_all):
    n_options = len(raw_by_key[rows[0]["key"]]["options"])
    source_ids = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(4)]
    label_ids = [tok.encode(chr(65 + i), add_special_tokens=False) for i in range(n_options)]
    assert all(len(v) == 1 for v in label_ids)
    total_calls = 0
    input_tokens = 0
    output_tokens = 0
    scores = []
    wall_start = time.perf_counter()
    for parsed in rows:
        raw = raw_by_key[parsed["key"]]
        source_prompts, relay_prompts, terminal_prompts = build_prompts(raw, n_options)
        source_ids_enc = [encode(tok, p) for p in source_prompts]
        relay_ids_enc = [encode(tok, p) for p in relay_prompts]
        terminal_ids_enc = [encode(tok, p) for p in terminal_prompts]
        if name == "stale":
            selected = []
            source = parsed["new_source"]
            base = parsed["base"]
            relay = parsed["old_relay"].copy()
            labels = parsed["labels"].copy()
        elif name == "tcr_guard":
            selected = list(parsed.get("changed_rows", []))
            source = parsed["new_source"]
            base = parsed["base"]
            relay = parsed["old_relay"].copy()
            labels = parsed["labels"].copy()
            for index in selected:
                logits = forward_one(model, tok, relay_ids_enc[int(index)], pad_id, device)
                relay[int(index)] = probs_from_logits(logits, source_ids)
                total_calls += 1
                input_tokens += len(relay_ids_enc[int(index)])
                output_tokens += 1
        elif name in ("answer_region_q1", "calibrated_route"):
            policy_row = (
                {**parsed, "changed_rows": [0, 1, 2, 3]}
                if name == "answer_region_q1"
                else parsed
            )
            selected = answer_choose(policy_row, radii_all)
            source = parsed["new_source"]
            base = parsed["base"]
            relay = parsed["old_relay"].copy()
            labels = parsed["labels"].copy()
            for index in selected:
                logits = forward_one(model, tok, relay_ids_enc[int(index)], pad_id, device)
                relay[int(index)] = probs_from_logits(logits, source_ids)
                total_calls += 1
                input_tokens += len(relay_ids_enc[int(index)])
                output_tokens += 1
        elif name == "cold_restart":
            source = np.zeros((3, 4), dtype=float)
            for index, ids in enumerate(source_ids_enc):
                logits = forward_one(model, tok, ids, pad_id, device)
                source[index] = probs_from_logits(logits, source_ids)
                total_calls += 1
                input_tokens += len(ids)
                output_tokens += 1
            base = source[2].copy()
            relay = np.zeros((4, 4), dtype=float)
            for index, ids in enumerate(relay_ids_enc):
                logits = forward_one(model, tok, ids, pad_id, device)
                relay[index] = probs_from_logits(logits, source_ids)
                total_calls += 1
                input_tokens += len(ids)
                output_tokens += 1
            labels = np.zeros((4, 4), dtype=int)
            for index, ids in enumerate(terminal_ids_enc):
                logits = forward_one(model, tok, ids, pad_id, device)
                labels[index // 4, index % 4] = labels_from_logits(logits, label_ids)
                total_calls += 1
                input_tokens += len(ids)
                output_tokens += 1
            source = source[1]
            selected = list(range(4))
        else:
            raise ValueError(name)
        scores.append(score_action(parsed, source, base, relay, labels, parsed["gold"]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - wall_start
    result = {
        "method": name,
        "n": len(rows),
        "forward_calls": int(total_calls),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "wall_seconds": float(elapsed),
        "mean_calls": float(total_calls / len(rows)),
        "mean_input_tokens": float(input_tokens / len(rows)),
        "mean_output_tokens": float(output_tokens / len(rows)),
        "mean_wall_seconds": float(elapsed / len(rows)),
    }
    for key in scores[0]:
        result[key] = float(np.mean([x[key] for x in scores]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantile", type=float, default=1.0)
    parser.add_argument("--route", choices=("q1", "calibrated"), default="q1")
    parser.add_argument("--methods", default="", help="comma-separated subset of methods")
    args = parser.parse_args()
    raw_rows = json.loads(args.input.read_text(encoding="utf-8"))
    raw_by_key = {row["key"]: row for row in raw_rows}
    parsed = parse_rows(args.worker, args.gold)
    for row in parsed:
        row["gold"] = int(args.gold.read_text(encoding="utf-8") and row.get("gold", 0))
    # q=1 all-row radius is calibrated on the first 60% of questions.
    calibration = []
    for row in parsed:
        if args.route == "q1":
            row["changed_rows"] = [0, 1, 2, 3]
        if int(row["key"][:8], 16) % 5 < 3:
            calibration.append(row)
    radius = radii(calibration, args.quantile)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else tok.unk_token_id
    if pad_id is None:
        pad_id = 0
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device).eval()
    # Restore the worker-provided change mask for TCR and keep all-row only in ARCC.
    for row in parsed:
        item = next(x for x in (json.loads(line) for line in args.worker.read_text(encoding="utf-8").splitlines() if line.strip()) if x["key"] == row["key"])
        row["changed_rows"] = item.get("changed_rows", [])
    results = []
    methods = ("stale", "tcr_guard", "answer_region_q1", "cold_restart")
    if args.route == "calibrated":
        methods = methods + ("calibrated_route",)
    if args.methods:
        methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    for method in methods:
        results.append(run_method(method, parsed, raw_by_key, tok, model, pad_id, device, radius))
    out = {"schema": "mab2_actual_service_forward_v1", "input": args.input.name,
           "worker": args.worker.name, "model": str(args.model), "results": results,
           "radius_q1_all": {str(k): float(v) for k, v in radius.items()}}
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()

