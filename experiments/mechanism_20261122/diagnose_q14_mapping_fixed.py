"""Compare service single-forward outputs with worker reference using correct indices.

Worker sequence layout is source[0:3], old_relay[3:7], new_relay[7:11],
terminal[11:27]. The service path evaluates source[0:3], new_relay[0:4],
and terminal[0:16].
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def encode(tok, text):
    if getattr(tok, "chat_template", None):
        value = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
    else:
        value = tok.encode(text, add_special_tokens=True)
    if hasattr(value, "input_ids"):
        return list(value.input_ids)
    if isinstance(value, dict):
        return list(value["input_ids"])
    return list(value)


def forward_one(model, ids, device):
    input_ids = torch.as_tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(input_ids)
    positions = mask.long().cumsum(1) - 1
    with torch.inference_mode():
        return model(input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=False).logits[0, -1].float().cpu()


def prompts(row, n_options):
    menu = "\n".join(chr(65 + i) + ". " + x for i, x in enumerate(row["options"]))
    problem = row["stem"] + "\n" + menu
    group = "\n".join(chr(65 + i) + ". " + (row["options"][2 * i] + " OR " + row["options"][2 * i + 1] if n_options == 8 else row["options"][i]) for i in range(4))
    source_problem = row["stem"] + "\nHypotheses:\n" + group
    source = [source_problem + ("\nEvidence: " + fact if fact else "") + "\nChoose the best hypothesis. Return A, B, C or D." for fact in (row.get("first_fact", ""), "", row.get("control_fact", ""))]
    old_fact = row.get("second_fact", "")
    new_fact = row.get("second_fact_new", old_fact)
    facts = row.get("second_facts_new")
    if not isinstance(facts, list) or len(facts) != 4:
        facts = [old_fact, new_fact, old_fact, old_fact]
    interfaces = row.get("new_relay_interface")
    if not isinstance(interfaces, list) or len(interfaces) != 4:
        interfaces = ["", "", "", ""]
    relay = [source_problem + "\nEvidence: " + facts[index] + interfaces[index] + (("\nFirst researcher proposes hypothesis " + chr(65 + i) + ".") if i >= 0 else "") + "\nChoose the best hypothesis. Return A, B, C or D." for index, i in enumerate((-1, 0, 1, 2))]
    terminal = [problem + "\nFirst researcher proposes hypothesis " + chr(65 + i) + ".\nSecond researcher proposes hypothesis " + chr(65 + j) + ".\nReturn only the best answer letter." for i, j in itertools.product(range(4), repeat=2)]
    return source, relay, terminal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    raw_by_key = {row["key"]: row for row in raw}
    worker = [json.loads(line) for line in args.worker.read_text(encoding="utf-8").splitlines() if line.strip()][:args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True).to(args.device).eval()
    source_ids = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(4)]
    label_ids = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(len(raw[0]["options"]))]
    records = []
    for item in worker:
        row = raw_by_key[item["key"]]
        source, relay, terminal = prompts(row, len(row["options"]))
        service_seq = [encode(tok, x) for x in source + relay + terminal]
        logits = [forward_one(model, x, args.device) for x in service_seq]
        prob_diffs = []
        for i in range(3):
            observed = np.asarray(item["full"][str(i)], dtype=float)
            predicted = logits[i][source_ids].softmax(0).numpy()
            prob_diffs.append(float(np.max(np.abs(observed - predicted))))
        for i in range(4):
            observed = np.asarray(item["full"][str(7 + i)], dtype=float)
            predicted = logits[3 + i][source_ids].softmax(0).numpy()
            prob_diffs.append(float(np.max(np.abs(observed - predicted))))
        label_mismatches = 0
        for i in range(16):
            observed = int(item["full"][str(11 + i)])
            predicted = int(torch.stack([logits[7 + i][x] for x in label_ids]).argmax().item())
            label_mismatches += int(observed != predicted)
        records.append({"key": item["key"], "max_prob_diff": float(max(prob_diffs)), "mean_prob_diff": float(np.mean(prob_diffs)), "terminal_label_mismatches": label_mismatches, "service_lengths": list(map(len, service_seq)), "worker_lengths": item["tokens"]})
    result = {"schema": "q14_worker_service_mapping_fixed_v1", "n": len(records), "records": records, "max_prob_diff": max(x["max_prob_diff"] for x in records), "terminal_label_mismatches": sum(x["terminal_label_mismatches"] for x in records)}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

