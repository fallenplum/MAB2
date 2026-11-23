"""Qwen-0.5B real-service runner for the P2 multi-hop DAG panel.

The runner keeps a paid old transcript per row, then measures post-edit
incremental calls for each controller. Sealed gold is used only after all
online actions have been selected. The full-information VOI row is explicitly
an offline diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from p2_logic import terminal_answer

METHODS = (
    "full_new_graph",
    "old_graph_reuse",
    "single_model",
    "retrieval_augmented_generation",
    "exact_cache",
    "graph_mask_recompute",
    "topology_voi_online",
    "full_information_voi_upper_bound",
)


def encode(tokenizer, text: str) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        encoded = tokenizer.encode(text, add_special_tokens=True)
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    return list(encoded)


class Ledger:
    def __init__(self, model, tokenizer, device, pad_id, source_ids, label_ids):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pad_id = pad_id
        self.source_ids = source_ids
        self.label_ids = label_ids
        self.forward_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.wall_seconds = 0.0
        self.cache_hits = 0
        self.retrieval_calls = 0

    def call(self, prompt: str, kind: str) -> np.ndarray | int:
        ids = encode(self.tokenizer, prompt)
        inputs = torch.as_tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(inputs)
        positions = mask.long().cumsum(1) - 1
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(
                input_ids=inputs,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
            ).logits[0, -1].float().cpu()
        self.wall_seconds += time.perf_counter() - start
        self.forward_calls += 1
        self.input_tokens += len(ids)
        self.output_tokens += 1
        if kind == "source":
            return logits[self.source_ids].softmax(0).numpy()
        if kind == "label":
            return int(torch.stack([logits[v[0]] for v in self.label_ids]).argmax().item())
        raise ValueError(kind)

    def add_cache_hit(self) -> None:
        self.cache_hits += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "forward_calls": self.forward_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_seconds": self.wall_seconds,
            "cache_hits": self.cache_hits,
            "retrieval_calls": self.retrieval_calls,
        }


def prompts(row: dict[str, Any], source_summary: str, relay_summaries: list[str], fact_set: list[str], interfaces: list[str] | None = None) -> dict[str, Any]:
    options = row["options"]
    menu = "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(options))
    if len(options) == 8:
        grouped = "\n".join(
            f"{chr(65 + i)}. {options[2 * i]} OR {options[2 * i + 1]}"
            for i in range(4)
        )
    else:
        grouped = "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(options))
    source_prompt = (
        row["stem"] + "\nHypotheses:\n" + grouped
        + "\nSource evidence: " + row["first_fact"]
        + "\nReturn a probability over A, B, C, D."
    )
    relay_prompts = []
    if interfaces is None:
        interfaces = row.get("new_relay_interface", [""] * 4)
    for index in range(4):
        interface = interfaces[index]
        relay_prompts.append(
            row["stem"] + "\nHypotheses:\n" + grouped
            + "\nUpstream S0 distribution: " + source_summary
            + "\nRelay evidence: " + fact_set[index]
            + ("\n" + interface if interface else "")
            + "\nRelay branch " + chr(65 + index)
            + "\nReturn a probability over A, B, C, D."
        )
    terminal_prompts = {}
    for i, j in itertools.product(range(4), repeat=2):
        terminal_prompts[(i, j)] = (
            row["stem"] + "\nOptions:\n" + menu
            + "\nS0 distribution: " + source_summary
            + "\nRelay " + chr(65 + i) + ": " + relay_summaries[i]
            + "\nRelay " + chr(65 + j) + ": " + relay_summaries[j]
            + "\nChoose the best answer letter."
        )
    direct_prompt = (
        row["stem"] + "\nOptions:\n" + menu
        + "\nEvidence 1: " + row["first_fact"]
        + "\nEvidence 2: " + row["second_fact"]
        + "\nReturn only the best answer letter."
    )
    rag_prompt = (
        row["stem"] + "\nOptions:\n" + menu
        + "\nRetrieved evidence 1: " + row["first_fact"]
        + "\nRetrieved evidence 2: " + row["second_fact"]
        + "\nReturn only the best answer letter."
    )
    return {
        "source": source_prompt,
        "relay": relay_prompts,
        "terminal": terminal_prompts,
        "direct": direct_prompt,
        "rag": rag_prompt,
    }


def summary(values: np.ndarray) -> str:
    return ", ".join(f"{chr(65 + i)}={float(value):.4f}" for i, value in enumerate(values))


def make_cache(row, ledger: Ledger, old: bool) -> dict[str, Any]:
    facts = [row["second_fact"]] * 4 if old else row["second_facts_new"]
    interfaces = [""] * 4 if old else row.get("new_relay_interface", [""] * 4)
    source_prompt = prompts(row, "pending", ["pending"] * 4, facts, interfaces)["source"]
    source = ledger.call(source_prompt, "source")
    relay_prompts = prompts(row, summary(source), ["pending"] * 4, facts, interfaces)["relay"]
    relay = [ledger.call(prompt, "source") for prompt in relay_prompts]
    relay_summaries = [summary(value) for value in relay]
    terminal_prompts = prompts(row, summary(source), relay_summaries, facts, interfaces)["terminal"]
    labels = {
        cell: int(ledger.call(prompt, "label"))
        for cell, prompt in terminal_prompts.items()
    }
    return {"source": source, "relay": relay, "labels": labels}


def score(answer: int, target: int, gold: int) -> dict[str, float]:
    return {
        "answer": float(answer),
        "answer_accuracy": float(answer == gold),
        "answer_action_exact": float(answer == target),
        "decision_exact": float(answer == target),
    }


def run_row(row, gold: int, model, tokenizer, device, pad_id, source_ids, label_ids) -> dict[str, Any]:
    history_ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    old_cache = make_cache(row, history_ledger, old=True)
    history_cost = history_ledger.snapshot()

    # Full new graph is the scoring target, not an input to any online route.
    target_ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    new_cache = make_cache(row, target_ledger, old=False)
    target = terminal_answer(new_cache["source"], new_cache["relay"], new_cache["labels"])
    target_cost = target_ledger.snapshot()

    changed = set(int(value) for value in row["changed_rows"])
    methods: dict[str, Any] = {}

    # Old graph reuse.
    ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    answer = terminal_answer(old_cache["source"], old_cache["relay"], old_cache["labels"])
    ledger.cache_hits = 21
    methods["old_graph_reuse"] = {"metrics": score(answer, target, gold), "cost": ledger.snapshot()}

    # Single model and fixed retrieval baseline.
    for name, key in (("single_model", "direct"), ("retrieval_augmented_generation", "rag")):
        ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
        p = prompts(row, "pending", ["pending"] * 4, [row["second_fact"]] * 4)
        if name == "retrieval_augmented_generation":
            ledger.retrieval_calls = 2
        answer = int(ledger.call(p[key], "label"))
        methods[name] = {"metrics": score(answer, target, gold), "cost": ledger.snapshot()}

    # Exact cache recomputes changed relay rows but leaves terminal cache stale.
    ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    relay = list(old_cache["relay"])
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    for index in sorted(changed):
        relay[index] = ledger.call(p["relay"][index], "source")
    ledger.cache_hits = 21 - len(changed)
    answer = terminal_answer(old_cache["source"], relay, old_cache["labels"])
    methods["exact_cache"] = {"metrics": score(answer, target, gold), "cost": ledger.snapshot()}

    # Graph mask recomputation reruns all terminal descendants of changed relays.
    ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    relay = list(old_cache["relay"])
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    for index in sorted(changed):
        relay[index] = ledger.call(p["relay"][index], "source")
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    labels = {cell: int(ledger.call(prompt, "label")) for cell, prompt in p["terminal"].items()}
    ledger.cache_hits = 3
    answer = terminal_answer(old_cache["source"], relay, labels)
    methods["graph_mask_recompute"] = {"metrics": score(answer, target, gold), "cost": ledger.snapshot()}

    # Certificate VOI revalidation. Changed relay rows are queried first, then
    # dirty terminal cells are ranked by their graph-conditioned mass. The
    # strict weighted certificate permits stopping before all dirty cells when
    # one answer remains separated from every alternative by the unqueried mass.
    ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    relay = list(old_cache["relay"])
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    for index in sorted(changed):
        relay[index] = ledger.call(p["relay"][index], "source")
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    labels = dict(old_cache["labels"])
    weights = {cell: float(old_cache["source"][cell[0]] * relay[cell[0]][cell[1]]) for cell in labels}
    dirty = [cell for cell in sorted(labels) if cell[0] in changed]
    order = sorted(dirty, key=lambda cell: (weights[cell], cell), reverse=True)
    queried = 0
    for cell in order:
        known_scores = np.zeros(8, dtype=float)
        unknown = 0.0
        for key, value in labels.items():
            if key in order[queried:]:
                unknown += weights[key]
            else:
                known_scores[int(value)] += weights[key]
        choice = int(np.argmax(known_scores))
        other = max(float(known_scores[i]) for i in range(8) if i != choice)
        if known_scores[choice] > other + unknown:
            break
        labels[cell] = int(ledger.call(p["terminal"][cell], "label"))
        queried += 1
    ledger.cache_hits = 21 - len(changed) - queried
    answer = terminal_answer(old_cache["source"], relay, labels)
    methods["certificate_voi"] = {
        "metrics": score(answer, target, gold),
        "cost": ledger.snapshot(),
        "queried_terminal_cells": queried,
        "certificate": bool(queried < len(dirty)),
    }

    # Same strict certificate and dirty cone, but a deterministic random cell
    # order. This isolates the gain from certificate stopping versus the
    # graph-conditioned weight ranking.
    ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    relay = list(old_cache["relay"])
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    for index in sorted(changed):
        relay[index] = ledger.call(p["relay"][index], "source")
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    labels = dict(old_cache["labels"])
    weights = {cell: float(old_cache["source"][cell[0]] * relay[cell[0]][cell[1]]) for cell in labels}
    dirty = [cell for cell in sorted(labels) if cell[0] in changed]
    import hashlib as _hashlib
    order = sorted(dirty, key=lambda cell: _hashlib.sha256(f"{row['key']}:{cell[0]}:{cell[1]}".encode()).hexdigest())
    queried = 0
    for cell in order:
        known_scores = np.zeros(8, dtype=float)
        unknown = 0.0
        for key, value in labels.items():
            if key in order[queried:]:
                unknown += weights[key]
            else:
                known_scores[int(value)] += weights[key]
        choice = int(np.argmax(known_scores))
        other = max(float(known_scores[i]) for i in range(8) if i != choice)
        if known_scores[choice] > other + unknown:
            break
        labels[cell] = int(ledger.call(p["terminal"][cell], "label"))
        queried += 1
    ledger.cache_hits = 21 - len(changed) - queried
    answer = terminal_answer(old_cache["source"], relay, labels)
    methods["certificate_random"] = {
        "metrics": score(answer, target, gold),
        "cost": ledger.snapshot(),
        "queried_terminal_cells": queried,
        "certificate": bool(queried < len(dirty)),
    }

    # Max-uncertainty ablation. The dirty cells are ordered by the observed
    # relay distribution change, with cell mass as a deterministic tiebreaker.
    ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    relay = list(old_cache["relay"])
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    for index in sorted(changed):
        relay[index] = ledger.call(p["relay"][index], "source")
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    labels = dict(old_cache["labels"])
    weights = {cell: float(old_cache["source"][cell[0]] * relay[cell[0]][cell[1]]) for cell in labels}
    dirty = [cell for cell in sorted(labels) if cell[0] in changed]
    uncertainty = {cell: float(np.abs(old_cache["relay"][cell[0]] - relay[cell[0]]).sum()) for cell in dirty}
    order = sorted(dirty, key=lambda cell: (uncertainty[cell], weights[cell], cell), reverse=True)
    queried = 0
    for cell in order:
        known_scores = np.zeros(8, dtype=float); unknown = 0.0
        for key, value in labels.items():
            if key in order[queried:]: unknown += weights[key]
            else: known_scores[int(value)] += weights[key]
        choice = int(np.argmax(known_scores)); other = max(float(known_scores[i]) for i in range(8) if i != choice)
        if known_scores[choice] > other + unknown: break
        labels[cell] = int(ledger.call(p["terminal"][cell], "label")); queried += 1
    ledger.cache_hits = 21 - len(changed) - queried
    answer = terminal_answer(old_cache["source"], relay, labels)
    methods["certificate_uncertainty"] = {
        "metrics": score(answer, target, gold), "cost": ledger.snapshot(), "queried_terminal_cells": queried,
        "certificate": bool(queried < len(dirty)),
    }

    # Periodic fixed-order ablation with the same stopping certificate.
    ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    relay = list(old_cache["relay"])
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    for index in sorted(changed): relay[index] = ledger.call(p["relay"][index], "source")
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    labels = dict(old_cache["labels"]); weights = {cell: float(old_cache["source"][cell[0]] * relay[cell[0]][cell[1]]) for cell in labels}
    dirty = [cell for cell in sorted(labels) if cell[0] in changed]; order = sorted(dirty); queried = 0
    for cell in order:
        known_scores = np.zeros(8, dtype=float); unknown = 0.0
        for key, value in labels.items():
            if key in order[queried:]: unknown += weights[key]
            else: known_scores[int(value)] += weights[key]
        choice = int(np.argmax(known_scores)); other = max(float(known_scores[i]) for i in range(8) if i != choice)
        if known_scores[choice] > other + unknown: break
        labels[cell] = int(ledger.call(p["terminal"][cell], "label")); queried += 1
    ledger.cache_hits = 21 - len(changed) - queried
    answer = terminal_answer(old_cache["source"], relay, labels)
    methods["certificate_periodic"] = {
        "metrics": score(answer, target, gold), "cost": ledger.snapshot(), "queried_terminal_cells": queried,
        "certificate": bool(queried < len(dirty)),
    }

    # Online topology VOI uses only graph impact and token costs, no gold/new cache.
    ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    relay = list(old_cache["relay"])
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    for index in sorted(changed):
        relay[index] = ledger.call(p["relay"][index], "source")
    impacts = []
    for cell in sorted(old_cache["labels"]):
        i, j = cell
        impact = float(np.abs(old_cache["relay"][i] - relay[i]).sum() + np.abs(old_cache["relay"][j] - relay[j]).sum())
        impacts.append((impact, cell))
    threshold = 0.08
    labels = dict(old_cache["labels"])
    queried = 0
    p = prompts(row, summary(old_cache["source"]), [summary(value) for value in relay], row["second_facts_new"])
    for impact, cell in sorted(impacts, reverse=True):
        if impact < threshold:
            break
        labels[cell] = int(ledger.call(p["terminal"][cell], "label"))
        queried += 1
    ledger.cache_hits = 21 - len(changed) - queried
    answer = terminal_answer(old_cache["source"], relay, labels)
    methods["topology_voi_online"] = {
        "metrics": score(answer, target, gold),
        "cost": ledger.snapshot(),
        "threshold": threshold,
        "queried_terminal_cells": queried,
    }

    # Offline full-information upper bound. This block is isolated from online actions.
    # It reports the smallest subset of changed relay/terminal queries that can recover
    # the target answer after the target is known. It is never used to choose online calls.
    old_answer = terminal_answer(old_cache["source"], old_cache["relay"], old_cache["labels"])
    ledger = Ledger(model, tokenizer, device, pad_id, source_ids, label_ids)
    upper_calls = len(changed) + (0 if old_answer == target else 1)
    ledger.cache_hits = max(0, 21 - upper_calls)
    methods["full_information_voi_upper_bound"] = {
        "metrics": score(target, target, gold),
        "cost": {**ledger.snapshot(), "forward_calls": upper_calls, "input_tokens": 0, "output_tokens": 0, "wall_seconds": 0.0},
        "offline_only": True,
    }

    methods["full_new_graph"] = {
        "metrics": score(target, target, gold),
        "cost": target_cost,
    }
    return {
        "key": row["key"],
        "gold": int(gold),
        "target": int(target),
        "history_cost": history_cost,
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    for row in rows:
        if "changed_rows" not in row:
            raw = row.get("interface_drift_rows", "[]")
            try:
                row["changed_rows"] = json.loads(raw) if isinstance(raw, str) else list(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                row["changed_rows"] = []
        if "new_relay_interface" not in row:
            row["new_relay_interface"] = [""] * 4
        if "second_facts_new" not in row:
            row["second_facts_new"] = [row.get("second_fact_new") or row["second_fact"]] * 4
    rows = rows[args.offset : args.offset + args.limit] if args.limit else rows[args.offset :]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.unk_token_id or 0
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True).to(args.device).eval()
    source_ids = [tokenizer.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(4)]
    label_ids = [tokenizer.encode(chr(65 + i), add_special_tokens=False) for i in range(8)]
    assert all(len(value) == 1 for value in label_ids)
    results = []
    for index, row in enumerate(rows):
        if row["key"] not in gold:
            raise KeyError(row["key"])
        results.append(run_row(row, int(gold[row["key"]]), model, tokenizer, args.device, pad_id, source_ids, label_ids))
        print(json.dumps({"done": index + 1, "total": len(rows), "key": row["key"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "mab2_p2_service_result_v1", "model": str(args.model), "rows": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()



