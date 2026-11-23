"""Aggregate pairwise answer-action certificate replays and fair baselines."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from pairwise_gap_certificate import parse_worker
from answer_safe_arcc import values


ROOT = Path(__file__).resolve().parents[2]


def fair_mask(rows: list[dict]) -> dict:
    records = []
    for row in rows:
        selected = sorted(set(int(i) for i in row["changed_rows"]))
        relay = row["old_relay"].copy()
        for index in selected:
            relay[index] = row["new_relay"][index]
        pred = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        records.append({
            "answer_action_exact": float(pred.argmax() == truth.argmax()),
            "tokens": float(sum(row["tokens"][7 + i] for i in selected)),
            "queries": float(len(selected)),
        })
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


def parse_name(name: str) -> tuple[str, str, str]:
    match = re.match(r"dedup_(qwen05|qwen14|llama3)_(commonsense|qasc|sciq)(?:_rows(1|2|4))?_pairwise\.json", name)
    if not match:
        raise ValueError(name)
    model, task, density = match.groups()
    return {"qwen05": "Qwen-0.5B", "qwen14": "Qwen-14B", "llama3": "Llama-3"}[model], task, density or "1"


def main() -> None:
    records = []
    for path in sorted(ROOT.glob("dedup_*_pairwise.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        worker = ROOT / result["worker"]
        rows = parse_worker(worker)
        test = [row for row in rows if int(row["key"][:8], 16) % 5 == 4]
        pairwise = result["test"]
        fair = fair_mask(test)
        model, task, density = parse_name(path.name)
        records.append({
            "cell": path.stem,
            "model": model,
            "task": task,
            "density": int(density),
            "test_n": len(test),
            "pairwise_action_exact": pairwise["answer_action_exact"],
            "pairwise_tokens": pairwise["tokens"],
            "pairwise_queries": pairwise["queries"],
            "fair_action_exact": fair["answer_action_exact"],
            "fair_tokens": fair["tokens"],
            "fair_queries": fair["queries"],
            "token_reduction": 1.0 - pairwise["tokens"] / max(fair["tokens"], 1e-12),
        })
    aggregate = {}
    for key, subset in [("all", records)] + [
        (f"model={model}", [r for r in records if r["model"] == model])
        for model in sorted({r["model"] for r in records})
    ] + [
        (f"density={density}", [r for r in records if r["density"] == density])
        for density in sorted({r["density"] for r in records})
    ]:
        aggregate[key] = {
            "cells": len(subset),
            "action_exact": float(np.mean([r["pairwise_action_exact"] for r in subset])),
            "pairwise_tokens": float(np.mean([r["pairwise_tokens"] for r in subset])),
            "fair_tokens": float(np.mean([r["fair_tokens"] for r in subset])),
            "token_reduction": float(np.mean([r["token_reduction"] for r in subset])),
            "weighted_token_reduction": float(1.0 - np.mean([r["pairwise_tokens"] for r in subset]) / max(np.mean([r["fair_tokens"] for r in subset]), 1e-12)),
            "fair_action_exact": float(np.mean([r["fair_action_exact"] for r in subset])),
        }
    payload = {
        "schema": "pairwise_gap_certificate_aggregate_v1",
        "selection": "per-cell q is selected on residues 0-3 and the test residue is 4",
        "records": records,
        "aggregate": aggregate,
    }
    out = Path(__file__).with_name("PAIRWISE_GAP_CERTIFICATE_AGGREGATE.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Pairwise answer-action certificate aggregate",
        "",
        "The pairwise route and fair descendant-mask baseline use the same cached transcript, graph mask and row prices. The pairwise route selects its radius and quantile on residues 0--3 and is evaluated once on residue 4.",
        "",
        "| group | cells | pairwise action | pairwise tokens | fair tokens | mean reduction | weighted reduction | fair action |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in aggregate.items():
        lines.append(f"| {key} | {item['cells']} | {item['action_exact']:.3f} | {item['pairwise_tokens']:.2f} | {item['fair_tokens']:.2f} | {item['token_reduction']:.1%} | {item['weighted_token_reduction']:.1%} | {item['fair_action_exact']:.3f} |")
    Path(__file__).with_name("PAIRWISE_GAP_CERTIFICATE_AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(records), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()

