"""Fair answer-action baseline for risk-gated route evaluation.

The baseline refreshes only graph-descendant relay rows and stops after scoring
the final answer action. It does not purchase conditional cells required for a
full edge-decision certificate, so it is comparable with the answer-region
route's answer-only target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from answer_region_only import parse_rows
from answer_safe_arcc import TAU, values


def evaluate(rows):
    records = []
    for row in rows:
        selected = sorted(set(int(i) for i in row.get("changed_rows", [])))
        relay = np.asarray(row["old_relay"], dtype=float).copy()
        for i in selected:
            relay[i] = row["new_relay"][i]
        pred = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        records.append({"answer_action_exact": float(pred.argmax() == truth.argmax()),
                        "answer_accuracy": float(pred.argmax() == row["gold"]),
                        "decision_reference_exact": float(np.array_equal(pred > TAU, truth > TAU)),
                        "tokens": float(sum(row["tokens"][7 + i] for i in selected)),
                        "queries": float(len(selected))})
    return {key: float(np.mean([row[key] for row in records])) for key in records[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = parse_rows(args.worker, args.gold)
    test = [row for row in rows if int(row["key"][:8], 16) % 5 == 4]
    result = {"schema": "fair_answer_action_baseline_v1", "worker": args.worker.name,
              "test_n": len(test), "aggregate": evaluate(test)}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

