"""Development-tuned quantile-radius answer-safe revalidation.

The policy only uses the edited source distribution, cached old relay rows,
and a radius calibrated on a disjoint calibration split.  New relay values are
used only to score the held-out answer/action, never to choose queries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from answer_safe_arcc import TAU, choose, validate_conditional_full, values


QUANTILES = (0.5, 0.75, 0.9, 0.95, 1.0)


def parse(worker_path):
    return [json.loads(line) for line in worker_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_rows(raw_rows):
    rows = []
    for item in raw_rows:
        parsed = validate_conditional_full(item["full"])
        old_source, new_source, base = parsed["source"]
        rows.append({**parsed, "old_source": old_source, "new_source": new_source, "base": base,
                     "key": item["key"], "tokens": item["tokens"],
                     "changed_rows": item.get("changed_rows", [])})
    return rows


def radii(rows, q):
    samples = {}
    for row in rows:
        for i in row["changed_rows"]:
            tv = 0.5 * np.abs(row["new_relay"][int(i)] - row["old_relay"][int(i)]).sum()
            samples.setdefault(int(i), []).append(float(tv))
    return {i: float(np.quantile(v, q)) for i, v in samples.items()}


def evaluate_rows(rows, radius):
    records = []
    for row in rows:
        selected = choose(row, radius)
        relay = np.asarray(row["old_relay"], dtype=float).copy()
        for i in selected:
            relay[i] = row["new_relay"][i]
        predicted = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        records.append({
            "answer_action_exact": float(int(predicted.argmax()) == int(truth.argmax())),
            "decision_exact": float(np.array_equal(predicted > TAU, truth > TAU)),
            "answer_accuracy": float(int(predicted.argmax()) == int(row["gold"])),
            "tokens": float(sum(row["tokens"][7 + i] for i in selected)),
            "queries": float(len(selected)),
        })
    return records


def summary(records):
    return {k: float(np.mean([r[k] for r in records])) for k in records[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    rows = build_rows(parse(args.worker))
    for row in rows:
        raw = int(gold[row["key"]])
        row["gold"] = 1 if raw == -15 else 3 if raw == -13 else raw
    calibration = [r for r in rows if int(r["key"][:8], 16) % 5 < 3]
    selection = [r for r in rows if int(r["key"][:8], 16) % 5 == 3]
    test = [r for r in rows if int(r["key"][:8], 16) % 5 == 4]
    candidates = []
    for q in QUANTILES:
        radius = radii(calibration, q)
        valid = evaluate_rows(selection, radius)
        candidates.append({"q": q, "radius": radius, "selection": summary(valid)})
    best = sorted(candidates, key=lambda x: (-x["selection"]["answer_accuracy"],
                                               -x["selection"]["answer_action_exact"],
                                               x["selection"]["tokens"], x["q"]))[0]
    test_records = evaluate_rows(test, radii(calibration, best["q"]))
    output = {
        "schema": "arcc_adaptive_quantile_answer_policy_v1",
        "worker": args.worker.name,
        "calibration_n": len(calibration),
        "selection_n": len(selection),
        "test_n": len(test),
        "candidates": candidates,
        "selected": {"q": best["q"], "radius": radii(calibration, best["q"])},
        "aggregate": summary(test_records),
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

