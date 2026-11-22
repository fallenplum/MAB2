"""Question-disjoint answer-region revalidation for the independent ARCC cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from conditional_schema import validate_conditional_full

TAU = 0.01


def values(source, base, relay, labels):
    weights = np.asarray(source)[:, None] * (np.asarray(relay) - np.asarray(base)[None, :])
    return np.asarray(
        [weights[np.asarray(labels) == label].sum() for label in range(int(np.max(labels)) + 1)]
    )


def choose(row, radii):
    source = np.asarray(row["new_source"], dtype=float)
    base = np.asarray(row["base"], dtype=float)
    old_relay = np.asarray(row["old_relay"], dtype=float)
    new_relay = np.asarray(row["new_relay"], dtype=float)
    labels = np.asarray(row["labels"], dtype=int)
    changed = set(int(i) for i in row.get("changed_rows", []))
    unknown = set(changed)
    selected = []
    while unknown:
        center = np.zeros(int(np.max(labels)) + 1, dtype=float)
        low = np.zeros_like(center)
        high = np.zeros_like(center)
        for i in range(4):
            relay = old_relay[i] if i in unknown else new_relay[i]
            contribution = source[i] * (relay - base)
            radius = source[i] * radii.get(i, 1.0) if i in unknown else 0.0
            for label in range(len(center)):
                mass = float(contribution[labels[i] == label].sum())
                center[label] += mass
                low[label] += mass - radius
                high[label] += mass + radius
        edge_safe = np.all((low > TAU) | (high < TAU))
        answer_order = int(np.argmax(center))
        competitors = np.delete(high, answer_order)
        answer_safe = True if competitors.size == 0 else bool(low[answer_order] > np.max(competitors))
        if edge_safe and answer_safe:
            break
        costs = {i: float(row["tokens"][7 + i]) for i in unknown}
        unresolved_mass = float(np.sum(~((low > TAU) | (high < TAU)))) + float(not answer_safe)
        chosen = max(
            unknown,
            key=lambda i: (2.0 * source[i] * radii.get(i, 1.0) * unresolved_mass / max(costs[i], 1.0), -i),
        )
        unknown.remove(chosen)
        selected.append(chosen)
    return selected


def evaluate(worker_path: Path, gold_path: Path):
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in worker_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for item in raw_rows:
        parsed = validate_conditional_full(item["full"])
        old_source, new_source, base = parsed["source"]
        rows.append({**parsed, "old_source": old_source, "new_source": new_source, "base": base,
                     "key": item["key"], "tokens": item["tokens"], "changed_rows": item.get("changed_rows", [])})
    train = [row for row in rows if int(row["key"][:8], 16) % 5 < 4]
    test = [row for row in rows if int(row["key"][:8], 16) % 5 == 4]
    radii = {}
    for row in train:
        for i in row["changed_rows"]:
            tv = 0.5 * np.abs(np.asarray(row["new_relay"])[i] - np.asarray(row["old_relay"])[i]).sum()
            radii[int(i)] = max(radii.get(int(i), 0.0), float(tv))
    records = []
    for row in test:
        selected = choose(row, radii)
        relay = np.asarray(row["old_relay"], dtype=float).copy()
        for i in selected:
            relay[i] = np.asarray(row["new_relay"])[i]
        predicted = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        raw = int(gold[row["key"]])
        corrected = 1 if raw == -15 else 3 if raw == -13 else raw
        records.append(
            {
                "decision_exact": float(np.array_equal(predicted > TAU, truth > TAU)),
                "answer_action_exact": float(int(predicted.argmax()) == int(truth.argmax())),
                "answer_accuracy": float(int(predicted.argmax()) == corrected),
                "tokens": float(sum(row["tokens"][7 + i] for i in selected)),
                "queries": float(len(selected)),
            }
        )
    return {
        "schema": "arcc_answer_safe_question_disjoint_v1",
        "worker": worker_path.name,
        "train_n": len(train),
        "test_n": len(test),
        "radii": {str(k): float(v) for k, v in radii.items()},
        "aggregate": {key: float(np.mean([r[key] for r in records])) for key in records[0]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.worker, args.gold)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

