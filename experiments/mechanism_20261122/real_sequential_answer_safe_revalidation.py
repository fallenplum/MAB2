"""Sequential conditional revalidation with both graph and answer certificates."""
import json
from pathlib import Path

import numpy as np

from real_conditional_interval_policy import load_rows, values


TAU = 0.01


def choose(row, radius):
    unknown = set(row["changed"])
    selected = []
    costs = [row["tokens"][7 + index] for index in range(4)]
    label_count = int(np.max(row["labels"])) + 1
    while unknown:
        low = np.zeros(label_count)
        high = np.zeros(label_count)
        center = np.zeros(label_count)
        for index in range(4):
            relay = row["old_relay"][index] if index in unknown else row["new_relay"][index]
            contribution = row["source"][index] * (relay - row["base"])
            center_contribution = contribution.copy()
            row_radius = row["source"][index] * radius.get((row["density"], index), 1.0) if index in unknown else 0.0
            for label in range(label_count):
                mask = row["labels"][index] == label
                center[label] += center_contribution[mask].sum()
                low[label] += center_contribution[mask].sum() - row_radius
                high[label] += center_contribution[mask].sum() + row_radius
        edge_unresolved = ~((low > TAU) | (high < TAU))
        answer_order = int(np.argmax(center))
        answer_safe = label_count == 1 or low[answer_order] > np.max(np.delete(high, answer_order))
        if not np.any(edge_unresolved) and answer_safe:
            break
        unresolved_mass = float(edge_unresolved.sum()) + float(not answer_safe)
        candidates = []
        for index in unknown:
            score = 2.0 * row["source"][index] * radius.get((row["density"], index), 1.0) * unresolved_mass / max(costs[index], 1.0)
            candidates.append((score, index))
        _, chosen = max(candidates)
        unknown.remove(chosen)
        selected.append(chosen)
    return selected


def evaluate(rows, radius):
    edge_exact = []
    answer_exact = []
    answer = []
    tokens = []
    queries = []
    for row in rows:
        selected = choose(row, radius)
        relay = row["old_relay"].copy()
        for index in selected:
            relay[index] = row["new_relay"][index]
        predicted = values(row, relay)
        truth = values(row, row["new_relay"])
        edge_exact.append(float(np.array_equal(predicted > TAU, truth > TAU)))
        answer_exact.append(float(int(predicted.argmax()) == int(truth.argmax())))
        answer.append(float(int(predicted.argmax()) == row["gold"]))
        tokens.append(float(sum(row["tokens"][7 + index] for index in selected)))
        queries.append(len(selected))
    return {"n": len(rows), "decision_exact": float(np.mean(edge_exact)), "answer_action_exact": float(np.mean(answer_exact)), "answer_accuracy": float(np.mean(answer)), "tokens": float(np.mean(tokens)), "queries": float(np.mean(queries))}


def run(task):
    rows = load_rows()
    train = [row for row in rows if row["task"] == task and int(row["key"][:8], 16) % 5 < 4]
    test = [row for row in rows if row["task"] == task and int(row["key"][:8], 16) % 5 == 4]
    radius = {}
    for row in train:
        for index in row["changed"]:
            tv = 0.5 * np.abs(row["new_relay"][index] - row["old_relay"][index]).sum()
            radius[(row["density"], index)] = max(radius.get((row["density"], index), 0.0), float(tv))
    return {"task": task, "train": len(train), "test": len(test), "radius": {f"{key[0]}:{key[1]}": value for key, value in radius.items()}, "answer_safe_sequential": evaluate(test, radius)}


if __name__ == "__main__":
    output = {"schema": "m60_real_sequential_answer_safe_revalidation_v1", "results": [run(task) for task in ("qasc", "sciq", "commonsense")], "interpretation": "Question-disjoint development transfer with a graph-decision and final-answer-region certificate. Empirical radii are not independent formal bounds."}
    path = Path("MAB2_recovered/experiments/mechanism_20261122/M60_REAL_SEQUENTIAL_ANSWER_SAFE_REVALIDATION.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

