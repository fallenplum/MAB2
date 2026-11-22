"""Question-disjoint transfer of sequential conditional revalidation."""
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
        for index in range(4):
            relay = row["old_relay"][index] if index in unknown else row["new_relay"][index]
            contribution = row["source"][index] * (relay - row["base"])
            delta = radius.get((row["density"], index), 1.0)
            row_radius = row["source"][index] * delta if index in unknown else 0.0
            for label in range(label_count):
                mask = row["labels"][index] == label
                low[label] += contribution[mask].sum() - row_radius
                high[label] += contribution[mask].sum() + row_radius
        unresolved = ~((low > TAU) | (high < TAU))
        if not np.any(unresolved):
            break
        candidates = []
        for index in unknown:
            score = (
                2.0
                * row["source"][index]
                * radius.get((row["density"], index), 1.0)
                * float(unresolved.sum())
                / max(costs[index], 1.0)
            )
            candidates.append((score, index))
        _, chosen = max(candidates)
        unknown.remove(chosen)
        selected.append(chosen)
    return selected


def evaluate(rows, radius):
    exact = []
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
        exact.append(float(np.array_equal(predicted > TAU, truth > TAU)))
        answer.append(float(int(predicted.argmax()) == row["gold"]))
        tokens.append(float(sum(row["tokens"][7 + index] for index in selected)))
        queries.append(len(selected))
    return {"n": len(rows), "decision_exact": float(np.mean(exact)), "answer_accuracy": float(np.mean(answer)), "tokens": float(np.mean(tokens)), "queries": float(np.mean(queries))}


def baseline(rows, name):
    exact = []
    answer = []
    tokens = []
    for row in rows:
        if name == "uncertainty":
            selected = [max(row["changed"], key=lambda i: -(row["old_relay"][i] * np.log(np.clip(row["old_relay"][i], 1e-8, 1))).sum())]
        elif name == "mask":
            selected = list(row["changed"])
        else:
            selected = []
        relay = row["old_relay"].copy()
        for index in selected:
            relay[index] = row["new_relay"][index]
        predicted = values(row, relay)
        truth = values(row, row["new_relay"])
        exact.append(float(np.array_equal(predicted > TAU, truth > TAU)))
        answer.append(float(int(predicted.argmax()) == row["gold"]))
        tokens.append(float(sum(row["tokens"][7 + index] for index in selected)))
    return {"n": len(rows), "decision_exact": float(np.mean(exact)), "answer_accuracy": float(np.mean(answer)), "tokens": float(np.mean(tokens)), "queries": float(np.mean([0 if name == "warm" else (1 if name == "uncertainty" else len(row["changed"])) for row in rows]))}


def run(task):
    rows = load_rows()
    train = [row for row in rows if row["task"] == task and int(row["key"][:8], 16) % 5 < 4]
    test = [row for row in rows if row["task"] == task and int(row["key"][:8], 16) % 5 == 4]
    radius = {}
    for row in train:
        for index in row["changed"]:
            tv = 0.5 * np.abs(row["new_relay"][index] - row["old_relay"][index]).sum()
            radius[(row["density"], index)] = max(radius.get((row["density"], index), 0.0), float(tv))
    return {"task": task, "train": len(train), "test": len(test), "radius": {f"{key[0]}:{key[1]}": value for key, value in radius.items()}, "warm": baseline(test, "warm"), "uncertainty": baseline(test, "uncertainty"), "mask": baseline(test, "mask"), "sequential": evaluate(test, radius)}


if __name__ == "__main__":
    output = {"schema": "m60_real_sequential_conditional_revalidation_v1", "results": [run(task) for task in ("qasc", "sciq", "commonsense")], "interpretation": "Question-disjoint transfer calibrated from private development labels. Empirical radii are not a formal independent bound."}
    path = Path("MAB2_recovered/experiments/mechanism_20261122/M60_REAL_SEQUENTIAL_CONDITIONAL_REVALIDATION.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

