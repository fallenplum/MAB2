"""Apply a train-calibrated conditional-drift interval to real M60 outputs."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(r"E:\图结构演化\artifacts\private\mechanism_20261122")
TASKS = ("qasc", "sciq", "commonsense")
DENSITIES = ("", "_rows2", "_rows4")
TAU = 0.01


def load_rows():
    rows = []
    for task in TASKS:
        for suffix in DENSITIES:
            directory = ROOT / (
                f"m60_conditional_server_{task}"
                if suffix == ""
                else f"m60_conditional_{task}{suffix}"
            )
            if not directory.exists():
                continue
            gold = json.loads((directory / "SEALED.json").read_text())
            for line in (directory / "WORKER0.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                full = item["full"]
                vector = lambda index: np.asarray(full[str(index)], dtype=np.float32)
                rows.append(
                    {
                        "key": item["key"],
                        "task": task,
                        "density": suffix or "_rows1",
                        "source": vector(1),
                        "base": vector(2),
                        "old_relay": np.stack([vector(i) for i in range(3, 7)]),
                        "new_relay": np.stack([vector(i) for i in range(7, 11)]),
                        "labels": np.asarray(
                            [int(full[str(i)]) for i in range(11, 27)], dtype=np.int64
                        ).reshape(4, 4),
                        "tokens": item["tokens"],
                        "changed": tuple(int(v) for v in item.get("changed_rows", [])),
                        "gold": int(gold[item["key"]]),
                    }
                )
    return rows


def values(row, relay):
    weights = row["source"][:, None] * (relay - row["base"][None, :])
    return np.asarray(
        [weights[row["labels"] == label].sum() for label in range(int(np.max(row["labels"])) + 1)]
    )


def interval(row, radius):
    label_count = int(np.max(row["labels"])) + 1
    low = np.zeros(label_count, dtype=np.float64)
    high = np.zeros(label_count, dtype=np.float64)
    for row_index in range(4):
        row_radius = radius.get((row["density"], row_index), 1.0)
        contribution = row["source"][row_index] * (
            row["old_relay"][row_index] - row["base"]
        )
        for label in range(label_count):
            mask = row["labels"][row_index] == label
            low[label] += contribution[mask].sum() - row["source"][row_index] * row_radius * mask.sum()
            high[label] += contribution[mask].sum() + row["source"][row_index] * row_radius * mask.sum()
    return low, high


def evaluate(rows, radius):
    exact = []
    answer = []
    tokens = []
    queries = []
    for row in rows:
        low, high = interval(row, radius)
        safe = bool(np.all((low > TAU) | (high < TAU)))
        selected = [] if safe else list(row["changed"])
        relay = row["old_relay"].copy()
        for index in selected:
            relay[index] = row["new_relay"][index]
        predicted = values(row, relay)
        truth = values(row, row["new_relay"])
        exact.append(float(np.array_equal(predicted > TAU, truth > TAU)))
        answer.append(float(int(predicted.argmax()) == row["gold"]))
        tokens.append(float(sum(row["tokens"][7 + index] for index in selected)))
        queries.append(len(selected))
    return {
        "n": len(rows),
        "decision_exact": float(np.mean(exact)),
        "answer_accuracy": float(np.mean(answer)),
        "tokens": float(np.mean(tokens)),
        "queries": float(np.mean(queries)),
    }


def baseline(rows, name):
    if name == "warm":
        exact = []
        answer = []
        for row in rows:
            predicted = values(row, row["old_relay"])
            truth = values(row, row["new_relay"])
            exact.append(float(np.array_equal(predicted > TAU, truth > TAU)))
            answer.append(float(int(predicted.argmax()) == row["gold"]))
        return {"n": len(rows), "decision_exact": float(np.mean(exact)), "answer_accuracy": float(np.mean(answer)), "tokens": 0.0, "queries": 0.0}
    if name == "uncertainty":
        exact = []
        answer = []
        tokens = []
        for row in rows:
            index = max(row["changed"], key=lambda i: -(row["old_relay"][i] * np.log(np.clip(row["old_relay"][i], 1e-8, 1))).sum())
            relay = row["old_relay"].copy()
            relay[index] = row["new_relay"][index]
            predicted = values(row, relay)
            truth = values(row, row["new_relay"])
            exact.append(float(np.array_equal(predicted > TAU, truth > TAU)))
            answer.append(float(int(predicted.argmax()) == row["gold"]))
            tokens.append(float(row["tokens"][7 + index]))
        return {"n": len(rows), "decision_exact": float(np.mean(exact)), "answer_accuracy": float(np.mean(answer)), "tokens": float(np.mean(tokens)), "queries": 1.0}
    raise ValueError(name)


def run(task):
    rows = load_rows()
    train = [row for row in rows if row["task"] == task and int(row["key"][:8], 16) % 5 < 4]
    test = [row for row in rows if row["task"] == task and int(row["key"][:8], 16) % 5 == 4]
    radius = {}
    for row in train:
        for index in row["changed"]:
            tv = 0.5 * np.abs(row["new_relay"][index] - row["old_relay"][index]).sum()
            radius[(row["density"], index)] = max(radius.get((row["density"], index), 0.0), float(tv))
    return {"task": task, "train": len(train), "test": len(test), "radius": {f"{key[0]}:{key[1]}": value for key, value in radius.items()}, "warm": baseline(test, "warm"), "uncertainty": baseline(test, "uncertainty"), "conditional_interval": evaluate(test, radius)}


if __name__ == "__main__":
    output = {"schema": "m60_real_conditional_interval_policy_v1", "results": [run(task) for task in TASKS], "interpretation": "Question-disjoint development transfer of a conditional-drift-safe interval calibrated from training rows. This is a cost-soundness diagnostic, not independent confirmation."}
    path = Path("MAB2_recovered/experiments/mechanism_20261122/M60_REAL_CONDITIONAL_INTERVAL_POLICY.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

