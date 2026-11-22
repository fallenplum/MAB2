"""Train a question-disjoint service-aware row selector on sealed development labels.

The selector scores each changed relay row by the answer obtained after applying
that row.  It is evaluated on question-disjoint rows.  This is a candidate
algorithm experiment, not a claim of independent benchmark performance.
"""
import json
import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"E:\图结构演化\artifacts\private\mechanism_20261122")
TASKS = ("qasc", "sciq", "commonsense")
DENSITIES = ("", "_rows2", "_rows4")


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
                        "new_source": vector(1),
                        "base": vector(2),
                        "old_relay": np.stack([vector(i) for i in range(3, 7)]),
                        "new_relay": np.stack([vector(i) for i in range(7, 11)]),
                        "labels": np.asarray(
                            [int(full[str(i)]) for i in range(11, 27)],
                            dtype=np.int64,
                        ).reshape(4, 4),
                        "tokens": item["tokens"],
                        "changed": tuple(int(value) for value in item.get("changed_rows", [])),
                        "gold": int(gold[item["key"]]),
                    }
                )
    return rows


def values(source, base, relay, labels):
    weights = np.asarray(source)[:, None] * (np.asarray(relay) - np.asarray(base)[None, :])
    return np.asarray(
        [weights[np.asarray(labels) == label].sum() for label in range(int(np.max(labels)) + 1)]
    )


def entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    return float(
        -(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))).sum()
    )


def padded(values_array):
    values_array = np.asarray(values_array, dtype=np.float32)
    if len(values_array) < 4:
        return np.r_[values_array, np.zeros(4 - len(values_array), dtype=np.float32)]
    return values_array[:4]


def candidate_features(row, row_index):
    old_values = padded(values(row["new_source"], row["base"], row["old_relay"], row["labels"]))

    def summary(vector):
        order = np.argsort(-vector)
        return np.r_[
            vector,
            vector[order[0]] - vector[order[1]],
            np.eye(4, dtype=np.float32)[order[0]],
            np.eye(4, dtype=np.float32)[order[1]],
        ]

    density_index = {"_rows1": 0, "_rows2": 1, "_rows4": 2}[row["density"]]
    return np.r_[
        summary(old_values),
        row["new_source"] - row["base"],
        row["new_source"] - row["base"],
        row["old_relay"][row_index],
        [
            entropy(row["old_relay"][row_index]),
            row_index,
            len(row["changed"]),
        ],
        np.eye(4, dtype=np.float32)[row_index],
        np.eye(3, dtype=np.float32)[density_index],
    ].astype(np.float32)


def candidate_label(row, row_index):
    relay = row["old_relay"].copy()
    relay[row_index] = row["new_relay"][row_index]
    prediction = int(values(row["new_source"], row["base"], relay, row["labels"]).argmax())
    return int(prediction == row["gold"])


def evaluate(rows, model):
    grouped = {}
    for row in rows:
        candidates = []
        for row_index in row["changed"]:
            candidates.append(
                (
                    model.predict_proba(candidate_features(row, row_index)[None, :])[0, 1],
                    row_index,
                    candidate_label(row, row_index),
                )
            )
        grouped[(row["density"], row["key"])] = (row, candidates)

    selected = []
    uncertainty = []
    oracle = []
    token_cost = []
    uncertainty_cost = []
    for row, candidates in grouped.values():
        chosen = max(candidates, key=lambda item: item[0])
        uncertain = max(
            candidates,
            key=lambda item: entropy(row["old_relay"][item[1]]),
        )
        selected.append(chosen[2])
        uncertainty.append(uncertain[2])
        oracle.append(max(item[2] for item in candidates))
        token_cost.append(float(row["tokens"][7 + chosen[1]]))
        uncertainty_cost.append(float(row["tokens"][7 + uncertain[1]]))
    return {
        "n": len(selected),
        "answer_accuracy": float(np.mean(selected)),
        "uncertainty_answer_accuracy": float(np.mean(uncertainty)),
        "oracle_answer_accuracy": float(np.mean(oracle)),
        "tokens": float(np.mean(token_cost)),
        "uncertainty_tokens": float(np.mean(uncertainty_cost)),
    }


def run(task, seed, model_name, all_rows):
    random.seed(seed)
    np.random.seed(seed)
    task_rows = [row for row in all_rows if row["task"] == task]
    train = [row for row in task_rows if int(row["key"][:8], 16) % 5 < 4]
    test = [row for row in task_rows if int(row["key"][:8], 16) % 5 == 4]
    train_x = []
    train_y = []
    for row in train:
        for row_index in row["changed"]:
            train_x.append(candidate_features(row, row_index))
            train_y.append(candidate_label(row, row_index))
    if model_name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    model.fit(np.asarray(train_x), np.asarray(train_y))
    result = evaluate(test, model)
    result.update(
        {
            "task": task,
            "seed": seed,
            "model": model_name,
            "train_questions": len(train),
            "test_questions": len(test),
            "positive_rate": float(np.mean(train_y)),
        }
    )
    return result


if __name__ == "__main__":
    all_rows = load_rows()
    results = [
        run(task, seed, model_name, all_rows)
        for task in TASKS
        for seed in (7, 19, 43)
        for model_name in ("logistic", "random_forest")
    ]
    output = {
        "schema": "m60_service_answer_selector_v1",
        "results": results,
        "interpretation": "Question-disjoint development evaluation on private sealed labels. The selector is a candidate algorithm and is not independent confirmation.",
    }
    path = Path(
        "MAB2_recovered/experiments/mechanism_20261122/M60_SERVICE_ANSWER_SELECTOR.json"
    )
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

