"""Direct ablations for graph-conditioned answer-region revalidation.

The script keeps the answer-region stopping rule and the same cached transcript
but removes either row-specific drift radii or source-aware query ranking. It is
used to isolate the graph-conditioned components without running new models.
"""
from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np

from answer_region_only import TAU, parse_rows, values

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def _split(rows):
    key = lambda row: int(row["key"][:8], 16) % 5
    return ([row for row in rows if key(row) < 3],
            [row for row in rows if key(row) == 4])


def _row_radii(rows, quantile):
    samples = {}
    for row in rows:
        for i in row.get("changed_rows", []):
            delta = 0.5 * np.abs(row["new_relay"][int(i)] - row["old_relay"][int(i)]).sum()
            samples.setdefault(int(i), []).append(float(delta))
    return {i: float(np.quantile(values_, quantile)) for i, values_ in samples.items()}


def _pooled_radius(rows, quantile):
    samples = []
    for row in rows:
        for i in row.get("changed_rows", []):
            samples.append(float(0.5 * np.abs(row["new_relay"][int(i)] - row["old_relay"][int(i)]).sum()))
    value = float(np.quantile(samples, quantile)) if samples else 1.0
    return {i: value for i in range(4)}


def _choose(row, radii, source_aware):
    source = np.asarray(row["new_source"], dtype=float)
    base = np.asarray(row["base"], dtype=float)
    old_relay = np.asarray(row["old_relay"], dtype=float)
    new_relay = np.asarray(row["new_relay"], dtype=float)
    labels = np.asarray(row["labels"], dtype=int)
    unknown = set(int(i) for i in row.get("changed_rows", range(4)))
    selected = []
    while unknown:
        labels_n = int(np.max(labels)) + 1
        center = np.zeros(labels_n)
        low = np.zeros(labels_n)
        high = np.zeros(labels_n)
        for i in range(4):
            relay = old_relay[i] if i in unknown else new_relay[i]
            contribution = source[i] * (relay - base)
            radius = source[i] * radii.get(i, 1.0) if i in unknown else 0.0
            for label in range(labels_n):
                mass = float(contribution[labels[i] == label].sum())
                center[label] += mass
                low[label] += mass - radius
                high[label] += mass + radius
        answer = int(np.argmax(center))
        competitors = np.delete(high, answer)
        if competitors.size == 0 or bool(low[answer] > np.max(competitors)):
            break
        unresolved = float(np.sum(~((low > TAU) | (high < TAU)))) + 1.0
        def priority(i):
            weight = source[i] if source_aware else 1.0
            return weight * radii.get(i, 1.0) * unresolved / max(float(row["tokens"][7 + i]), 1.0)
        chosen = max(unknown, key=lambda i: (priority(i), -i))
        unknown.remove(chosen)
        selected.append(chosen)
    return selected


def _score(rows, radii, source_aware):
    records = []
    for row in rows:
        selected = _choose(row, radii, source_aware)
        relay = np.asarray(row["old_relay"], dtype=float).copy()
        for i in selected:
            relay[i] = row["new_relay"][i]
        predicted = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        records.append({
            "decision_exact": float(np.array_equal(predicted > TAU, truth > TAU)),
            "answer_action_exact": float(predicted.argmax() == truth.argmax()),
            "answer_accuracy": float(predicted.argmax() == row["gold"]),
            "tokens": float(sum(row["tokens"][7 + i] for i in selected)),
            "queries": float(len(selected)),
        })
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


def main():
    cells = []
    for worker in sorted(ROOT.rglob("WORKER*.jsonl")):
        gold = worker.parent / "SEALED.json"
        if not gold.exists():
            continue
        rows = parse_rows(worker, gold)
        masked_rows = copy.deepcopy(rows)
        calibration, test = _split(rows)
        for row in calibration + test:
            row["changed_rows"] = list(range(4))
        q = 0.9
        cell = {"worker": str(worker.relative_to(ROOT))}
        cell["graph_conditioned"] = _score(test, _row_radii(calibration, q), True)
        cell["pooled_radius"] = _score(test, _pooled_radius(calibration, q), True)
        cell["pooled_uniform_priority"] = _score(test, _pooled_radius(calibration, q), False)
        masked_calibration, masked_test = _split(masked_rows)
        cell["masked_graph_conditioned"] = _score(masked_test, _row_radii(masked_calibration, q), True)
        cell["masked_pooled_radius"] = _score(masked_test, _pooled_radius(masked_calibration, q), True)
        cell["masked_pooled_uniform_priority"] = _score(masked_test, _pooled_radius(masked_calibration, q), False)
        cells.append(cell)
    methods = ("graph_conditioned", "pooled_radius", "pooled_uniform_priority",
               "masked_graph_conditioned", "masked_pooled_radius",
               "masked_pooled_uniform_priority")
    mean = {method: {key: float(np.mean([cell[method][key] for cell in cells]))
                     for key in cells[0][method]}
            for method in methods}
    result = {
        "schema": "generic_answer_region_ablation_v1",
        "cell_count": len(cells),
        "quantile": 0.9,
        "mean": mean,
        "cells": cells,
        "interpretation": "Pooled-radius removes row-specific graph-conditioned drift calibration. Pooled-uniform-priority additionally removes source-aware cost ranking. The masked controls use the same trusted changed-row mask. The all-row gap includes a zero-radius interface-invariance assumption for rows with no observed calibration drift.",
    }
    out = HERE / "GENERIC_ANSWER_REGION_ABLATION.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

