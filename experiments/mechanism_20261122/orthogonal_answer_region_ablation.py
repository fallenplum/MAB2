"""Offline orthogonal ablations for the answer-region controller.

All variants use the same cached transcript and test split. The only changes
are the radius source and whether token cost enters the next-row ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from answer_safe_arcc import TAU, values, validate_conditional_full


ROOT = Path(__file__).resolve().parents[2]


def parse_worker(worker, score_path):
    scored = json.loads(score_path.read_text(encoding="utf-8"))
    gold = {str(row["key"]): int(row["corrected_gold"]) for row in scored["rows"]}
    rows = []
    for line in worker.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        old_source, new_source, base = validate_conditional_full(item["full"])["source"]
        parsed = validate_conditional_full(item["full"])
        rows.append({
            **parsed,
            "old_source": old_source,
            "new_source": new_source,
            "base": base,
            "key": item["key"],
            "tokens": item["tokens"],
            "changed_rows": list(item.get("changed_rows", [])),
            "gold": gold[item["key"]],
        })
    return rows


def make_radii(rows, q, pooled=False):
    samples = []
    by_row = {}
    for row in rows:
        for index in row["changed_rows"]:
            value = 0.5 * np.abs(row["new_relay"][int(index)] - row["old_relay"][int(index)]).sum()
            samples.append(float(value))
            by_row.setdefault(int(index), []).append(float(value))
    if pooled:
        radius = float(np.quantile(samples, q)) if samples else 1.0
        return {index: radius for index in range(4)}
    return {index: float(np.quantile(values_, q)) for index, values_ in by_row.items()}


def choose_variant(row, radii, cost_aware=True):
    source = np.asarray(row["new_source"], dtype=float)
    base = np.asarray(row["base"], dtype=float)
    old_relay = np.asarray(row["old_relay"], dtype=float)
    new_relay = np.asarray(row["new_relay"], dtype=float)
    labels = np.asarray(row["labels"], dtype=int)
    unknown = set(int(index) for index in row["changed_rows"])
    selected = []
    while unknown:
        n_labels = int(np.max(labels)) + 1
        center = np.zeros(n_labels)
        low = np.zeros(n_labels)
        high = np.zeros(n_labels)
        for index in range(4):
            relay = old_relay[index] if index in unknown else new_relay[index]
            contribution = source[index] * (relay - base)
            radius = source[index] * radii.get(index, 1.0) if index in unknown else 0.0
            for label in range(n_labels):
                mass = float(contribution[labels[index] == label].sum())
                center[label] += mass
                low[label] += mass - radius
                high[label] += mass + radius
        answer = int(np.argmax(center))
        competitors = np.delete(high, answer)
        safe = competitors.size == 0 or bool(low[answer] > np.max(competitors))
        if safe:
            break
        unresolved = float(np.sum(~((low > TAU) | (high < TAU)))) + float(not safe)
        def score(index):
            uncertainty = source[index] * radii.get(index, 1.0) * unresolved
            if cost_aware:
                uncertainty /= max(float(row["tokens"][7 + index]), 1.0)
            return uncertainty, -index
        chosen = max(unknown, key=score)
        unknown.remove(chosen)
        selected.append(chosen)
    return selected


def score(rows, radii, cost_aware):
    records = []
    for row in rows:
        selected = choose_variant(row, radii, cost_aware)
        relay = np.asarray(row["old_relay"], dtype=float).copy()
        for index in selected:
            relay[index] = row["new_relay"][index]
        predicted = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        records.append({
            "decision_exact": float(np.array_equal(predicted > TAU, truth > TAU)),
            "answer_action_exact": float(predicted.argmax() == truth.argmax()),
            "answer_accuracy": float(predicted.argmax() == row["gold"]),
            "tokens": float(sum(row["tokens"][7 + index] for index in selected)),
            "queries": float(len(selected)),
        })
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


def main():
    cells = []
    for worker in sorted(ROOT.rglob("WORKER*.jsonl")):
        score_path = worker.with_name(worker.stem + "_score_auto.json")
        if not score_path.exists():
            continue
        rows = parse_worker(worker, score_path)
        calibration = [row for row in rows if int(row["key"][:8], 16) % 5 < 3]
        test = [row for row in rows if int(row["key"][:8], 16) % 5 == 4]
        for mask in ("changed", "all"):
            # Calibrate on the observed graph-change rows before expanding the
            # test mask. Missing rows in an all-row test remain conservative
            # through choose_variant's default radius of one.
            graph = make_radii(calibration, 0.9, pooled=False)
            pooled = make_radii(calibration, 0.9, pooled=True)
            work_test = [dict(row) for row in test]
            if mask == "all":
                for row in work_test:
                    row["changed_rows"] = [0, 1, 2, 3]
            for name, radii, cost_aware in (
                ("graph_radius_cost", graph, True),
                ("pooled_radius_cost", pooled, True),
                ("graph_radius_uncertainty", graph, False),
            ):
                cells.append({"cell": str(worker.relative_to(ROOT)), "mask": mask,
                              "method": name, "metrics": score(work_test, radii, cost_aware)})
    methods = sorted({cell["method"] for cell in cells})
    result = {"schema": "orthogonal_answer_region_ablation_v1", "cells": cells,
              "mean": {method: {mask: {key: float(np.mean([
                  c["metrics"][key] for c in cells if c["method"] == method and c["mask"] == mask]))
                  for key in cells[0]["metrics"]} for mask in ("changed", "all")}
                        for method in methods}}
    out = Path(__file__).with_name("ORTHOGONAL_ANSWER_REGION_ABLATION.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()

