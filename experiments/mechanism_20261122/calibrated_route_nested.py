"""Nested development selection for mask versus answer-region routing.

The radius and route are selected on two development partitions.  A route is
eligible only when it recovers both the edge decision set and final answer
action on each partition.  The held-out fifth partition is evaluated once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from answer_region_only import choose, parse_rows, radii
from answer_safe_arcc import TAU, values


def split(rows, residue):
    return [r for r in rows if int(r["key"][:8], 16) % 5 in residue]


def evaluate(rows, radius, route):
    rec = []
    for row in rows:
        if route == "mask":
            selected = sorted(set(int(i) for i in row.get("changed_rows", [])))
        else:
            probe = dict(row)
            probe["changed_rows"] = [0, 1, 2, 3]
            selected = choose(probe, radius)
        relay = np.asarray(row["old_relay"], dtype=float).copy()
        for i in selected:
            relay[i] = row["new_relay"][i]
        pred = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        rec.append({"decision_exact": float(np.array_equal(pred > TAU, truth > TAU)),
                    "answer_action_exact": float(pred.argmax() == truth.argmax()),
                    "answer_accuracy": float(pred.argmax() == row["gold"]),
                    "tokens": float(sum(row["tokens"][7 + int(i)] for i in selected)),
                    "queries": float(len(selected))})
    return {k: float(np.mean([r[k] for r in rec])) for k in rec[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = parse_rows(args.worker, args.gold)
    train, valid, test = split(rows, {0, 1}), split(rows, {2, 3}), split(rows, {4})
    candidates = []
    for q in (0.5, 0.75, 0.9, 0.95, 1.0):
        radius = radii(train, q)
        for route in ("mask", "answer_region"):
            candidates.append({"q": q, "route": route, "radius": radius,
                               "train": evaluate(train, radius, route),
                               "valid": evaluate(valid, radius, route)})
    feasible = [c for c in candidates if c["train"]["decision_exact"] >= 1.0
                and c["train"]["answer_action_exact"] >= 1.0
                and c["valid"]["decision_exact"] >= 1.0
                and c["valid"]["answer_action_exact"] >= 1.0]
    selected = sorted(feasible or candidates,
                      key=lambda c: (c["valid"]["tokens"], c["valid"]["queries"],
                                     -c["valid"]["answer_action_exact"], c["q"]))[0]
    result = {"schema": "calibrated_route_nested_v1", "worker": args.worker.name,
              "train_n": len(train), "valid_n": len(valid), "test_n": len(test),
              "selected": selected, "test": evaluate(test, selected["radius"], selected["route"]),
              "candidates": candidates}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

