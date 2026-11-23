"""Risk-gated route selection using an answer-margin certificate proxy.

The gate estimates whether the graph-descendant mask leaves an answer margin
smaller than its calibrated conditional-drift mass.  It uses cached old relay
values only.  The answer-region route is activated only above a development-
selected risk threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from answer_region_only import choose, parse_rows, radii
from answer_safe_arcc import TAU, values


def risk_score(row, radius):
    source = np.asarray(row["new_source"], dtype=float)
    base = np.asarray(row["base"], dtype=float)
    old_relay = np.asarray(row["old_relay"], dtype=float)
    labels = np.asarray(row["labels"], dtype=int)
    score = values(source, base, old_relay, labels)
    ordered = np.sort(score)[::-1]
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float("inf")
    uncertainty = float(sum(source[int(i)] * radius.get(int(i), 1.0)
                            for i in row.get("changed_rows", [])))
    return uncertainty / max(margin, 1e-9)


def evaluate(rows, radius, threshold):
    rec = []
    for row in rows:
        mask = sorted(set(int(i) for i in row.get("changed_rows", [])))
        probe = dict(row)
        probe["changed_rows"] = [0, 1, 2, 3]
        answer_region = choose(probe, radius)
        if risk_score(row, radius) >= threshold:
            selected, route = answer_region, "answer_region"
        else:
            selected, route = mask, "mask"
        relay = np.asarray(row["old_relay"], dtype=float).copy()
        for i in selected:
            relay[i] = row["new_relay"][i]
        pred = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        rec.append({"route": route, "risk": risk_score(row, radius),
                    "decision_exact": float(np.array_equal(pred > TAU, truth > TAU)),
                    "answer_action_exact": float(pred.argmax() == truth.argmax()),
                    "answer_accuracy": float(pred.argmax() == row["gold"]),
                    "tokens": float(sum(row["tokens"][7 + int(i)] for i in selected)),
                    "queries": float(len(selected))})
    return {**{k: float(np.mean([r[k] for r in rec])) for k in rec[0] if k not in {"route"}},
            "route_fraction_answer_region": float(np.mean([r["route"] == "answer_region" for r in rec])),
            "route_n": {name: sum(r["route"] == name for r in rec) for name in ("mask", "answer_region")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = parse_rows(args.worker, args.gold)
    train = [r for r in rows if int(r["key"][:8], 16) % 5 < 3]
    test = [r for r in rows if int(r["key"][:8], 16) % 5 == 4]
    candidates = []
    for q in (0.5, 0.75, 0.9, 0.95, 1.0):
        radius = radii(train, q)
        for threshold in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")):
            candidates.append({"q": q, "threshold": threshold, "radius": radius,
                               "calibration": evaluate(train, radius, threshold)})
    feasible = [c for c in candidates if c["calibration"]["decision_exact"] >= 1.0
                and c["calibration"]["answer_action_exact"] >= 1.0]
    selected = sorted(feasible or candidates,
                      key=lambda c: (c["calibration"]["tokens"], c["calibration"]["queries"], c["q"]))[0]
    result = {"schema": "risk_gated_route_v1", "worker": args.worker.name,
              "calibration_n": len(train), "test_n": len(test), "selected": selected,
              "test": evaluate(test, selected["radius"], selected["threshold"]),
              "candidates": candidates}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

