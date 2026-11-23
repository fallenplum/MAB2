"""Conservative route gate that minimizes the number of new relay forwards.

The graph-descendant mask is the default.  The answer-region route is used
only when its old-transcript certificate schedules strictly fewer relay rows
than the mask route.  Selection uses no new relay values or sealed labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from answer_region_only import choose, parse_rows, radii
from answer_safe_arcc import TAU, values


def route_for(row, radius):
    mask = sorted(set(int(i) for i in row.get("changed_rows", [])))
    probe = dict(row)
    probe["changed_rows"] = [0, 1, 2, 3]
    answer_region = choose(probe, radius)
    if len(answer_region) < len(mask):
        return answer_region, "answer_region"
    return mask, "mask"


def evaluate(rows, radius):
    rec = []
    for row in rows:
        selected, route = route_for(row, radius)
        relay = np.asarray(row["old_relay"], dtype=float).copy()
        for i in selected:
            relay[i] = row["new_relay"][i]
        pred = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        rec.append({"route": route, "decision_exact": float(np.array_equal(pred > TAU, truth > TAU)),
                    "answer_action_exact": float(pred.argmax() == truth.argmax()),
                    "answer_accuracy": float(pred.argmax() == row["gold"]),
                    "tokens": float(sum(row["tokens"][7 + int(i)] for i in selected)),
                    "queries": float(len(selected))})
    return {**{k: float(np.mean([r[k] for r in rec])) for k in rec[0] if k != "route"},
            "route_fraction_answer_region": float(np.mean([r["route"] == "answer_region" for r in rec])),
            "route_n": {name: sum(r["route"] == name for r in rec) for name in ("mask", "answer_region")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quantile", type=float, default=0.5)
    args = p.parse_args()
    rows = parse_rows(args.worker, args.gold)
    calibration = [r for r in rows if int(r["key"][:8], 16) % 5 < 3]
    test = [r for r in rows if int(r["key"][:8], 16) % 5 == 4]
    radius = radii(calibration, args.quantile)
    out = {"schema": "adaptive_min_calls_v1", "worker": args.worker.name,
           "quantile": args.quantile, "test_n": len(test), "radius": radius,
           "test": evaluate(test, radius)}
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

