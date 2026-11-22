"""Answer-region-only revalidation policy for the ARCC cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from answer_safe_arcc import TAU, values, validate_conditional_full


def choose(row, radii):
    source = np.asarray(row["new_source"], dtype=float)
    base = np.asarray(row["base"], dtype=float)
    old_relay = np.asarray(row["old_relay"], dtype=float)
    new_relay = np.asarray(row["new_relay"], dtype=float)
    labels = np.asarray(row["labels"], dtype=int)
    unknown = set(int(i) for i in row.get("changed_rows", []))
    selected = []
    while unknown:
        n_labels = int(np.max(labels)) + 1
        center = np.zeros(n_labels)
        low = np.zeros(n_labels)
        high = np.zeros(n_labels)
        for i in range(4):
            relay = old_relay[i] if i in unknown else new_relay[i]
            contribution = source[i] * (relay - base)
            radius = source[i] * radii.get(i, 1.0) if i in unknown else 0.0
            for label in range(n_labels):
                mass = float(contribution[labels[i] == label].sum())
                center[label] += mass
                low[label] += mass - radius
                high[label] += mass + radius
        answer_order = int(np.argmax(center))
        competitors = np.delete(high, answer_order)
        answer_safe = competitors.size == 0 or bool(low[answer_order] > np.max(competitors))
        if answer_safe:
            break
        unresolved = float(np.sum(~((low > TAU) | (high < TAU)))) + float(not answer_safe)
        chosen = max(unknown, key=lambda i: (source[i] * radii.get(i, 1.0) * unresolved /
                                              max(float(row["tokens"][7 + i]), 1.0), -i))
        unknown.remove(chosen)
        selected.append(chosen)
    return selected


def parse_rows(worker, gold_path):
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    rows = []
    for line in worker.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        parsed = validate_conditional_full(item["full"])
        old_source, new_source, base = parsed["source"]
        raw = int(gold[item["key"]])
        rows.append({**parsed, "old_source": old_source, "new_source": new_source, "base": base,
                     "key": item["key"], "tokens": item["tokens"],
                     "changed_rows": item.get("changed_rows", []),
                     "gold": 1 if raw == -15 else 3 if raw == -13 else raw})
    return rows


def radii(rows, q):
    samples = {}
    for row in rows:
        for i in row["changed_rows"]:
            value = 0.5 * np.abs(row["new_relay"][int(i)] - row["old_relay"][int(i)]).sum()
            samples.setdefault(int(i), []).append(float(value))
    return {i: float(np.quantile(v, q)) for i, v in samples.items()}


def score(rows, radius):
    out = []
    for row in rows:
        selected = choose(row, radius)
        relay = np.asarray(row["old_relay"], dtype=float).copy()
        for i in selected:
            relay[i] = row["new_relay"][i]
        predicted = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        out.append({"decision_exact": float(np.array_equal(predicted > TAU, truth > TAU)),
                    "answer_action_exact": float(predicted.argmax() == truth.argmax()),
                    "answer_accuracy": float(predicted.argmax() == row["gold"]),
                    "tokens": float(sum(row["tokens"][7 + i] for i in selected)),
                    "queries": float(len(selected))})
    return {k: float(np.mean([x[k] for x in out])) for k in out[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quantile", type=float, default=None,
                   help="Fixed calibration quantile. Omit only for development selection.")
    p.add_argument("--mask", choices=("changed", "all"), default="changed")
    args = p.parse_args()
    rows = parse_rows(args.worker, args.gold)
    if args.mask == "all":
        for row in rows:
            row["changed_rows"] = [0, 1, 2, 3]
    calibration = [r for r in rows if int(r["key"][:8], 16) % 5 < 3]
    selection = [r for r in rows if int(r["key"][:8], 16) % 5 == 3]
    test = [r for r in rows if int(r["key"][:8], 16) % 5 == 4]
    candidates = []
    for q in (0.5, 0.75, 0.9, 0.95, 1.0):
        radius = radii(calibration, q)
        candidates.append({"q": q, "selection": score(selection, radius), "radius": radius})
    if args.quantile is None:
        best = sorted(candidates, key=lambda x: (-x["selection"]["answer_accuracy"],
                                                   -x["selection"]["answer_action_exact"],
                                                   x["selection"]["tokens"], x["q"]))[0]
        selected_q = best["q"]
    else:
        selected_q = float(args.quantile)
        best = {"q": selected_q, "radius": radii(calibration, selected_q),
                "selection": score(selection, radii(calibration, selected_q))}
    test_candidates = [{"q": q, "aggregate": score(test, radii(calibration, q))} for q in (0.5, 0.75, 0.9, 0.95, 1.0)]
    result = {"schema": "arcc_answer_region_only_v1", "worker": args.worker.name,
              "calibration_n": len(calibration), "selection_n": len(selection), "test_n": len(test),
              "candidates": candidates, "test_candidates": test_candidates, "selected": selected_q,
              "mask": args.mask, "fixed_quantile": args.quantile,
              "aggregate": score(test, radii(calibration, selected_q))}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

