"""Describe answer-region query cost as a function of answer margin."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from answer_region_only import values, parse_rows
from answer_safe_arcc import values
from generic_answer_region_ablation import _choose, _row_radii, ROOT, _split


BINS = ((0.0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, float("inf")))


def main():
    records = []
    for worker in sorted(ROOT.rglob("WORKER*.jsonl")):
        gold = worker.parent / "SEALED.json"
        if not gold.exists():
            continue
        rows = parse_rows(worker, gold)
        calibration, test = _split(rows)
        calibration = [dict(row) for row in calibration]
        test = [dict(row) for row in test]
        for row in calibration + test:
            row["changed_rows"] = [0, 1, 2, 3]
        for q in (0.9, 1.0):
            radius = _row_radii(calibration, q)
            for row in test:
                selected = _choose(row, radius, True)
                relay = np.asarray(row["old_relay"], dtype=float).copy()
                for index in selected:
                    relay[index] = row["new_relay"][index]
                truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
                ordered = np.sort(truth)
                margin = float(ordered[-1] - ordered[-2]) if ordered.size > 1 else float("inf")
                records.append({"q": q, "margin": margin,
                                "tokens": float(sum(row["tokens"][7 + i] for i in selected)),
                                "queries": float(len(selected)),
                                "action_exact": float(values(row["new_source"], row["base"], relay, row["labels"]).argmax() == truth.argmax())})
    result = {"schema": "answer_region_margin_stratification_v1", "bins": []}
    for low, high in BINS:
        for q in (0.9, 1.0):
            subset = [r for r in records if r["q"] == q and low <= r["margin"] < high]
            result["bins"].append({"q": q, "margin_low": low, "margin_high": high,
                                   "n": len(subset),
                                   "tokens": float(np.mean([r["tokens"] for r in subset])) if subset else None,
                                   "queries": float(np.mean([r["queries"] for r in subset])) if subset else None,
                                   "action_exact": float(np.mean([r["action_exact"] for r in subset])) if subset else None})
    out = Path(__file__).with_name("MARGIN_STRATIFICATION.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

