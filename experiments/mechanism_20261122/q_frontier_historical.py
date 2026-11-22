"""Evaluate the fixed radius frontier on all historical worker cells."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from answer_region_only import parse_rows, radii, score

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
QS = (0.5, 0.75, 0.9, 0.95, 1.0)


def main():
    cells = []
    for worker in sorted(ROOT.rglob("WORKER*.jsonl")):
        gold = worker.parent / "SEALED.json"
        if not gold.exists():
            continue
        rows = parse_rows(worker, gold)
        for row in rows:
            row["changed_rows"] = list(range(4))
        key = lambda row: int(row["key"][:8], 16) % 5
        calibration = [row for row in rows if key(row) < 3]
        test = [row for row in rows if key(row) == 4]
        cells.append({
            "worker": str(worker.relative_to(ROOT)),
            "frontier": {str(q): score(test, radii(calibration, q)) for q in QS},
        })
    mean = {str(q): {name: float(np.mean([cell["frontier"][str(q)][name] for cell in cells]))
                     for name in cells[0]["frontier"][str(q)]}
            for q in QS}
    result = {"schema": "historical_answer_region_q_frontier_v1",
              "cell_count": len(cells), "quantiles": list(QS), "mean": mean,
              "cells": cells}
    (HERE / "Q_FRONTIER_HISTORICAL.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Historical answer-region radius frontier", "", "| q | decision exact | action exact | accuracy | relay tokens | queries |", "|---:|---:|---:|---:|---:|---:|"]
    for q in QS:
        row = mean[str(q)]
        lines.append("| {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:.3f} |".format(q, row["decision_exact"], row["answer_action_exact"], row["answer_accuracy"], row["tokens"], row["queries"]))
    (HERE / "Q_FRONTIER_HISTORICAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()

