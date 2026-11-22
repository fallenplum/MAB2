"""Aggregate answer-safe revalidation against matched historical baselines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def mean(records, key):
    values = [float(r[key]) for r in records]
    return float(np.mean(values)) if values else None


def main():
    rows = []
    for safe_path in ROOT.rglob("WORKER*_asafe.json"):
        if safe_path.name.endswith("_score_auto.json"):
            continue
        score_path = safe_path.with_name(safe_path.name.replace("_asafe.json", "_score_auto.json"))
        if not score_path.exists():
            continue
        safe = json.loads(safe_path.read_text(encoding="utf-8"))
        score = json.loads(score_path.read_text(encoding="utf-8"))
        test_rows = [r for r in score["rows"] if int(r["key"][:8], 16) % 5 == 4]
        baseline = {}
        for method in ("stale", "max_uncertainty", "tcr_guard", "tcr_bound"):
            records = [r["methods"][method] for r in test_rows]
            baseline[method] = {
                "decision_exact": mean(records, "decision_exact"),
                "answer_action_exact": mean(records, "answer_action_exact"),
                "answer_accuracy": mean(records, "answer_accuracy_corrected"),
                "tokens": mean(records, "tokens"),
                "queries": mean(records, "queries"),
            }
        rel = str(safe_path.parent.relative_to(ROOT)).replace("\\", "/")
        rows.append({
            "cohort": rel,
            "worker": safe_path.stem.replace("_asafe", ""),
            "n": int(safe["test_n"]),
            "answer_safe": safe["aggregate"],
            "baselines": baseline,
        })
    rows.sort(key=lambda r: (r["cohort"], r["worker"]))
    out = ROOT / "experiments" / "mechanism_20261122" / "ARCC_ANSWER_SAFE_HISTORICAL_RESULTS.json"
    out.write_text(json.dumps({"schema": "arcc_answer_safe_historical_v1", "cells": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"cells": len(rows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()

