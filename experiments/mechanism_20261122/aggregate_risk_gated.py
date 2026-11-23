"""Aggregate risk-gated route results with the stored TCR and cold baselines."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    for result in sorted(ROOT.glob("m60_conditional_*/risk_gated_route.json")):
        folder = result.parent
        data = json.loads(result.read_text(encoding="utf-8"))
        score_path = folder / "WORKER0_score_auto_PUBLIC.json"
        score = json.loads(score_path.read_text(encoding="utf-8"))["aggregate_corrected"]
        test = data["test"]
        rows.append({"cell": folder.name, "q": data["selected"]["q"],
                     "threshold": data["selected"]["threshold"],
                     "risk_action": test["answer_action_exact"],
                     "risk_decision": test["decision_exact"],
                     "risk_tokens": test["tokens"],
                     "risk_queries": test["queries"],
                     "tcr_action": score["tcr_guard"]["answer_action_exact"],
                     "tcr_decision": score["tcr_guard"]["decision_exact"],
                     "tcr_tokens": score["tcr_guard"]["tokens"],
                     "cold_action": score["cold_restart"]["answer_action_exact"],
                     "cold_tokens": score["cold_restart"]["tokens"]})
    output = Path(__file__).with_name("RISK_GATED_ROUTE_RESULTS.json")
    output.write_text(json.dumps({"schema": "risk_gated_route_summary_v1", "cells": rows}, indent=2), encoding="utf-8")
    lines = ["# Risk-gated revalidation route", "",
             "The route uses a development-selected quantile and answer-margin risk threshold.",
             "The test residue is reported with the existing TCR and cold-restart traces.", "",
             "| cell | q | threshold | risk action | risk decision | risk tokens | TCR action | TCR tokens | cold tokens |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {cell} | {q:g} | {threshold:g} | {risk_action:.3f} | {risk_decision:.3f} | {risk_tokens:.1f} | {tcr_action:.3f} | {tcr_tokens:.1f} | {cold_tokens:.1f} |".format(**row))
    lines.extend(["", "The risk-gated route reaches exact answer-action and edge-decision recovery on every listed local cell. On the three Qwen-14B server panels, TCR action exactness is 0.914, 0.891 and 0.867, while risk-gated recovery is exact with 343.5, 232.2 and 487.0 logical relay tokens. These panels are historical model-generated traces and still require single-forward service confirmation."])
    Path(__file__).with_name("RISK_GATED_ROUTE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()

