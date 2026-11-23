"""P0-5/P0-6 audits for manifest-matched Pareto and quality transitions."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / "M60_STRONG_BASELINE_MATRIX.json"
SERVICE = HERE / "SERVICE_UTILITY_SUMMARY.json"


def transitions(rows, baseline: str, candidate: str):
    counts = {"corrected": 0, "damaged": 0, "unchanged_correct": 0, "unchanged_wrong": 0, "n": len(rows)}
    for row in rows:
        b = int(row["methods"][baseline]["action_error"])
        c = int(row["methods"][candidate]["action_error"])
        if b == 1 and c == 0:
            counts["corrected"] += 1
        elif b == 0 and c == 1:
            counts["damaged"] += 1
        elif b == 0 and c == 0:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    return counts


def pareto(aggregate):
    methods = list(aggregate)
    return [
        name for name in methods
        if not any(
            other != name
            and aggregate[other]["tokens"] <= aggregate[name]["tokens"]
            and aggregate[other]["action_error"] <= aggregate[name]["action_error"]
            and (
                aggregate[other]["tokens"] < aggregate[name]["tokens"]
                or aggregate[other]["action_error"] < aggregate[name]["action_error"]
            )
            for other in methods
        )
    ]


def main() -> None:
    matrix = json.loads(PRIVATE.read_text(encoding="utf-8"))
    rows = matrix["rows"]
    aggregate = matrix["aggregate"]
    methods = list(aggregate)
    budget_violations = []
    for row in rows:
        budgets = {row["methods"][m]["budget_tokens"] for m in methods}
        if len(budgets) != 1:
            budget_violations.append(row["key"])
        for method in methods:
            rec = row["methods"][method]
            if rec["tokens"] > rec["budget_tokens"] + 1e-9:
                budget_violations.append(row["key"] + ":" + method)
    transition = {
        baseline: transitions(rows, baseline, "full_ctdr")
        for baseline in ("exact_warm_cache", "reweighted_old_history", "max_uncertainty", "periodic", "random")
    }
    pareto_summary = {
        "frontier_methods": pareto(aggregate),
        "aggregate": {
            name: {"exact_set": float(value["exact_set"]), "action_error": float(value["action_error"]), "tokens": float(value["tokens"])}
            for name, value in aggregate.items()
        },
        "protocol": matrix["protocol"],
        "budget_violations": budget_violations,
    }
    quality_path = HERE / "P0_6_TASK_QUALITY.json"
    service_status = {
        "per_question_task_quality_available": quality_path.exists(),
        "reason": "private question-level service predictions and sealed labels are retained locally and summarized separately",
        "quality_transition_artifact": quality_path.name if quality_path.exists() else None,
    }
    if SERVICE.exists():
        service = json.loads(SERVICE.read_text(encoding="utf-8"))
        service_status["aggregate_methods"] = sorted(service.get("summary", {}))
        service_status["task_aggregate_available"] = sorted(service.get("by_task", {}))
    quality = {
        "schema": "mab2_p0_6_quality_transition_v1",
        "decision_action_transitions": transition,
        "interpretation": {
            "corrected": "baseline action error and full CTDR action error is zero",
            "damaged": "baseline action error is zero and full CTDR action error is nonzero",
            "unchanged_correct": "both actions agree with the sealed decision target",
            "unchanged_wrong": "both actions disagree with the sealed decision target",
        },
        "task_quality": service_status,
    }
    (HERE / "P0_5_MANIFEST_MATCHED_PARETO.json").write_text(json.dumps(pareto_summary, indent=2), encoding="utf-8")
    (HERE / "P0_6_QUALITY_TRANSITIONS.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    lines = [
        "# P0-5 manifest-matched Pareto audit", "",
        "All methods share the same graph-manifest affected-row set, old transcript, post-edit source, token prices, full-restart budget and exact-set stopping target.", "",
        "| method | exact set | action error | tokens | Pareto |", "| --- | ---: | ---: | ---: | --- |",
    ]
    front = set(pareto_summary["frontier_methods"])
    for name, value in pareto_summary["aggregate"].items():
        lines.append(f"| {name} | {value['exact_set']:.4f} | {value['action_error']:.4f} | {value['tokens']:.2f} | {'yes' if name in front else 'no'} |")
    lines += ["", f"Budget violations: {len(budget_violations)}.", "", "# P0-6 decision and task-quality transition audit", "", "The transition counts below refer to sealed decision-target agreement. They are not task-answer accuracy.", "", "| baseline | corrected | damaged | unchanged correct | unchanged wrong | n |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for baseline, counts in transition.items():
        lines.append(f"| {baseline} | {counts['corrected']} | {counts['damaged']} | {counts['unchanged_correct']} | {counts['unchanged_wrong']} | {counts['n']} |")
    lines += ["", "Per-question answer correctness transitions are computed separately from private WORKER*_score_auto.json rows and published in P0_6_TASK_QUALITY.md. They are not inferred from decision-set transitions."]
    (HERE / "P0_5_6_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"frontier": pareto_summary["frontier_methods"], "budget_violations": len(budget_violations), "transitions": transition}, indent=2))


if __name__ == "__main__":
    main()

