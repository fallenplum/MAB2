"""Per-question service-quality transitions from sealed score artifacts."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
METHODS = ("warm_reweighted", "max_uncertainty", "source_only_ctdr", "tcr_guard", "cold_restart")


def transition(rows, baseline, candidate, field):
    out = {"corrected": 0, "damaged": 0, "unchanged_correct": 0, "unchanged_wrong": 0, "n": len(rows)}
    for row in rows:
        b = float(row["methods"][baseline][field])
        c = float(row["methods"][candidate][field])
        if b < 0.5 and c >= 0.5:
            out["corrected"] += 1
        elif b >= 0.5 and c < 0.5:
            out["damaged"] += 1
        elif b >= 0.5 and c >= 0.5:
            out["unchanged_correct"] += 1
        else:
            out["unchanged_wrong"] += 1
    return out


def main() -> None:
    files = sorted(HERE.glob("m60_conditional_*/WORKER*_score_auto.json"))
    rows = []
    by_cell = defaultdict(list)
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if all(method in row.get("methods", {}) for method in METHODS):
                rows.append(row)
                by_cell[path.parent.name].append(row)
    results = {
        "schema": "mab2_p0_6_task_quality_v1",
        "n_score_files": len(files),
        "n_question_rows": len(rows),
        "methods": METHODS,
        "transitions": {
            baseline: {
                "answer_accuracy_corrected": transition(rows, baseline, "tcr_guard", "answer_accuracy_corrected"),
                "answer_action_exact": transition(rows, baseline, "tcr_guard", "answer_action_exact"),
            }
            for baseline in ("warm_reweighted", "max_uncertainty", "source_only_ctdr")
        },
        "by_cell_answer_accuracy": {
            cell: {
                method: sum(float(row["methods"][method]["answer_accuracy_corrected"]) for row in cell_rows) / len(cell_rows)
                for method in METHODS
            }
            for cell, cell_rows in sorted(by_cell.items())
        },
        "scope": "sealed per-question score artifacts, answer accuracy is separate from decision-set exactness",
    }
    (OUT / "P0_6_TASK_QUALITY.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# P0-6 per-question task-quality audit", "",
        f"Scored question rows: {len(rows)} across {len(files)} score files.", "",
        "Corrected means baseline answer incorrect and candidate answer correct. Damaged means baseline answer correct and candidate answer incorrect.", "",
        "| baseline -> TCR guard | corrected | damaged | unchanged correct | unchanged wrong | n |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline, metrics in results["transitions"].items():
        c = metrics["answer_accuracy_corrected"]
        lines.append(f"| {baseline} -> tcr_guard | {c['corrected']} | {c['damaged']} | {c['unchanged_correct']} | {c['unchanged_wrong']} | {c['n']} |")
    lines += ["", "The answer-accuracy transitions are computed from per-question sealed score artifacts. They are not inferred from decision-set transitions."]
    (OUT / "P0_6_TASK_QUALITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results["transitions"], indent=2))


if __name__ == "__main__":
    main()

