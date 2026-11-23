"""Pooled and leave-group-out quantile selection for Pairwise Gap Certificate."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from answer_safe_arcc import values
from pairwise_gap_certificate import parse_worker, radii, select_rows


ROOT = Path(__file__).resolve().parents[2]
QUANTILES = (0.5, 0.75, 0.9, 0.95, 1.0)


def identity(path: Path) -> dict:
    match = re.match(r"dedup_(qwen05|qwen14|llama3)_(commonsense|qasc|sciq)(?:_rows(1|2|4))?_pairwise\.json", path.name)
    if not match:
        raise ValueError(path.name)
    model, task, density = match.groups()
    return {"model": model, "task": task, "density": int(density or 1)}


def split(rows: list[dict], residue: int) -> list[dict]:
    return [r for r in rows if int(r["key"][:8], 16) % 5 == residue]


def evaluate(rows: list[dict], radius: dict[int, float]) -> dict:
    records = []
    for row in rows:
        selected = select_rows(row, radius)
        relay = row["old_relay"].copy()
        for index in selected:
            relay[index] = row["new_relay"][index]
        pred = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        records.append({
            "action": float(pred.argmax() == truth.argmax()),
            "tokens": float(sum(row["tokens"][7 + i] for i in selected)),
        })
    return {"action": float(np.mean([r["action"] for r in records])), "tokens": float(np.mean([r["tokens"] for r in records]))}


def load_cells() -> list[dict]:
    cells = []
    for output in sorted(ROOT.glob("dedup_*_pairwise.json")):
        info = identity(output)
        worker = ROOT / json.loads(output.read_text(encoding="utf-8"))["worker"]
        rows = parse_worker(worker)
        calibration = [r for residue in (0, 1, 2) for r in split(rows, residue)]
        cells.append({**info, "rows": rows, "calibration": calibration})
    return cells


def select_q(cells: list[dict]) -> dict:
    candidates = []
    for q in QUANTILES:
        selection = [evaluate(split(cell["rows"], 3), radii(cell["calibration"], q)) for cell in cells]
        candidates.append({"q": q, "action": float(np.mean([x["action"] for x in selection])), "tokens": float(np.mean([x["tokens"] for x in selection]))})
    feasible = [x for x in candidates if x["action"] >= 1.0]
    return min(feasible or candidates, key=lambda x: (x["tokens"], x["q"])) | {"candidates": candidates}


def evaluate_group_out(cells: list[dict], key: str, values: list[str | int]) -> list[dict]:
    out = []
    for held in values:
        train = [cell for cell in cells if cell[key] != held]
        target = [cell for cell in cells if cell[key] == held]
        chosen = select_q(train)
        reports = []
        for cell in target:
            score = evaluate(split(cell["rows"], 4), radii(cell["calibration"], chosen["q"]))
            reports.append({"model": cell["model"], "task": cell["task"], "density": cell["density"], **score})
        out.append({"held_out": held, "selection": chosen, "test": reports})
    return out


def main() -> None:
    cells = load_cells()
    pooled = select_q(cells)
    pooled_test = []
    for cell in cells:
        pooled_test.append({
            "model": cell["model"],
            "task": cell["task"],
            "density": cell["density"],
            **evaluate(split(cell["rows"], 4), radii(cell["calibration"], pooled["q"])),
        })
    payload = {
        "schema": "pairwise_gap_global_selection_v1",
        "pooled": {"selection": pooled, "test": pooled_test},
        "leave_one_task_out": evaluate_group_out(cells, "task", sorted({cell["task"] for cell in cells})),
        "leave_one_model_out": evaluate_group_out(cells, "model", sorted({cell["model"] for cell in cells})),
    }
    out = Path(__file__).with_name("PAIRWISE_GAP_GLOBAL_SELECTION.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"pooled_q": pooled["q"], "pooled_action": float(np.mean([x["action"] for x in pooled_test])), "pooled_tokens": float(np.mean([x["tokens"] for x in pooled_test])), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()

