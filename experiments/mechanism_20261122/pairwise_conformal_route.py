"""Split-conformal Pairwise Gap Certificate with an explicit miscoverage level."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from answer_safe_arcc import values
from pairwise_gap_certificate import parse_worker, select_rows


ROOT = Path(__file__).resolve().parents[2]


def split(rows: list[dict], residue: int) -> list[dict]:
    return [row for row in rows if int(row["key"][:8], 16) % 5 == residue]


def tv_score(row: dict, index: int) -> float:
    return float(0.5 * np.abs(row["new_relay"][index] - row["old_relay"][index]).sum())


def conformal_radius(rows: list[dict], alpha: float) -> dict[int, float]:
    radii = {}
    for index in range(4):
        scores = sorted(tv_score(row, index) for row in rows if index in row["changed_rows"])
        if not scores:
            radii[index] = 1.0
            continue
        rank = max(1, min(len(scores), int(math.ceil((len(scores) + 1) * (1.0 - alpha / 4.0)))))
        radii[index] = float(scores[rank - 1])
    return radii


def evaluate(rows: list[dict], radius: dict[int, float]) -> dict:
    records = []
    for row in rows:
        selected = select_rows(row, radius)
        relay = row["old_relay"].copy()
        for index in selected:
            relay[index] = row["new_relay"][index]
        pred = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        covered = [tv_score(row, index) <= radius.get(index, 1.0) + 1e-12 for index in row["changed_rows"]]
        records.append({
            "action_exact": float(pred.argmax() == truth.argmax()),
            "tokens": float(sum(row["tokens"][7 + index] for index in selected)),
            "queries": float(len(selected)),
            "radius_coverage": float(np.mean(covered)) if covered else 1.0,
            "all_rows_covered": float(all(covered)),
        })
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


def parse_name(path: Path) -> dict:
    match = re.match(r"dedup_(qwen05|qwen14|llama3)_(commonsense|qasc|sciq)(?:_rows(1|2|4))?_pairwise\.json", path.name)
    if not match:
        raise ValueError(path.name)
    model, task, density = match.groups()
    return {"model": model, "task": task, "density": int(density or 1)}


def main() -> None:
    alpha = 0.1
    records = []
    for output in sorted(ROOT.glob("dedup_*_pairwise.json")):
        info = parse_name(output)
        worker = ROOT / json.loads(output.read_text(encoding="utf-8"))["worker"]
        rows = parse_worker(worker)
        calibration = [row for residue in (0, 1, 2) for row in split(rows, residue)]
        result = evaluate(split(rows, 4), conformal_radius(calibration, alpha))
        records.append({**info, **result})
    payload = {
        "schema": "pairwise_split_conformal_route_v1",
        "alpha": alpha,
        "rowwise_bonferroni": True,
        "claim": "Under row-wise exchangeability, the split-conformal radii cover all four changed-row TV scores with marginal failure controlled by alpha under the stated calibration contract. The action certificate is conditional on all row bounds holding.",
        "records": records,
        "aggregate": {key: float(np.mean([row[key] for row in records])) for key in records[0] if key not in {"model", "task", "density"}},
    }
    out = Path(__file__).with_name("PAIRWISE_CONFORMAL_ROUTE.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()

