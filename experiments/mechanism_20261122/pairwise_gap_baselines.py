"""Matched row-order controls for the Pairwise Gap Certificate."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from answer_safe_arcc import values
from pairwise_gap_certificate import parse_worker, radii


def select_rows(row: dict, radius: dict[int, float], policy: str) -> list[int]:
    unknown = set(row["changed_rows"])
    selected = []
    rng = random.Random(int(row["key"][:8], 16))
    while unknown:
        relay = row["old_relay"].copy()
        for index in selected:
            relay[index] = row["new_relay"][index]
        score = values(row["new_source"], row["base"], relay, row["labels"])
        answer = int(np.argmax(score))
        uncertainty = sum(2.0 * row["new_source"][i] * radius.get(i, 1.0) for i in unknown)
        safe = all(float(score[answer] - score[b] - uncertainty) > 0.0 for b in range(len(score)) if b != answer)
        if safe:
            return selected
        if policy == "pairwise_gap":
            index = max(unknown, key=lambda i: (2.0 * row["new_source"][i] * radius.get(i, 1.0) / max(float(row["tokens"][7 + i]), 1.0), -i))
        elif policy == "max_uncertainty":
            index = max(unknown, key=lambda i: (2.0 * row["new_source"][i] * radius.get(i, 1.0), -i))
        elif policy == "periodic":
            index = min(unknown)
        elif policy == "random":
            index = rng.choice(sorted(unknown))
        else:
            raise ValueError(policy)
        unknown.remove(index)
        selected.append(index)
    return selected


def evaluate(rows: list[dict], radius: dict[int, float], policy: str) -> dict:
    records = []
    for row in rows:
        selected = select_rows(row, radius, policy)
        relay = row["old_relay"].copy()
        for index in selected:
            relay[index] = row["new_relay"][index]
        pred = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        records.append({"action_exact": float(pred.argmax() == truth.argmax()), "tokens": float(sum(row["tokens"][7 + i] for i in selected)), "queries": float(len(selected))})
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    rows_out = []
    for output in sorted(root.glob("dedup_*_pairwise.json")):
        data = json.loads(output.read_text(encoding="utf-8"))
        rows = parse_worker(root / data["worker"])
        calibration = [row for residue in (0, 1, 2) for row in rows if int(row["key"][:8], 16) % 5 == residue]
        test = [row for row in rows if int(row["key"][:8], 16) % 5 == 4]
        radius = radii(calibration, data["selected"]["q"])
        for policy in ("pairwise_gap", "max_uncertainty", "periodic", "random"):
            rows_out.append({"cell": output.stem, "policy": policy, **evaluate(test, radius, policy)})
    payload = {"schema": "pairwise_gap_matched_baselines_v1", "records": rows_out}
    out = Path(__file__).with_name("PAIRWISE_GAP_MATCHED_BASELINES.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for policy in ("pairwise_gap", "max_uncertainty", "periodic", "random"):
        subset = [r for r in rows_out if r["policy"] == policy]
        print(policy, {key: float(np.mean([row[key] for row in subset])) for key in ("action_exact", "tokens", "queries")})


if __name__ == "__main__":
    main()

