"""Pairwise answer-action certificate for graph-conditioned credit revalidation.

The route certifies only the final answer action.  For the current candidate
answer ``a`` and every competitor ``b``, it maintains a lower bound on
``V(a)-V(b)``.  An unqueried changed row with conditional TV radius ``delta_i``
can reduce this pairwise gap by at most ``2 * source_i * delta_i``.  Rows are
queried by this worst-case gap reduction per token until every pairwise gap is
strictly positive.  The certificate is conditional on the supplied radii.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from answer_safe_arcc import values
from conditional_schema import validate_conditional_full


def resolve_changed_rows(item: dict) -> tuple[list[int], str]:
    """Resolve the changed-row mask and fail closed on missing metadata."""
    old = item.get("old_relay_fingerprints")
    new = item.get("new_relay_fingerprints")
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new) == 4:
        return [i for i, (before, after) in enumerate(zip(old, new)) if before != after], "fingerprint"
    explicit = item.get("changed_rows")
    if isinstance(explicit, list) and all(isinstance(i, int) and 0 <= i < 4 for i in explicit):
        return sorted(set(int(i) for i in explicit)), "graph_mask_legacy"
    return list(range(4)), "fail_closed_all_rows"


def parse_worker(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        parsed = validate_conditional_full(item["full"])
        old_source, new_source, base = parsed["source"]
        changed_rows, metadata_source = resolve_changed_rows(item)
        rows.append({
            **parsed,
            "old_source": np.asarray(old_source, dtype=float),
            "new_source": np.asarray(new_source, dtype=float),
            "base": np.asarray(base, dtype=float),
            "old_relay": np.asarray(parsed["old_relay"], dtype=float),
            "new_relay": np.asarray(parsed["new_relay"], dtype=float),
            "key": item["key"],
            "tokens": item["tokens"],
            "changed_rows": changed_rows,
            "metadata_source": metadata_source,
            "old_relay_fingerprints": item.get("old_relay_fingerprints"),
            "new_relay_fingerprints": item.get("new_relay_fingerprints"),
        })
    return rows


def radii(rows: list[dict], quantile: float) -> dict[int, float]:
    samples: dict[int, list[float]] = {}
    for row in rows:
        for i in row["changed_rows"]:
            tv = 0.5 * np.abs(row["new_relay"][i] - row["old_relay"][i]).sum()
            samples.setdefault(i, []).append(float(tv))
    return {i: float(np.quantile(v, quantile)) for i, v in samples.items()}


def select_rows(row: dict, radius: dict[int, float]) -> list[int]:
    unknown = set(row["changed_rows"])
    selected: list[int] = []
    while True:
        relay = row["old_relay"].copy()
        for i in selected:
            relay[i] = row["new_relay"][i]
        score = values(row["new_source"], row["base"], relay, row["labels"])
        answer = int(np.argmax(score))
        remaining = unknown.difference(selected)
        gaps = []
        for competitor in range(len(score)):
            if competitor == answer:
                continue
            uncertainty = sum(
                2.0 * row["new_source"][i] * radius.get(i, 1.0)
                for i in remaining
            )
            gaps.append(float(score[answer] - score[competitor] - uncertainty))
        if not remaining or not gaps or min(gaps) > 0.0:
            return selected
        selected.append(max(
            remaining,
            key=lambda i: (
                2.0 * row["new_source"][i] * radius.get(i, 1.0)
                / max(float(row["tokens"][7 + i]), 1.0),
                -i,
            ),
        ))


def evaluate(rows: list[dict], radius: dict[int, float]) -> dict:
    records = []
    for row in rows:
        selected = select_rows(row, radius)
        relay = row["old_relay"].copy()
        for i in selected:
            relay[i] = row["new_relay"][i]
        predicted = values(row["new_source"], row["base"], relay, row["labels"])
        truth = values(row["new_source"], row["base"], row["new_relay"], row["labels"])
        records.append({
            "answer_action_exact": float(int(predicted.argmax()) == int(truth.argmax())),
            "tokens": float(sum(row["tokens"][7 + i] for i in selected)),
            "queries": float(len(selected)),
        })
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = parse_worker(args.worker)
    calibration = [r for r in rows if int(r["key"][:8], 16) % 5 < 3]
    selection = [r for r in rows if int(r["key"][:8], 16) % 5 == 3]
    test = [r for r in rows if int(r["key"][:8], 16) % 5 == 4]
    candidates = []
    for q in (0.5, 0.75, 0.9, 0.95, 1.0):
        radius = radii(calibration, q)
        candidates.append({"q": q, "radius": radius,
                           "selection": evaluate(selection, radius)})
    chosen = sorted(candidates, key=lambda x: (
        -x["selection"]["answer_action_exact"],
        x["selection"]["tokens"], x["q"],
    ))[0]
    result = {
        "schema": "pairwise_gap_certificate_v1",
        "worker": args.worker.name,
        "calibration_n": len(calibration),
        "selection_n": len(selection),
        "test_n": len(test),
        "candidates": candidates,
        "selected": chosen,
        "test": evaluate(test, chosen["radius"]),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

