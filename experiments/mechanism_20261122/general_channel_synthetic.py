"""Synthetic conditional-channel drift experiment for the route-B audit.

This file deliberately uses an abstract binary service action.  It does not
stand in for a real non-MCQA model or task.  The purpose is to test whether the
same certificate logic extends beyond the finite MCQA table and to expose the
conditions under which it degenerates to warm reuse.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TAU = 0.5
ROWS = 4


def _clip(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def decision(q: np.ndarray, source: np.ndarray) -> int:
    return int(float(np.dot(source, q)) > TAU)


def interval(q_old: np.ndarray, q_new: np.ndarray, source: np.ndarray, radii: np.ndarray, known: np.ndarray) -> tuple[float, float]:
    low = np.clip(q_old - radii, 0.0, 1.0)
    high = np.clip(q_old + radii, 0.0, 1.0)
    low[known] = q_new[known]
    high[known] = q_new[known]
    return float(np.dot(source, low)), float(np.dot(source, high))


def route(case: dict[str, object], method: str) -> dict[str, object]:
    source = np.asarray(case["new_source"], dtype=float)
    q_old = np.asarray(case["old_channel"], dtype=float)
    q_new = np.asarray(case["new_channel"], dtype=float)
    radii = np.asarray(case["radii"], dtype=float)
    prices = np.asarray(case["prices"], dtype=float)
    changed = np.asarray(case["changed_rows"], dtype=bool)
    target = decision(q_new, source)

    if method in {"stale", "warm_reweighted"}:
        pred = decision(q_old, source)
        return {"pred": pred, "target": target, "tokens": 0.0, "queries": 0,
                "false_action": int(pred != target), "certificate": "reuse"}
    if method == "oracle_lower_bound":
        return {"pred": target, "target": target, "tokens": 0.0, "queries": 0,
                "false_action": 0, "certificate": "oracle"}

    known = np.zeros(ROWS, dtype=bool)
    if method == "full_restart":
        known[:] = True
        tokens = float(prices.sum())
        return {"pred": target, "target": target, "tokens": tokens, "queries": ROWS,
                "false_action": 0, "certificate": "full"}

    # Source-only edits have no conditional uncertainty.  This is the formal
    # zero-drift degeneration test for CTDR.
    if method == "ctdr_transport":
        candidates = np.flatnonzero(changed & (radii > 0))
    elif method == "fixed_reverse_recheck":
        candidates = np.flatnonzero(changed & (radii > 0))[::-1]
    else:
        raise ValueError(method)

    tokens = 0.0
    queries = 0
    while True:
        lo, hi = interval(q_old, q_new, source, radii, known)
        if lo > TAU:
            pred = 1
            certificate = "lower_positive"
            break
        if hi <= TAU:
            pred = 0
            certificate = "upper_nonpositive"
            break
        pending = [int(i) for i in candidates if not known[i]]
        if not pending:
            # The radius was too loose to certify the action.  A conservative
            # fallback queries every remaining row before returning.
            pending = [int(i) for i in np.flatnonzero(~known)]
        if not pending:
            pred = decision(q_new, source)
            certificate = "exhausted"
            break
        if method == "ctdr_transport":
            row = max(pending, key=lambda i: (float(source[i] * radii[i]) / max(prices[i], 1e-9), -i))
        else:
            row = pending[0]
        known[row] = True
        tokens += float(prices[row])
        queries += 1
    return {"pred": int(pred), "target": target, "tokens": tokens, "queries": queries,
            "false_action": int(pred != target), "certificate": certificate}


def make_case(rng: np.random.Generator, drift_type: str) -> dict[str, object]:
    old_source = rng.dirichlet(np.ones(ROWS) * 3.0)
    new_source = old_source.copy()
    if drift_type == "source_only":
        transfer = min(0.18, float(old_source[0]) * 0.5)
        new_source[0] -= transfer
        new_source[1] += transfer
    q_old = rng.uniform(0.35, 0.65, size=ROWS)
    # Keep a substantial fraction of examples near the action boundary so the
    # certificate must sometimes buy feedback.
    q_old = 0.5 + 0.20 * (q_old - 0.5)
    q_new = q_old.copy()
    changed = np.zeros(ROWS, dtype=bool)
    if drift_type in {"conditional", "interface", "mixed"}:
        n = 1 if drift_type != "mixed" else 2
        indices = rng.choice(ROWS, size=n, replace=False)
        for idx in indices:
            changed[idx] = True
            q_new[idx] = _clip(float(q_old[idx] + rng.choice([-1.0, 1.0]) * rng.uniform(0.06, 0.16)))
    if drift_type == "mixed":
        transfer = min(0.15, float(new_source[0]) * 0.4)
        new_source[0] -= transfer
        new_source[2] += transfer
    # The radius is the exact binary-channel TV drift for changed rows.  This
    # is an oracle radius used only for a synthetic mechanism stress test.
    radii = np.abs(q_new - q_old)
    prices = rng.integers(6, 16, size=ROWS).astype(float)
    return {"old_source": old_source.tolist(), "new_source": new_source.tolist(),
            "old_channel": q_old.tolist(), "new_channel": q_new.tolist(),
            "changed_rows": changed.tolist(), "radii": radii.tolist(),
            "prices": prices.tolist(), "drift_type": drift_type}


def run(seed: int = 20261123, n_per_cell: int = 500) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    methods = ["stale", "warm_reweighted", "ctdr_transport", "fixed_reverse_recheck",
               "full_restart", "oracle_lower_bound"]
    cells: list[dict[str, object]] = []
    for drift_type in ("source_only", "conditional", "interface", "mixed"):
        records = defaultdict(list)
        for _ in range(n_per_cell):
            case = make_case(rng, drift_type)
            for method in methods:
                records[method].append(route(case, method))
        summary = {}
        for method in methods:
            rows = records[method]
            summary[method] = {
                "n": len(rows),
                "action_error": float(np.mean([x["false_action"] for x in rows])),
                "mean_tokens": float(np.mean([x["tokens"] for x in rows])),
                "mean_queries": float(np.mean([x["queries"] for x in rows])),
                "certificate_counts": dict(sorted({k: sum(1 for x in rows if x["certificate"] == k) for k in {x["certificate"] for x in rows}}.items())),
            }
        cells.append({"drift_type": drift_type, "methods": summary})
    return {
        "schema": "general_condition_binary_channel_synthetic_v1",
        "seed": seed,
        "n_per_cell": n_per_cell,
        "tau": TAU,
        "methods": methods,
        "cells": cells,
        "scope": "Synthetic binary conditional-channel stress test. No model, task, or sealed label is used.",
        "radius_note": "Radii equal the generated conditional TV drift and are an oracle synthetic diagnostic.",
    }


def write_report(result: dict[str, object], path: Path) -> None:
    lines = [
        "# General conditional-channel extension",
        "",
        "This route-B attempt uses a synthetic binary service action rather than a real language-model task. It tests whether the certificate logic survives outside the finite MCQA representation. The generated TV radii are oracle diagnostics, so the result is a mechanism sanity check and not an ACL service-quality result.",
        "",
        "| drift type | method | action error | mean tokens | mean queries |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for cell in result["cells"]:
        for method, values in cell["methods"].items():
            lines.append(f"| {cell['drift_type']} | {method} | {values['action_error']:.4f} | {values['mean_tokens']:.2f} | {values['mean_queries']:.2f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Source-only edits have zero conditional radius and CTDR performs zero new queries, formally matching warm reuse. Conditional and interface drift cells require revalidation when the old action interval crosses the threshold. Full restart is an upper-cost exact reference. Random recheck is included as a non-adaptive control.",
        "",
        "The experiment does not establish a new-task or new-checkpoint result. A real BoolQ, GSM8K, or other non-MCQA worker run was not claimed because the current server execution path was unavailable during this cycle. The route-B claim therefore remains unvalidated and the paper's empirical scope stays on graph-conditioned certificate transport.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    out = Path(__file__).with_name("GENERAL_CONDITION_SYNTHETIC.json")
    report = out.with_suffix(".md")
    result = run()
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, report)
    print(json.dumps(result["cells"], indent=2))

