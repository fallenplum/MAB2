"""Synthetic test of source-only versus conditional-drift-safe certificates.

The experiment isolates a finite-row mechanism. A changed row may alter its
conditional relay distribution within a known total-variation budget delta.
The robust policy uses a coordinate-wise interval radius delta and queries all
masked rows only when the interval cannot certify the action.
"""
import json
from pathlib import Path

import numpy as np


TAU = 0.01


def sample_distribution(rng, rows=4, labels=4):
    return rng.dirichlet(np.ones(labels, dtype=np.float64), size=rows)


def values(source, base, relay, labels):
    weights = source[:, None] * (relay - base[None, :])
    return np.asarray([weights[labels == label].sum() for label in range(4)])


def intervals(source, base, old_relay, labels, changed, delta):
    low = np.zeros(4, dtype=np.float64)
    high = np.zeros(4, dtype=np.float64)
    for row_index in range(4):
        radius = delta if row_index in changed else 0.0
        contribution = source[row_index] * (old_relay[row_index] - base)
        for label in range(4):
            mask = labels[row_index] == label
            low[label] += contribution[mask].sum() - source[row_index] * radius * mask.sum()
            high[label] += contribution[mask].sum() + source[row_index] * radius * mask.sum()
    return low, high


def run_cell(seed, rows_changed, drift, samples=5000):
    rng = np.random.default_rng(seed)
    source_old = rng.dirichlet(np.ones(4), size=samples)
    source_new = np.asarray(
        [rng.dirichlet(np.ones(4)) for _ in range(samples)], dtype=np.float64
    )
    base = np.asarray([rng.dirichlet(np.ones(4)) for _ in range(samples)])
    old_relay = np.asarray(
        [sample_distribution(rng, 4, 4) for _ in range(samples)], dtype=np.float64
    )
    labels = rng.integers(0, 4, size=(samples, 4, 4))
    changed = tuple(range(rows_changed))
    new_relay = old_relay.copy()
    for row_index in changed:
        proposal = np.asarray(
            [rng.dirichlet(np.ones(4)) for _ in range(samples)], dtype=np.float64
        )
        new_relay[:, row_index, :] = (1.0 - drift) * old_relay[:, row_index, :] + drift * proposal

    source_only_safe = 0
    robust_safe = 0
    source_only_exact = 0
    robust_exact = 0
    robust_queries = 0
    source_queries = 0
    for index in range(samples):
        true_values = values(source_new[index], base[index], new_relay[index], labels[index])
        true_action = true_values > TAU

        lo0, hi0 = intervals(source_new[index], base[index], old_relay[index], labels[index], changed, 0.0)
        safe0 = bool(np.all((lo0 > TAU) | (hi0 < TAU)))
        source_only_safe += int(safe0)
        source_only_exact += int(np.array_equal(true_action, values(source_new[index], base[index], old_relay[index], labels[index]) > TAU))
        source_queries += int(not safe0) * rows_changed

        lo, hi = intervals(source_new[index], base[index], old_relay[index], labels[index], changed, drift)
        safe = bool(np.all((lo > TAU) | (hi < TAU)))
        robust_safe += int(safe)
        if safe:
            robust_action = values(source_new[index], base[index], old_relay[index], labels[index]) > TAU
            robust_queries += 0
        else:
            robust_action = true_action
            robust_queries += rows_changed
        robust_exact += int(np.array_equal(robust_action, true_action))
    return {
        "n": samples,
        "rows_changed": rows_changed,
        "drift": drift,
        "source_only_safe_rate": source_only_safe / samples,
        "source_only_exact": source_only_exact / samples,
        "source_only_queries": source_queries / samples,
        "robust_safe_rate": robust_safe / samples,
        "robust_exact": robust_exact / samples,
        "robust_queries": robust_queries / samples,
    }


if __name__ == "__main__":
    results = [
        run_cell(seed=20261123 + rows * 10 + int(drift * 100), rows_changed=rows, drift=drift)
        for rows in (1, 2, 4)
        for drift in (0.0, 0.1, 0.4, 1.0)
    ]
    output = {
        "schema": "m60_conditional_drift_safe_policy_v1",
        "results": results,
        "interpretation": "The robust interval is a finite-row soundness mechanism under a known conditional TV radius. It is synthetic and does not establish task-level answer quality.",
    }
    path = Path("MAB2_recovered/experiments/mechanism_20261122/M60_CONDITIONAL_DRIFT_SAFE_POLICY.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

