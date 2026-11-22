"""Sequential conditional-drift-safe revalidation with cost-aware row choice."""
import json
from pathlib import Path

import numpy as np


TAU = 0.01


def values(source, base, relay, labels):
    weights = source[:, None] * (relay - base[None, :])
    return np.asarray([weights[labels == label].sum() for label in range(4)])


def bounds(source, base, old_relay, new_relay, labels, unknown, delta):
    low = np.zeros(4)
    high = np.zeros(4)
    for i in range(4):
        relay = old_relay[i] if i in unknown else new_relay[i]
        contribution = source[i] * (relay - base)
        radius = source[i] * delta if i in unknown else 0.0
        for label in range(4):
            mask = labels[i] == label
            low[label] += contribution[mask].sum() - radius
            high[label] += contribution[mask].sum() + radius
    return low, high


def choose_rows(source, base, old_relay, new_relay, labels, changed, delta, costs):
    unknown = set(changed)
    queries = []
    while unknown:
        low, high = bounds(source, base, old_relay, new_relay, labels, unknown, delta)
        unresolved = ~((low > TAU) | (high < TAU))
        if not np.any(unresolved):
            break
        candidates = []
        for i in unknown:
            width = 2.0 * source[i] * delta
            score = float(width * unresolved.sum() / max(costs[i], 1.0))
            candidates.append((score, i))
        _, chosen = max(candidates)
        unknown.remove(chosen)
        queries.append(chosen)
    return queries


def run_cell(seed, rows_changed, drift, samples=5000):
    rng = np.random.default_rng(seed)
    source = rng.dirichlet(np.ones(4), size=samples)
    base = rng.dirichlet(np.ones(4), size=samples)
    old_relay = np.asarray([rng.dirichlet(np.ones(4), size=4) for _ in range(samples)])
    labels = rng.integers(0, 4, size=(samples, 4, 4))
    changed = tuple(range(rows_changed))
    new_relay = old_relay.copy()
    for i in changed:
        proposal = np.asarray([rng.dirichlet(np.ones(4)) for _ in range(samples)])
        new_relay[:, i, :] = (1.0 - drift) * old_relay[:, i, :] + drift * proposal
    exact = []
    query_count = []
    for k in range(samples):
        chosen = choose_rows(source[k], base[k], old_relay[k], new_relay[k], labels[k], changed, drift, [1.0] * 4)
        relay = old_relay[k].copy()
        for i in chosen:
            relay[i] = new_relay[k, i]
        exact.append(np.array_equal(values(source[k], base[k], relay, labels[k]) > TAU, values(source[k], base[k], new_relay[k], labels[k]) > TAU))
        query_count.append(len(chosen))
    return {"n": samples, "rows_changed": rows_changed, "drift": drift, "exact": float(np.mean(exact)), "queries": float(np.mean(query_count))}


if __name__ == "__main__":
    output = {"schema": "m60_sequential_conditional_revalidation_v1", "results": [run_cell(20261201 + m * 10 + int(d * 100), m, d) for m in (1, 2, 4) for d in (0.1, 0.4, 1.0)], "interpretation": "Synthetic sequential safe revalidation under a known conditional TV radius. It is a mechanism experiment and does not establish task-level answer quality."}
    path = Path("MAB2_recovered/experiments/mechanism_20261122/M60_SEQUENTIAL_CONDITIONAL_REVALIDATION.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

