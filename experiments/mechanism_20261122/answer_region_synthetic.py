"""Synthetic comparison of graph-sign and final-answer region certificates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TAU = 0.01


def values(source, base, relay, labels):
    weights = source[:, None] * (relay - base[None, :])
    return np.asarray([weights[labels == label].sum() for label in range(4)])


def intervals(source, base, relay, labels, changed, delta):
    low = np.zeros(4)
    high = np.zeros(4)
    for i in range(4):
        contribution = source[i] * (relay[i] - base)
        radius = source[i] * delta if i in changed else 0.0
        for y in range(4):
            mass = contribution[labels[i] == y].sum()
            low[y] += mass - radius
            high[y] += mass + radius
    return low, high


def graph_safe(low, high):
    return bool(np.all((low > TAU) | (high < TAU)))


def answer_safe(center, low, high):
    winner = int(np.argmax(center))
    rivals = np.delete(high, winner)
    return rivals.size == 0 or bool(low[winner] > np.max(rivals))


def run(seed, rows_changed, delta, samples=5000):
    rng = np.random.default_rng(seed)
    graph_correct = answer_correct = stale_correct = 0
    graph_queries = answer_queries = 0
    changed = tuple(range(rows_changed))
    for _ in range(samples):
        source = rng.dirichlet(np.ones(4))
        base = rng.dirichlet(np.ones(4))
        old_relay = rng.dirichlet(np.ones(4), size=4)
        labels = rng.integers(0, 4, size=(4, 4))
        new_relay = old_relay.copy()
        for i in changed:
            proposal = rng.dirichlet(np.ones(4))
            new_relay[i] = (1.0 - delta) * old_relay[i] + delta * proposal
        true = values(source, base, new_relay, labels)
        old = values(source, base, old_relay, labels)
        lo, hi = intervals(source, base, old_relay, labels, changed, delta)
        gs = graph_safe(lo, hi)
        ans = answer_safe(old, lo, hi)
        graph_queries += int(not gs) * rows_changed
        answer_queries += int(not ans) * rows_changed
        graph_pred = old > TAU if gs else true > TAU
        answer_pred = int(old.argmax()) if ans else int(true.argmax())
        graph_correct += int(np.array_equal(graph_pred, true > TAU))
        answer_correct += int(answer_pred == int(true.argmax()))
        stale_correct += int(int(old.argmax()) == int(true.argmax()))
    return {
        "rows_changed": rows_changed,
        "delta": delta,
        "n": samples,
        "graph_decision_exact": graph_correct / samples,
        "answer_action_exact": answer_correct / samples,
        "stale_answer_action_exact": stale_correct / samples,
        "graph_queries": graph_queries / samples,
        "answer_queries": answer_queries / samples,
    }


def main():
    results = [run(20261123 + rows * 100 + int(delta * 10), rows, delta)
               for rows in (1, 2, 4) for delta in (0.1, 0.4, 1.0)]
    out = {"schema": "arcc_answer_region_synthetic_v1", "results": results}
    path = Path("MAB2_recovered/experiments/mechanism_20261122/M60_ANSWER_REGION_SYNTHETIC.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

