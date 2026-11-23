"""Synthetic conditional-drift test for the pairwise answer certificate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pairwise_gap_certificate import select_rows


def values(source, base, relay, labels):
    weights = source[:, None] * (relay - base[None, :])
    return np.asarray([weights[labels == label].sum() for label in range(4)])


def run_cell(seed: int, rows_changed: int, drift: float, samples: int = 5000) -> dict:
    rng = np.random.default_rng(seed)
    changed = list(range(rows_changed))
    exact = 0
    queries = 0
    for _ in range(samples):
        source = rng.dirichlet(np.ones(4))
        base = rng.dirichlet(np.ones(4))
        old_relay = rng.dirichlet(np.ones(4), size=4)
        labels = rng.integers(0, 4, size=(4, 4))
        new_relay = old_relay.copy()
        for i in changed:
            proposal = rng.dirichlet(np.ones(4))
            new_relay[i] = (1.0 - drift) * old_relay[i] + drift * proposal
        row = {
            "new_source": source,
            "base": base,
            "old_relay": old_relay,
            "new_relay": new_relay,
            "labels": labels,
            "changed_rows": changed,
            "tokens": [1.0] * 27,
        }
        selected = select_rows(row, {i: drift for i in changed})
        relay = old_relay.copy()
        for i in selected:
            relay[i] = new_relay[i]
        exact += int(values(source, base, relay, labels).argmax() ==
                     values(source, base, new_relay, labels).argmax())
        queries += len(selected)
    return {
        "rows_changed": rows_changed,
        "drift": drift,
        "n": samples,
        "pairwise_action_exact": exact / samples,
        "pairwise_queries": queries / samples,
        "full_mask_queries": rows_changed,
    }


def main() -> None:
    results = [
        run_cell(20261123 + rows * 10 + int(drift * 100), rows, drift)
        for rows in (1, 2, 4)
        for drift in (0.1, 0.4, 1.0)
    ]
    payload = {
        "schema": "pairwise_gap_synthetic_v1",
        "results": results,
        "interpretation": (
            "Under a supplied conditional TV radius, every selected action is "
            "certified by pairwise gaps. The experiment is synthetic and does "
            "not establish task-level answer quality."
        ),
    }
    path = Path("MAB2_recovered/experiments/mechanism_20261122/PAIRWISE_GAP_SYNTHETIC.json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

