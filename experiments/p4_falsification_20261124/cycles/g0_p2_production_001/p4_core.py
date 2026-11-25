"""Public certificate core for P4 integration auditing.

This module contains no fixture, target, gold or future-output access. The
broker supplies only resolved labels, legal actions, weights and paid costs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Certificate:
    status: str
    action: str | None
    known_gap: int
    unknown_weight: int


def strict_certificate(actions: Sequence[str], weights: Mapping[str, int], known: Mapping[str, str]) -> Certificate:
    if len(actions) < 2 or len(set(actions)) != len(actions) or not weights:
        return Certificate("INVALID", None, 0, sum(int(weight) for weight in weights.values()))
    if any(isinstance(weight, bool) or not isinstance(weight, int) or weight < 0 for weight in weights.values()):
        return Certificate("INVALID", None, 0, sum(int(weight) for weight in weights.values() if isinstance(weight, int) and not isinstance(weight, bool) and weight > 0))
    if not any(weight > 0 for weight in weights.values()):
        return Certificate("INVALID", None, 0, sum(int(weight) for weight in weights.values()))
    invalid_known = [
        node for node, label in known.items()
        if node in weights and label not in actions
    ]
    if invalid_known:
        return Certificate("INVALID", None, 0, sum(int(weights[node]) for node in invalid_known))
    scores = {a: 0 for a in actions}
    for node, label in known.items():
        if node in weights and label in scores:
            scores[label] += int(weights[node])
    unknown = sum(int(weight) for node, weight in weights.items() if weight > 0 and node not in known)
    winner = max(actions, key=lambda action: scores[action])
    other = max(scores[action] for action in actions if action != winner)
    gap = scores[winner] - other
    # A complete transcript is resolved, not an early certificate. Keep the
    # status distinction explicit even when the resolved action is unique.
    if unknown == 0:
        return Certificate("FULL_RESOLVED", winner, gap, 0)
    if gap > unknown:
        return Certificate("CERTIFIED", winner, gap, unknown)
    return Certificate("UNRESOLVED", None, gap, unknown)


def legal_node(node_id: str, parents: Sequence[str], resolved: set[str], relevant: set[str]) -> bool:
    return node_id in relevant and node_id not in resolved and all(parent in resolved for parent in parents)
