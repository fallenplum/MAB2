"""Fixed-threshold certificate for the E0 decision-service contract.

The implementation intentionally keeps three states explicit.  A certificate
is different from a fully resolved result, and an empty task is invalid.
No target, gold label, or future output is accessed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Certificate:
    status: str
    decision: str | None
    n: int
    required_support: int
    supported: int
    failed: int
    unknown: int
    reason: str | None = None


def fixed_threshold_certificate(labels: Sequence[int | None], threshold: float = 0.80) -> Certificate:
    """Return a conservative PASS/FAIL/UNRESOLVED certificate.

    ``1`` denotes support, ``0`` denotes failure, and ``None`` denotes an
    unresolved fact.  The decision is certified only when every legal
    completion of unresolved facts has the same threshold outcome.
    """

    n = len(labels)
    if n == 0:
        return Certificate("INVALID", None, 0, 0, 0, 0, 0, "empty_task")
    if not 0 < threshold <= 1:
        return Certificate("INVALID", None, n, 0, 0, 0, 0, "invalid_threshold")
    if any(label not in (0, 1, None) for label in labels):
        return Certificate("INVALID", None, n, 0, 0, 0, 0, "illegal_label")

    required = ceil(threshold * n)
    supported = sum(label == 1 for label in labels)
    failed = sum(label == 0 for label in labels)
    unknown = sum(label is None for label in labels)

    if supported >= required:
        return Certificate("CERTIFIED", "PASS", n, required, supported, failed, unknown)
    if supported + unknown < required:
        return Certificate("CERTIFIED", "FAIL", n, required, supported, failed, unknown)
    if unknown == 0:
        # This branch is reachable only for a malformed threshold arithmetic
        # change. Keep the status explicit if the contract is modified later.
        return Certificate("FULL_RESOLVED", "FAIL", n, required, supported, failed, 0)
    return Certificate("UNRESOLVED", None, n, required, supported, failed, unknown)


def full_resolution_status(labels: Iterable[int | None]) -> str:
    """Expose whether all facts were resolved without making a decision."""

    values = list(labels)
    if any(value is None for value in values):
        return "PARTIAL"
    return "FULL_RESOLVED"

