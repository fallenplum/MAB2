"""Fixed-denominator certificate used by the E0-R1 offline replay."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence


@dataclass(frozen=True)
class Certificate:
    status: str
    action: str | None
    n: int
    m: int
    supported: int
    failed: int
    unknown: int


def required_support(n: int, threshold_pct: int) -> int:
    if n <= 0 or threshold_pct <= 0 or threshold_pct > 100:
        raise ValueError("invalid task size or threshold")
    return (threshold_pct * n + 99) // 100


def certificate(labels: Sequence[int | None], threshold_pct: int = 80) -> Certificate:
    n = len(labels)
    if n == 0:
        return Certificate("INVALID", None, 0, 0, 0, 0, 0)
    if any(label not in (0, 1, None) for label in labels):
        return Certificate("INVALID", None, n, 0, 0, 0, 0)
    m = required_support(n, threshold_pct)
    supported = sum(label == 1 for label in labels)
    failed = sum(label == 0 for label in labels)
    unknown = n - supported - failed
    if unknown == 0:
        if supported >= m:
            return Certificate("FULL_RESOLVED", "PASS", n, m, supported, failed, unknown)
        return Certificate("FULL_RESOLVED", "FAIL", n, m, supported, failed, unknown)
    if supported >= m:
        return Certificate("CERTIFIED", "PASS", n, m, supported, failed, unknown)
    if supported + unknown < m:
        return Certificate("CERTIFIED", "FAIL", n, m, supported, failed, unknown)
    return Certificate("UNRESOLVED", None, n, m, supported, failed, unknown)


def assert_invariant(cert: Certificate) -> None:
    if cert.n != cert.supported + cert.failed + cert.unknown:
        raise AssertionError("n != s + f + u")

