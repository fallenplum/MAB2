"""Exact random and additive-cost calculations for flat R1 replay."""

from __future__ import annotations


def uniform_expected_count(k: int, n: int, m: int) -> float:
    if not (0 <= k <= m <= n) or m == 0:
        raise ValueError("invalid random-order parameters")
    return k * (n + 1) / (m + 1)


def uniform_expected_cost(k: int, witness_costs: list[float], non_witness_costs: list[float]) -> float:
    m = len(witness_costs)
    if not 0 < k <= m:
        raise ValueError("invalid witness count")
    return (k / m) * sum(witness_costs) + (k / (m + 1)) * sum(non_witness_costs)


def clairvoyant_cost(k: int, witness_costs: list[float]) -> float:
    if not 0 <= k <= len(witness_costs):
        raise ValueError("invalid k")
    return sum(sorted(witness_costs)[:k])

