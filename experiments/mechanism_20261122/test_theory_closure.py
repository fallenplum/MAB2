"""Small executable checks for the transport-theory closure."""
from __future__ import annotations

import unittest


def pair_gap(q_a: float, q_b: float, source: float = 1.0) -> float:
    return source * (q_a - q_b)


def tv_binary(old_a: float, new_a: float) -> float:
    return abs(old_a - new_a)


class TheoryClosureTests(unittest.TestCase):
    def test_tv_row_term_is_tight(self) -> None:
        delta = 0.2
        old_gap = pair_gap(delta, 1.0 - delta)
        new_gap = pair_gap(0.0, 1.0)
        self.assertAlmostEqual(tv_binary(delta, 0.0), delta)
        self.assertAlmostEqual(old_gap - new_gap, 2.0 * delta)

    def test_no_drift_degenerates_to_stale(self) -> None:
        cached_gap = 0.37
        tv_radius = 0.0
        source_radius = 0.0
        transported_gap = cached_gap - tv_radius - source_radius
        unresolved_rows: list[int] = []
        queries = len(unresolved_rows) if (tv_radius or source_radius) else 0
        self.assertEqual(transported_gap, cached_gap)
        self.assertEqual(queries, 0)

    def test_cost_aware_rank_differs_only_with_prices(self) -> None:
        margins = {0: 0.8, 1: 0.6}
        prices = {0: 4.0, 1: 1.0}
        uncertainty_order = max(margins, key=margins.get)
        cost_order = max(margins, key=lambda i: margins[i] / prices[i])
        self.assertEqual(uncertainty_order, 0)
        self.assertEqual(cost_order, 1)


if __name__ == "__main__":
    unittest.main()

