from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strict_graph_contract import certified_action, exhaustive_actions  # noqa: E402


class ExhaustiveCertificateTest(unittest.TestCase):
    def test_random_small_instances_never_certify_a_flippable_action(self) -> None:
        rng = random.Random(20261124)
        for _ in range(500):
            n_labels = 3
            cells = list(range(4))
            weights = {cell: rng.random() for cell in cells}
            known_cells = cells[:2]
            unknown_cells = cells[2:]
            known = {cell: rng.randrange(n_labels) for cell in known_cells}
            action = certified_action(known, weights, unknown_cells, n_labels)
            attainable = exhaustive_actions(known, weights, unknown_cells, n_labels)
            if action is not None:
                self.assertEqual(attainable, {action})

    def test_tie_requires_more_information(self) -> None:
        weights = {0: 0.5, 1: 0.5}
        known = {0: 0, 1: 1}
        self.assertIsNone(certified_action(known, weights, [], 3))


if __name__ == "__main__":
    unittest.main()

