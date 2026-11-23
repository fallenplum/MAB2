from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blind_scheduler import Audit
from conditional_real_score import decision_set


class DecisionNoneSemanticsTest(unittest.TestCase):
    def test_unfinished_audit_is_not_empty_decision(self) -> None:
        audit = Audit(np.asarray([1.0, 0.0, 0.0, 0.0]),
                      np.full(4, 0.25), np.ones(4), np.ones((4, 4)), "joint")
        self.assertIsNone(audit.decision())
        with self.assertRaisesRegex(RuntimeError, "unfinished audit"):
            decision_set(audit, context="unit-test")

    def test_proved_empty_decision_remains_empty_set(self) -> None:
        audit = Audit(np.full(4, 0.25), np.full(4, 0.25),
                      np.ones(4), np.ones((4, 4)), "joint")
        for i in range(4):
            audit.update((i, None), np.full(4, 0.25))
        for i in range(4):
            for j in range(4):
                audit.update((i, j), 0)
        self.assertEqual(decision_set(audit, context="unit-test-empty"), set())


if __name__ == "__main__":
    unittest.main()
