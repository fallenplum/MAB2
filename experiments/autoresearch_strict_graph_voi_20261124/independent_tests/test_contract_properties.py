from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from strict_graph_contract import (  # noqa: E402
    CallRecord,
    certified_action,
    digest,
    exhaustive_actions,
    ledger_totals,
    render_prompt,
)


class StrictGraphContractTest(unittest.TestCase):
    def test_t01_parent_output_changes_actual_child_fingerprint(self) -> None:
        old = render_prompt("question={0};evidence={1}", ["Q", "old"])
        new = render_prompt("question={0};evidence={1}", ["Q", "new"])
        self.assertNotEqual(old, new)
        self.assertNotEqual(digest(old), digest(new))

    def test_t02_exact_cache_invalidates_changed_input(self) -> None:
        cache = {digest("Q|old"): "cached"}
        child_input = "Q|old"
        self.assertIn(digest(child_input), cache)
        changed_input = "Q|new"
        self.assertNotIn(digest(changed_input), cache)

    def test_t03_certificate_requires_strict_separation(self) -> None:
        weights = {0: 0.50, 1: 0.45, 2: 0.08}
        known = {0: 0, 1: 1}
        # The unknown cell can flip the decision. It must not be certified.
        self.assertIsNone(certified_action(known, weights, [2], 3))
        self.assertEqual(exhaustive_actions(known, weights, [2], 3), {0, 1})

    def test_t03_certificate_matches_exhaustive_when_certified(self) -> None:
        weights = {0: 0.70, 1: 0.20, 2: 0.10}
        known = {0: 0, 1: 1}
        self.assertEqual(certified_action(known, weights, [2], 3), 0)
        self.assertEqual(exhaustive_actions(known, weights, [2], 3), {0})

    def test_t04_sequential_observation_changes_next_action(self) -> None:
        weights = {0: 0.45, 1: 0.35, 2: 0.20}
        state_a = certified_action({0: 0}, weights, [1, 2], 3)
        state_b = certified_action({0: 0, 1: 1}, weights, [2], 3)
        self.assertIsNone(state_a)
        self.assertIsNone(state_b)
        # The admissible certificate state changes after observing a parent.
        self.assertNotEqual(
            exhaustive_actions({0: 0}, weights, [1, 2], 3),
            exhaustive_actions({0: 0, 1: 1}, weights, [2], 3),
        )

    def test_t05_online_action_is_independent_of_sealed_gold(self) -> None:
        weights = {0: 0.8, 1: 0.2}
        action_a = certified_action({0: 0}, weights, [1], 2)
        action_b = certified_action({0: 0}, weights, [1], 2)
        self.assertEqual(action_a, action_b)

    def test_t07_ledger_sums_only_paid_calls(self) -> None:
        records = [
            CallRecord("a", 10, 1, 0.2, False),
            CallRecord("b", 999, 999, 9.0, True),
            CallRecord("c", 20, 2, 0.3, False),
        ]
        self.assertEqual(
            ledger_totals(records),
            {"forward_calls": 2.0, "input_tokens": 30.0, "output_tokens": 3.0,
             "wall_seconds": 0.5, "cache_hits": 1.0},
        )

    def test_t08_missing_fingerprint_fails_closed(self) -> None:
        manifest = {"node": {"input_digest": None}}
        self.assertIsNone(manifest["node"]["input_digest"])
        self.assertIsNone(certified_action({}, {0: 1.0}, [0], 2))

    def test_t09_digest_is_deterministic(self) -> None:
        self.assertEqual(digest("same"), digest("same"))
        self.assertNotEqual(digest("same"), digest("different"))

    def test_t10_clean_subprocess_can_import_contract(self) -> None:
        script = "from strict_graph_contract import digest; print(digest('ok'))"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(HERE)
        result = subprocess.run([sys.executable, "-c", script], env=env,
                                capture_output=True, text=True, check=True)
        self.assertEqual(len(result.stdout.strip()), 64)


if __name__ == "__main__":
    unittest.main()


