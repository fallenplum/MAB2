"""Tests for the controlled DAG topology VOI experiment."""
from __future__ import annotations

import unittest

import p1_topology_voi as p1


class TopologyVOITest(unittest.TestCase):
    def test_affected_set_is_transitive_descendant_closure(self) -> None:
        edges = {0: [1], 1: [2], 2: [3], 3: []}
        self.assertEqual(p1.descendants(edges, [0]), {0, 1, 2, 3})

    def test_topology_voi_beats_controls_on_fixed_seed(self) -> None:
        result = p1.run(seed=20261124, n_cases=300)
        summary = result["summary"]
        self.assertEqual(summary["topology_voi"]["action_error"], 0.0)
        self.assertLess(summary["topology_voi"]["mean_tokens"], summary["pairwise"]["mean_tokens"])
        self.assertLess(summary["topology_voi"]["mean_tokens"], summary["generic_adaptive"]["mean_tokens"])

    def test_offline_oracle_is_explicit(self) -> None:
        result = p1.run(seed=20261124, n_cases=10)
        self.assertEqual(result["summary"]["offline_full_info"]["mean_tokens"], 0.0)
        self.assertEqual(result["graph_protocol"]["offline_oracle"], "offline_full_info reports the generated upper bound and is not an online policy")


if __name__ == "__main__":
    unittest.main()
