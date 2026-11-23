"""Regression checks for metadata resolution and certificate edge cases."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairwise_gap_certificate import resolve_changed_rows


class MetadataResolutionTest(unittest.TestCase):
    def test_fingerprint_mask_is_preferred(self):
        rows, source = resolve_changed_rows({
            "old_relay_fingerprints": ["a", "b", "c", "d"],
            "new_relay_fingerprints": ["a", "x", "c", "d"],
            "changed_rows": [3],
        })
        self.assertEqual(rows, [1])
        self.assertEqual(source, "fingerprint")

    def test_legacy_mask_is_supported(self):
        rows, source = resolve_changed_rows({"changed_rows": [2, 2]})
        self.assertEqual(rows, [2])
        self.assertEqual(source, "graph_mask_legacy")

    def test_missing_metadata_falls_back_to_all_rows(self):
        rows, source = resolve_changed_rows({})
        self.assertEqual(rows, [0, 1, 2, 3])
        self.assertEqual(source, "fail_closed_all_rows")


if __name__ == "__main__":
    unittest.main()

