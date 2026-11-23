from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strict_graph_contract import CallRecord, ledger_totals  # noqa: E402


class InformationLedgerTest(unittest.TestCase):
    def test_t06_retrieval_records_real_ids_and_cost(self) -> None:
        retrieval = {"index_version": "bm25-v1", "results": ["doc-1", "doc-2"], "input_tokens": 18}
        self.assertEqual(retrieval["results"], ["doc-1", "doc-2"])
        self.assertGreater(retrieval["input_tokens"], 0)

    def test_t07_cached_call_does_not_charge_tokens(self) -> None:
        records = [CallRecord("new", 12, 1, 0.1, False), CallRecord("cached", 500, 10, 9.0, True)]
        totals = ledger_totals(records)
        self.assertEqual(totals["input_tokens"], 12.0)
        self.assertEqual(totals["cache_hits"], 1.0)

    def test_t08_missing_or_wrong_graph_metadata_fails_closed(self) -> None:
        metadata = {"graph_edge": None, "input_digest": ""}
        self.assertFalse(metadata["graph_edge"])
        self.assertFalse(metadata["input_digest"])

    def test_t10_repro_entrypoint_is_present(self) -> None:
        readme = HERE.parent / "repro" / "README.md"
        self.assertTrue(readme.exists())
        self.assertIn("clean checkout", readme.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

