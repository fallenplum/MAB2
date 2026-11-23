from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_inputs import build  # noqa: E402
from run_replay import run_pipeline  # noqa: E402


class CleanReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build()

    def test_pipeline_reports_all_public_controllers(self) -> None:
        result = run_pipeline(ROOT / "fixture_manifest.json")
        names = {row["method"] for row in result["table"]}
        self.assertEqual(len(result["table"]), 9)
        self.assertIn("CTDR transport only", names)
        self.assertIn("CTDR transport + selective ranking", names)
        self.assertIn("oracle lower bound", names)
        self.assertTrue(all(row["n"] == 4 for row in result["table"]))

    def test_fixture_schema_is_replayable(self) -> None:
        rows = [json.loads(line) for line in (ROOT / "fixture" / "WORKER0.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(set(row["full"]) == {str(i) for i in range(27)} for row in rows))
        self.assertTrue(all(len(row["tokens"]) == 27 for row in rows))


if __name__ == "__main__":
    unittest.main()

