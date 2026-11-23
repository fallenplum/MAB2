import json
import unittest
from pathlib import Path

from prepare_p2_multihop import build_row

HERE = Path(__file__).resolve().parent


class P2PreparationTest(unittest.TestCase):
    def test_record_contains_raw_and_structured_paths(self) -> None:
        row = json.loads((HERE.parent.parent / "MAB2_recovered" / "qasc_rows2_INPUT.json").read_text(encoding="utf-8"))[0]
        prepared = build_row(row)
        self.assertIn("raw_evidence", prepared)
        self.assertIn("structured_interface", prepared)
        self.assertEqual(prepared["structured_interface"]["nodes"], ["S0", "R0", "R1", "R2", "R3", "T"])
        self.assertEqual(prepared["changed_rows"], [1, 3])
        self.assertEqual(len(prepared["new_relay_interface"]), 4)

    def test_no_gold_key_is_written(self) -> None:
        data = json.loads((HERE / "P2_MULTIHOP_INPUT.json").read_text(encoding="utf-8"))
        manifest = json.loads((HERE / "P2_MULTIHOP_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data), 32)
        self.assertFalse(manifest["gold_in_public_input"])
        self.assertTrue(all("gold" not in row for row in data))


if __name__ == "__main__":
    unittest.main()
