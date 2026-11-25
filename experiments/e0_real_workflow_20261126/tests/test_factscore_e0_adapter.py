import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.factscore_e0_adapter import assert_no_human_labels, fixed_contract, load_online_groups, online_payload
from src.factscore_runtime import parse_verdict


class FactScoreAdapterTests(unittest.TestCase):
    def test_online_view_excludes_human_labels(self):
        row = {
            "topic": "Example",
            "output": "A short answer.",
            "annotations": [{
                "model-atomic-facts": [{"text": "A short answer."}],
                "human-atomic-facts": [{"text": "A short answer.", "label": "S"}],
            }],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            group_id = "e0g_" + hashlib.sha256(b"salt|Example").hexdigest()[:16]
            groups = load_online_groups(path, {group_id}, "salt")
        payload = online_payload(groups[0])
        assert_no_human_labels(payload)
        self.assertNotIn("label", json.dumps(payload))

    def test_contract_has_fixed_unit_weights(self):
        row = {"topic": "Example", "output": "Answer.", "annotations": [{"model-atomic-facts": [{"text": "Answer."}]}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            group_id = "e0g_" + hashlib.sha256(b"salt|Example").hexdigest()[:16]
            group = load_online_groups(path, {group_id}, "salt")[0]
        contract = fixed_contract(group)
        self.assertEqual(contract["weights"], [1])
        self.assertEqual(contract["required_support"], 1)

    def test_human_fact_text_can_be_frozen_without_labels(self):
        row = {
            "topic": "Example",
            "output": "Answer.",
            "annotations": [{"human-atomic-facts": [{"text": "Answer.", "label": "S"}]}],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            group_id = "e0g_" + hashlib.sha256(b"salt|Example").hexdigest()[:16]
            group = load_online_groups(path, {group_id}, "salt", fact_source="human")[0]
        self.assertEqual(group.atomic_facts, ("Answer.",))
        assert_no_human_labels(online_payload(group))

    def test_verdict_parser_handles_split_unsupported(self):
        self.assertEqual(parse_verdict("UN SUPPORTED"), 0)
        self.assertEqual(parse_verdict("SUPPORT\\nTHE FACT"), 1)
        self.assertIsNone(parse_verdict("DECISION: UNS"))

    def test_missing_frozen_facts_can_be_retained_as_invalid_group(self):
        row = {"topic": "Example", "output": "Answer.", "annotations": None}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            group_id = "e0g_" + hashlib.sha256(b"salt|Example").hexdigest()[:16]
            group = load_online_groups(path, {group_id}, "salt", fact_source="human", allow_missing=True)[0]
        self.assertEqual(group.atomic_facts, ())
        self.assertEqual(group.decomposition_source, "INVALID_NO_FROZEN_FACTS")


if __name__ == "__main__":
    unittest.main()

