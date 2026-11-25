from __future__ import annotations

import copy
import unittest

from audit_p4_call_ledger import audit


def fixture():
    event = {
        "call_id": "call-1", "node_id": "source", "input_digest": "1" * 64,
        "input_tokens": 7, "output_digest": "2" * 64, "paid_cost": 8,
        "model_version": "model-v1", "tool_version": "tool-v1",
        "raw_output_digest": "3" * 64, "parse_status": "OK",
    }
    return {
        "event_rows": {"history": [event], "online": [], "post_stop_audit": []},
        "actual_call_counts": {"history": 1, "online": 0, "post_stop_audit": 0},
        "actual_token_counts": {"history": 7, "online": 0, "post_stop_audit": 0},
        "actual_cost_units": {"history": 8, "online": 0, "post_stop_audit": 0},
        "hook_calls": {"pre": 1, "post": 1},
    }


class IndependentLedgerAuditTests(unittest.TestCase):
    def test_accepts_complete_hook_rows(self):
        self.assertEqual(audit(fixture())["status"], "PASS_INDEPENDENT_EVENT_RECONCILIATION_DEV")

    def test_rejects_missing_event(self):
        payload = fixture()
        payload["event_rows"]["history"] = []
        self.assertEqual(audit(payload)["status"], "INVALID")

    def test_rejects_duplicate_id_across_stages(self):
        payload = fixture()
        payload["event_rows"]["online"] = [copy.deepcopy(payload["event_rows"]["history"][0])]
        payload["actual_call_counts"]["online"] = 1
        payload["actual_token_counts"]["online"] = 7
        payload["actual_cost_units"]["online"] = 8
        payload["hook_calls"] = {"pre": 2, "post": 2}
        self.assertIn("DUPLICATE_OR_EMPTY_CALL_ID", " ".join(audit(payload)["errors"]))

    def test_rejects_forged_token_cost(self):
        payload = fixture()
        payload["event_rows"]["history"][0]["paid_cost"] = 99
        payload["actual_cost_units"]["history"] = 99
        self.assertIn("TOKEN_COST_DISAGREEMENT", " ".join(audit(payload)["errors"]))

    def test_rejects_hook_count_mismatch(self):
        payload = fixture()
        payload["hook_calls"]["post"] = 0
        self.assertIn("HOOK_EVENT_TOTAL_MISMATCH", " ".join(audit(payload)["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
