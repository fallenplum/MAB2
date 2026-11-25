from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from p4_p2_service_adapter import combine_repeat_runs, compare_repeat_runs, run_row


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 and "A" <= text <= "H":
            return [ord(text) - ord("A")]
        raw = hashlib.sha256(text.encode("utf-8")).digest()
        return ([120] if add_special_tokens else []) + [16 + value for value in raw[:24]]


class FakeModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        seed = int(input_ids[0, -1].item())
        logits = torch.full((1, input_ids.shape[1], 16), -1.0, device=input_ids.device)
        logits[..., 0] = 2.0 + (seed % 7) * 0.05
        logits[..., 1] = 1.5
        logits[..., 2] = 1.0
        logits[..., 3] = 0.5
        return SimpleNamespace(logits=logits)


class P2AdapterTests(unittest.TestCase):
    def test_locked_production_p2_renderer_runs_through_online_broker(self):
        rows = json.loads((WORKSPACE / "experiments/p2_multihop_20261124/P2_MULTIHOP_INPUT.json").read_text(encoding="utf-8"))
        row = rows[0]
        self.assertTrue(any(row["second_facts_new"][i] != row["second_fact"] for i in row["changed_rows"]))
        result = run_row(
            row, FakeModel().eval(), FakeTokenizer(), "cpu", "fake-qwen-checkpoint", "locked-p2-renderer-test",
            HERE / "p4_policy_worker.py", cap_seconds=30,
        )
        self.assertEqual(result["status"], "PASS_T01_T02_T03_T04_T05_T07_T11_BOUNDED_DEV")
        self.assertEqual(result["actual_call_counts"]["history"], 21)
        self.assertEqual(result["ledger_checks"]["online"][0], "VALID_DEVELOPMENT_LEDGER")
        self.assertTrue(result["comparisons"]["unchanged_source_cached"])
        self.assertTrue(result["comparisons"]["every_positive_terminal_cache_hit_matches_complete_actual_prompt_ids"])
        self.assertTrue(result["comparisons"]["actual_second_parent_changes_invalidate_terminal_cache"])
        self.assertTrue(result["second_parent_terminal_invalidation"][1])
        self.assertTrue(result["second_parent_terminal_invalidation"][3])
        self.assertTrue(result["comparisons"]["online_action_matches_post_stop_full_new_graph_action"])
        self.assertEqual(result["hook_calls"]["pre"], result["hook_calls"]["post"])
        self.assertFalse(result["gold_or_target_loaded_before_online_stop"])
        repeated = run_row(
            row, FakeModel().eval(), FakeTokenizer(), "cpu", "fake-qwen-checkpoint", "locked-p2-renderer-test",
            HERE / "p4_policy_worker.py", cap_seconds=30,
        )
        self.assertEqual(compare_repeat_runs([result, repeated])["status"], "PASS")
        combined = combine_repeat_runs([result, repeated])
        self.assertEqual(combined["repeat_check"]["status"], "PASS")
        self.assertEqual(combined["actual_call_counts"]["history"], 42)
        self.assertEqual(combined["ledger_checks"]["history"][0], "VALID_DEVELOPMENT_LEDGER")
        self.assertEqual(len({event["call_id"] for event in combined["event_rows"]["history"]}), 42)
        changed = json.loads(json.dumps(repeated))
        changed["event_rows"]["history"][0]["raw_output_digest"] = "f" * 64
        self.assertEqual(combine_repeat_runs([result, changed])["status"], "FAIL_REPEAT_SEMANTICS_BOUNDED_DEV")


if __name__ == "__main__":
    unittest.main(verbosity=2)
