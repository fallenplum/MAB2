from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent / "g0_online_runtime_001"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from p4_online_runtime import CallEvent, CallResult, OnlineBroker, RuntimeNode, input_key


class Backend:
    def __init__(self, outputs):
        self.outputs = dict(outputs)
        self.calls = 0

    def __call__(self, node_id, ids, model_version, tool_version, paid_cost, hook_sink):
        self.calls += 1
        output = self.outputs[node_id]
        digest = input_key(ids, model_version, tool_version)
        event = CallEvent(
            call_id=f"adversarial-{self.calls}", node_id=node_id,
            input_digest=digest, input_tokens=len(ids),
            output_digest=hashlib.sha256(output.encode()).hexdigest(),
            paid_cost=paid_cost, model_version=model_version,
            tool_version=tool_version,
            raw_output_digest=hashlib.sha256(output.encode()).hexdigest(),
        )
        hook_sink(event)
        return CallResult(output, event)


def node(node_id, parents, render, old_ids, old_output, *, terminal=False):
    return RuntimeNode(
        node_id=node_id, parents=tuple(parents), render=render,
        old_input_digest=input_key(old_ids, "model-v1"),
        old_output=old_output, model_version="model-v1", terminal=terminal,
    )


class AdversarialBoundaryTests(unittest.TestCase):
    def test_t08_out_of_order_parent_manifest_fails_closed(self):
        nodes = (
            node("join", ("missing",), lambda p: [2], [2], "A", terminal=True),
        )
        with self.assertRaises(ValueError):
            OnlineBroker(("A", "B"), nodes, {"join": 1}, 20, Backend({"join": "A"}))

    def test_t08_changed_second_parent_blocks_terminal_until_resolved(self):
        nodes = (
            node("p0", (), lambda p: [1], [1], "left"),
            node("p1", (), lambda p: [2], [3], "right-old"),
            node("join", ("p0", "p1"), lambda p: [10 if p["p1"] == "right-old" else 11], [10], "A", terminal=True),
        )
        broker = OnlineBroker(("A", "B"), nodes, {"join": 1}, 20, Backend({"p1": "right-new", "join": "A"}))
        self.assertEqual(broker.public_state()["legal_actions"], ["p1"])
        broker.query("p1")
        self.assertIn("join", broker.public_state()["legal_actions"])

    def test_t14_unresolved_stop_never_uses_reference_like_action(self):
        broker = OnlineBroker(
            ("A", "B"),
            (node("answer", (), lambda p: [7], [8], "A", terminal=True),),
            {"answer": 1}, 0, Backend({"answer": "A"}),
        )
        response = broker.accept_policy_action({"type": "stop", "action": "A"})
        self.assertEqual(response["status"], "INVALID")
        self.assertEqual(response["reason"], "UNPROVEN_STOP")
        self.assertEqual(broker.events, [])

    def test_t14_invalid_output_is_charged_and_never_certified(self):
        broker = OnlineBroker(
            ("A", "B"),
            (node("answer", (), lambda p: [7], [8], "A", terminal=True),),
            {"answer": 1}, 20, Backend({"answer": "target"}),
        )
        broker.query("answer")
        self.assertEqual(broker.status, "INVALID")
        self.assertEqual(len(broker.events), 1)
        self.assertGreater(broker.spent, 0)
        self.assertEqual(broker.public_state()["certificate"]["status"], "INVALID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
