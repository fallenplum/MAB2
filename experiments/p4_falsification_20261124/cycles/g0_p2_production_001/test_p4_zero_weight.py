from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from p4_core import strict_certificate
from p4_online_runtime import CallEvent, CallResult, OnlineBroker, RuntimeNode, input_key


class Backend:
    def __call__(self, node_id, ids, model_version, tool_version, paid_cost, hook_sink):
        output = "A" if node_id == "positive" else "B"
        event = CallEvent(
            call_id=f"call-{node_id}", node_id=node_id,
            input_digest=input_key(ids, model_version, tool_version),
            input_tokens=len(ids), output_digest=hashlib.sha256(output.encode()).hexdigest(),
            paid_cost=paid_cost, model_version=model_version, tool_version=tool_version,
            raw_output_digest=hashlib.sha256(("raw-" + output).encode()).hexdigest(),
        )
        hook_sink(event)
        return CallResult(output, event)


def terminal(node_id, old_output="A", cached=False):
    current_ids = [10 if node_id == "positive" else 20]
    return RuntimeNode(
        node_id=node_id, parents=(), render=lambda _parents: current_ids,
        old_input_digest=input_key(current_ids if cached else [0], "model-v1", "tool-v1"), old_output=old_output,
        model_version="model-v1", tool_version="tool-v1", terminal=True,
    )


class ZeroWeightTests(unittest.TestCase):
    def test_zero_mass_unresolved_terminal_does_not_block_full_resolution(self):
        cert = strict_certificate(("A", "B"), {"positive": 5, "zero": 0}, {"positive": "A"})
        self.assertEqual((cert.status, cert.action, cert.unknown_weight), ("FULL_RESOLVED", "A", 0))

    def test_all_zero_weights_are_invalid(self):
        cert = strict_certificate(("A", "B"), {"a": 0, "b": 0}, {})
        self.assertEqual(cert.status, "INVALID")

    def test_zero_weight_terminal_is_not_a_eligible_paid_query(self):
        nodes = (terminal("positive", "old-A"), terminal("zero", "old-B"))
        broker = OnlineBroker(("A", "B"), nodes, {"positive": 1, "zero": 1}, 100, Backend())
        broker.set_terminal_weights({"positive": 5, "zero": 0})
        self.assertEqual(broker.public_state()["legal_actions"], ["positive"])
        broker.query("positive")
        self.assertEqual(broker.public_state()["certificate"]["status"], "FULL_RESOLVED")
        self.assertEqual([event.node_id for event in broker.events], ["positive"])

    def test_weights_cannot_change_after_paid_terminal_feedback(self):
        broker = OnlineBroker(("A", "B"), (terminal("positive", "old-A"),), {"positive": 1}, 100, Backend())
        broker.query("positive")
        with self.assertRaisesRegex(ValueError, "frozen"):
            broker.set_terminal_weights({"positive": 2})

    def test_zero_weight_exact_cache_is_not_leaked_as_a_known_label(self):
        nodes = (terminal("positive", "A", cached=True), terminal("zero", "B", cached=True))
        broker = OnlineBroker(("A", "B"), nodes, {"positive": 1, "zero": 1}, 100, Backend())
        broker.set_terminal_weights({"positive": 2, "zero": 0})
        state = broker.public_state()
        self.assertEqual(state["known_labels"], {"positive": "A"})
        self.assertEqual(state["certificate"]["status"], "FULL_RESOLVED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
