"""Synthetic CPU-only contract tests. Never import an HF model loader."""
import collections
import copy
import dataclasses
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "frozen_dependencies/src"),
               str(ROOT / "frozen_dependencies/experiments/cycle2_20261113/heterogeneous_qualification_v2"), str(ROOT)]

from graph_credit.executor import Executor, Generation, GraphConfig, canonical, derive_seed, digest
from routed_executor import RoutedBackend
from independent12_executor import (
    AGGREGATE_FORMAT_SUFFIX, AGGREGATE_INPUT_PREFIX, WORKER_NODES,
    Independent12ExecutionError, Independent12Executor,
)


class MockBackend:
    def __init__(self, name, shared, repeat):
        self.identity = {"model": name, "tokenizer_repeat": repeat, "mock_only": True}
        self.name, self.shared, self.repeat = name, shared, repeat
        self.physical_calls = []

    def encode(self, text):
        return [ord(c) for c in text for _ in range(self.repeat)]

    def decode(self, tokens):
        return "".join(chr(tokens[i]) for i in range(0, len(tokens), self.repeat))

    def generate(self, messages, seed, max_new_tokens, config):
        index = len(self.shared["calls"]) + 1
        row = {"model": self.name, "messages": copy.deepcopy(messages), "seed": seed, "index": index}
        self.shared["calls"].append(row)
        input_tokens = len(self.encode(canonical(messages)))
        # Matches the frozen backend's pre-generation input rejection semantics.
        if input_tokens > config.max_input_tokens or index == self.shared.get("unlogged_failure_at"):
            raise ValueError("Mock pre-generation refusal")
        call_id = f"mock-{self.name}-{index}"
        if index == self.shared.get("logged_failure_at"):
            self.physical_calls.append({"call_id": call_id, "status": "FAILED", "seed": seed,
                "input_tokens": input_tokens, "output_tokens": None, "elapsed_seconds": .3,
                "error_type": "RuntimeError"})
            raise RuntimeError("Mock recorded generation failure")
        text = canonical({"answer": f"a{index}", "evidence": f"nonce{index}", "uncertainty": ""})
        if index == self.shared.get("format_failure_at"):
            text = "not-json"
        output_tokens = len(self.encode(text))
        if index == self.shared.get("output_violation_at"):
            output_tokens = max_new_tokens + 1
        generation = Generation(text, input_tokens, output_tokens, .1, .02, 1000 + index, "eos", call_id)
        row["text"] = text
        self.physical_calls.append({"call_id": call_id, "status": "SUCCESS", "seed": seed,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "elapsed_seconds": .1,
            "gpu_seconds": .02, "gpu_peak_allocated_bytes": 1000 + index, "finish_reason": "eos",
            "prompt_sha256": digest(messages)})
        return generation


def make(config=None, **flags):
    state = {"calls": [], **flags}
    qwen, mistral = MockBackend("qwen", state, 1), MockBackend("mistral", state, 2)
    router = RoutedBackend({"qwen": qwen, "mistral": mistral}, ("qwen", "mistral", "qwen", "mistral"), "mistral")
    return Independent12Executor(router, config), router, state


class Independent12Tests(unittest.TestCase):
    def setUp(self):
        self.question = {"id": "synthetic_independent12", "question": "Which marker is requested?",
                         "context": "Complete permitted context: the requested marker is cobalt. Keep this full sentence."}
        self.seed = 99191

    def test_worker_prompts_are_isolated_and_exact_original_round1(self):
        exe, router, state = make()
        result = exe.execute(self.question, self.seed)
        plain = Executor(router, GraphConfig(rounds=3))
        for role in range(4):
            expected, received, own = plain._messages(self.question, f"r1:w{role}", (0,) * 12, {})
            workers = result.slots[role * 3:role * 3 + 3]
            self.assertTrue(all(s.messages == expected for s in workers))
            self.assertEqual((received, own), (0, 0))
            for slot in workers:
                text = canonical(slot.messages)
                self.assertIn(self.question["context"], text)
                self.assertNotIn("previous state", text)
                self.assertNotIn("External messages", text)
                self.assertEqual(slot.self_state_tokens, 0)
                self.assertEqual(slot.received_message_tokens, 0)
                for call in state["calls"]:
                    self.assertNotIn(f"nonce{call['index']}", text)

    def test_aggregate_has_exact_12_slots_in_role_sample_order(self):
        exe, _, _ = make()
        result = exe.execute(self.question, self.seed)
        text = result.slots[-1].messages[1]["content"]
        body = text.split(AGGREGATE_INPUT_PREFIX, 1)[1].removesuffix(AGGREGATE_FORMAT_SUFFIX)
        inbound = json.loads(body)
        expected = [(role, sample) for role in range(4) for sample in range(1, 4)]
        self.assertEqual([(x["role"], x["sample"]) for x in inbound], expected)
        self.assertEqual([x["node"] for x in inbound], list(WORKER_NODES))
        self.assertEqual([x["message"] for x in inbound], [s.record.wire_message for s in result.slots[:12]])
        self.assertIn(self.question["context"], text)

    def test_model_dispatch_6_qwen_7_mistral_and_no_cache(self):
        exe, router, state = make()
        result = exe.execute(self.question, self.seed)
        self.assertEqual(collections.Counter(c["model"] for c in state["calls"]), {"qwen": 6, "mistral": 7})
        self.assertEqual([r["node"] for r in router.route_events], list(WORKER_NODES) + ["aggregate"])
        self.assertEqual(len(result.nodes), 13)
        self.assertTrue(all(not r.reused and r.source_execution_id is None for r in result.nodes.values()))
        self.assertEqual(result.to_dict()["cache_reuse_count"], 0)

    def test_seed_stream_matches_frozen_nodes_but_samples_are_distinct(self):
        exe, _, state = make()
        result = exe.execute(self.question, self.seed)
        expected = [derive_seed(self.question["id"], self.seed, s.node) for s in result.slots]
        self.assertEqual([s.seed for s in result.slots], expected)
        self.assertEqual([c["seed"] for c in state["calls"]], expected)
        self.assertEqual(len(set(expected[:12])), 12)
        other = exe.execute(self.question, self.seed + 1)
        self.assertTrue(all(a.seed != b.seed for a, b in zip(result.slots, other.slots)))

    def test_sender_receiver_accounting_and_full_physical_logical_cost(self):
        exe, router, _ = make()
        result = exe.execute(self.question, self.seed)
        for slot in result.slots:
            backend = router.backends[slot.model]
            self.assertEqual(slot.record.message_metadata["message_tokens"], len(backend.encode(slot.record.wire_message)))
            self.assertEqual(slot.record.output_digest, digest(slot.record.integrity_value()))
        aggregate = result.slots[-1]
        receiving = router.backends["mistral"]
        count = sum(len(receiving.encode(s.record.wire_message)) for s in result.slots[:12])
        sender_sum = sum(s.record.message_metadata["message_tokens"] for s in result.slots[:12])
        self.assertEqual(aggregate.received_message_tokens, count)
        self.assertNotEqual(count, sender_sum)
        payload = result.to_dict()
        self.assertEqual(payload["physical_cost"], payload["logical_service_cost"])
        cost = payload["physical_cost"]
        self.assertEqual(cost["calls"], 13)
        self.assertEqual(cost["input_tokens"], sum(s.generation.input_tokens for s in result.slots))
        self.assertEqual(cost["output_tokens"], sum(s.generation.output_tokens for s in result.slots))
        self.assertEqual(cost["received_message_tokens"], count)
        self.assertEqual(cost["self_state_tokens"], 0)
        self.assertTrue(cost["cost_fields_complete"] and cost["service_complete"])
        self.assertIsNone(cost["resident_GPU_hours"])

    def test_format_failure_is_paid_and_keeps_its_aggregate_slot(self):
        exe, _, state = make(format_failure_at=4)
        result = exe.execute(self.question, self.seed)
        bad = result.slots[3]
        self.assertEqual(bad.status, "FORMAT_ERROR")
        self.assertEqual(bad.generation.text, "not-json")
        self.assertEqual(bad.record.parsed, {"answer": "", "evidence": "", "uncertainty": "FORMAT_ERROR"})
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(len(state["calls"]), 13)
        self.assertEqual(result.to_dict()["format_failures"], 1)
        self.assertEqual(result.cost()["returned_generations"], 13)
        text = result.slots[-1].messages[1]["content"]
        inbound = json.loads(text.split(AGGREGATE_INPUT_PREFIX, 1)[1].removesuffix(AGGREGATE_FORMAT_SUFFIX))
        self.assertEqual(len(inbound), 12)
        self.assertEqual(inbound[3]["message"], bad.record.wire_message)

    def test_logged_backend_failure_aborts_preserves_slots_and_releases_lock(self):
        exe, router, state = make(logged_failure_at=3)
        with self.assertRaises(Independent12ExecutionError) as caught:
            exe.execute(self.question, self.seed)
        result = caught.exception.partial_result
        self.assertEqual([s.status for s in result.slots], ["SUCCESS", "SUCCESS", "BACKEND_ERROR"] + ["NOT_ATTEMPTED"] * 10)
        self.assertEqual(len(state["calls"]), 3)
        self.assertEqual(len(router.route_events), 3)
        self.assertEqual(result.slots[2].physical_events[0]["status"], "FAILED")
        self.assertIsNone(result.answer)
        costs = result.cost()
        self.assertEqual(costs["calls"], 3)
        self.assertEqual(costs["unattempted_calls"], 10)
        self.assertIsNotNone(costs["input_tokens"])
        self.assertIsNone(costs["output_tokens"])
        self.assertIsNone(costs["gpu_seconds"])
        self.assertAlmostEqual(costs["model_elapsed_seconds"], .5)
        self.assertFalse(costs["service_complete"])
        with self.assertRaises(RuntimeError):
            router.encode("outside active node")
        state["logged_failure_at"] = None
        self.assertEqual(exe.execute(self.question, self.seed).status, "COMPLETE")

    def test_unlogged_preflight_failure_costs_remain_unknown(self):
        exe, _, state = make(unlogged_failure_at=3)
        with self.assertRaises(Independent12ExecutionError) as caught:
            exe.execute(self.question, self.seed)
        result = caught.exception.partial_result
        costs = result.cost()
        self.assertEqual(len(state["calls"]), 3)
        self.assertEqual(costs["failed_requests_without_physical_record"], 1)
        self.assertIsNone(costs["input_tokens"])
        self.assertIsNone(costs["output_tokens"])
        self.assertIsNone(costs["model_elapsed_seconds"])
        self.assertGreater(costs["known_cost_components"]["input_tokens"], 0)
        self.assertEqual(result.to_dict()["logical_service_cost"]["calls"], 3)

    def test_backend_input_cap_rejection_does_not_truncate_context(self):
        exe, _, state = make(GraphConfig(rounds=3, max_input_tokens=128))
        with self.assertRaises(Independent12ExecutionError) as caught:
            exe.execute(self.question, self.seed)
        result = caught.exception.partial_result
        self.assertEqual(len(state["calls"]), 1)
        self.assertIn(self.question["context"], state["calls"][0]["messages"][1]["content"])
        self.assertEqual(result.slots[0].status, "BACKEND_ERROR")
        self.assertTrue(all(s.status == "NOT_ATTEMPTED" for s in result.slots[1:]))

    def test_post_generation_bound_failure_preserves_generation_and_charge(self):
        exe, _, state = make(output_violation_at=2)
        with self.assertRaises(Independent12ExecutionError) as caught:
            exe.execute(self.question, self.seed)
        result = caught.exception.partial_result
        self.assertEqual(len(state["calls"]), 2)
        self.assertEqual(result.slots[1].status, "POSTPROCESS_ERROR")
        self.assertIsNotNone(result.slots[1].generation)
        self.assertEqual(result.cost()["returned_generations"], 2)
        self.assertEqual(result.cost()["output_tokens"], sum(s.generation.output_tokens for s in result.slots[:2]))

    def test_aggregate_failure_preserves_all_twelve_candidates(self):
        exe, _, state = make(logged_failure_at=13)
        with self.assertRaises(Independent12ExecutionError) as caught:
            exe.execute(self.question, self.seed)
        result = caught.exception.partial_result
        self.assertEqual(len(result.slots), 13)
        self.assertTrue(all(s.record is not None for s in result.slots[:12]))
        self.assertEqual(result.slots[-1].status, "BACKEND_ERROR")
        self.assertEqual(len(state["calls"]), 13)
        self.assertIsNone(result.answer)

    def test_protocol_identity_is_distinct_and_binds_model_identity(self):
        exe, router, _ = make()
        plain = Executor(router, GraphConfig(rounds=3))
        self.assertNotEqual(exe.protocol_hash, plain.protocol_hash)
        other, _, _ = make()
        self.assertEqual(exe.protocol_hash, other.protocol_hash)
        other.backend.backends["qwen"].identity["revision"] = "new"
        changed_router = RoutedBackend(other.backend.backends, ("qwen", "mistral", "qwen", "mistral"), "mistral")
        changed = Independent12Executor(changed_router)
        self.assertNotEqual(exe.protocol_hash, changed.protocol_hash)
        self.assertEqual(exe.protocol_identity["samples_per_role"], 3)
        self.assertEqual(exe.protocol_identity["worker_node_order"], list(WORKER_NODES))

    def test_incompatible_configuration_and_cache_arguments_rejected(self):
        with self.assertRaises(ValueError):
            make(GraphConfig(rounds=2))
        exe, router, state = make()
        with self.assertRaises(TypeError):
            exe.execute(self.question, self.seed, base=None)
        with self.assertRaises(TypeError):
            exe.execute(self.question, self.seed, mask=(0,) * 12)
        with self.assertRaises(ValueError):
            exe.execute({**self.question, "gold": "hidden"}, self.seed)
        router.worker_models = ("mistral", "qwen", "qwen", "mistral")
        with self.assertRaises(RuntimeError):
            exe.execute(self.question, self.seed)
        self.assertEqual(len(state["calls"]), 0)

    def test_concurrent_and_mutated_protocol_rejected_before_calls(self):
        exe, router, state = make()
        router._execution_lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                exe.execute(self.question, self.seed)
        finally:
            router._execution_lock.release()
        exe.protocol_identity["samples_per_role"] = 2
        with self.assertRaises(RuntimeError):
            exe.execute(self.question, self.seed)
        self.assertEqual(state["calls"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
