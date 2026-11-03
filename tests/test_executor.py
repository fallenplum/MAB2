"""Deterministic semantics tests; these are not scientific model results."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graph_credit.executor import (CacheIntegrityError, Executor, Generation, GraphConfig,
                                   AGGREGATOR_PROMPT, MESSAGE_INSTRUCTION, ROLE_PROMPTS,
                                   affected_closure, canonical, logical_edges, normalize_message,
                                   unrolled_dependencies, validate_mask)


class FakeBackend:
    identity = {"backend": "deterministic_test_only", "weights_sha256": "fake-model-v1", "tokenizer": "unicode_codepoints"}

    def __init__(self, constant=False):
        self.calls = []
        self.constant = constant

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)

    def generate(self, messages, seed, max_new_tokens, config):
        payload = canonical([messages, seed])
        answer = "same" if self.constant else hashlib.sha256(payload.encode()).hexdigest()[:12]
        text = canonical({"answer": answer, "evidence": "checked", "uncertainty": ""})
        result = Generation(text, len(payload), len(text), .01, .008, 4096, "eos")
        self.calls.append(result)
        return result


class ExecutionSemantics(unittest.TestCase):
    def setUp(self):
        self.question = {"id": "dev-question-0", "question": "How much is 2 plus 2?", "task": "gsm8k", "split": "dev"}
        self.full = [1] * 12
        self.changed = [0] + [1] * 11

    def compare_nodes(self, left, right):
        self.assertEqual(set(left.nodes), set(right.nodes))
        for node in left.nodes:
            self.assertEqual(left.nodes[node].messages, right.nodes[node].messages, node)
            self.assertEqual(left.nodes[node].generation.text, right.nodes[node].generation.text, node)
            self.assertEqual(left.nodes[node].wire_message, right.nodes[node].wire_message, node)
            self.assertEqual(left.nodes[node].seed, right.nodes[node].seed, node)

    def test_two_round_closure_is_receiver_and_aggregator(self):
        backend = FakeBackend()
        executor = Executor(backend)
        original = executor.execute(self.question, self.full, 1701)
        partial = executor.execute(self.question, self.changed, 1701, base=original)
        full = executor.execute(self.question, self.changed, 1701)
        self.compare_nodes(partial, full)
        self.assertEqual(set(partial.closure), {"r2:w1", "aggregate"})
        self.assertEqual(partial.cost()["calls"], 2)
        self.assertEqual(partial.cost(False)["calls"], 9)
        self.assertFalse(partial.nodes["r2:w1"].reused)
        self.assertTrue(partial.nodes["r2:w0"].reused)

    def test_three_round_union_closure_includes_fixed_self_and_all_descendants(self):
        backend = FakeBackend()
        executor = Executor(backend, GraphConfig(rounds=3))
        original = executor.execute(self.question, self.full, 1701)
        partial = executor.execute(self.question, self.changed, 1701, base=original)
        full = executor.execute(self.question, self.changed, 1701)
        self.compare_nodes(partial, full)
        self.assertEqual(set(partial.closure), {"r2:w1", "r3:w0", "r3:w1", "r3:w2", "r3:w3", "aggregate"})
        self.assertEqual(partial.cost()["calls"], 6)

    def test_fixed_self_transmits_dependency_without_other_edges(self):
        empty, one = [0] * 12, [1] + [0] * 11
        self.assertEqual(affected_closure(empty, one, 3), {"r2:w1", "r3:w1", "aggregate"})
        self.assertEqual(affected_closure(one, empty, 3), {"r2:w1", "r3:w1", "aggregate"})

    def test_random_masks_replay_matches_full_for_additions_deletions_and_multiple_edits(self):
        rng = random.Random(2718)
        for rounds in (2, 3):
            for _ in range(12):
                old = [rng.randrange(2) for i in range(12)]
                new = old.copy()
                for i in rng.sample(range(12), rng.randrange(1, 5)):
                    new[i] ^= 1
                executor = Executor(FakeBackend(), GraphConfig(rounds=rounds))
                original = executor.execute(self.question, old, 99)
                partial = executor.execute(self.question, new, 99, base=original)
                full = executor.execute(self.question, new, 99)
                self.compare_nodes(partial, full)

    def test_empty_graph_has_self_state_and_fixed_aggregation(self):
        result = Executor(FakeBackend()).execute(self.question, [0] * 12, 19)
        for name, node in result.nodes.items():
            if name != "aggregate":
                self.assertEqual(node.received_message_tokens, 0)
            if name.startswith("r2"):
                self.assertGreater(node.self_state_tokens, 0)
                self.assertIn("no external messages", node.messages[1]["content"])
        self.assertGreater(result.nodes["aggregate"].received_message_tokens, 0)
        self.assertEqual(result.cost()["calls"], 9)

    def test_node_seed_pairing_ignores_graph_but_changes_with_execution_seed(self):
        executor = Executor(FakeBackend())
        old = executor.execute(self.question, self.full, 1)
        new = executor.execute(self.question, self.changed, 1, base=old)
        other_seed = executor.execute(self.question, self.changed, 2, base=new)
        for name in old.nodes:
            self.assertEqual(old.nodes[name].seed, new.nodes[name].seed)
            self.assertNotEqual(old.nodes[name].seed, other_seed.nodes[name].seed)
        self.assertEqual(other_seed.cost()["calls"], 9)

    def test_question_or_model_or_prompt_or_generation_change_disables_reuse(self):
        backend = FakeBackend()
        executor = Executor(backend)
        base = executor.execute(self.question, self.full, 17)
        changed_question = {**self.question, "question": "A different question under the same ID"}
        self.assertEqual(executor.execute(changed_question, self.full, 17, base=base).cost()["calls"], 9)
        changed_model = Executor(backend, model_identity={**backend.identity, "weights_sha256": "different"})
        self.assertEqual(changed_model.execute(self.question, self.full, 17, base=base).cost()["calls"], 9)
        changed_prompt = Executor(backend, GraphConfig(role_prompts=("different",) + GraphConfig().role_prompts[1:]))
        self.assertEqual(changed_prompt.execute(self.question, self.full, 17, base=base).cost()["calls"], 9)
        changed_sampling = Executor(backend, GraphConfig(temperature=.6))
        self.assertEqual(changed_sampling.execute(self.question, self.full, 17, base=base).cost()["calls"], 9)

    def test_output_cache_tampering_rejected(self):
        executor = Executor(FakeBackend())
        base = executor.execute(self.question, self.full, 3)
        base.nodes["r1:w0"].parsed["answer"] = "tampered"
        with self.assertRaises(CacheIntegrityError):
            executor.execute(self.question, self.changed, 3, base=base)

    def test_input_cache_tampering_rejected(self):
        executor = Executor(FakeBackend())
        base = executor.execute(self.question, self.full, 3)
        base.nodes["r1:w0"].messages[1]["content"] = "tampered"
        with self.assertRaises(CacheIntegrityError):
            executor.execute(self.question, self.changed, 3, base=base)

    def test_affected_nodes_always_reexecuted_even_identical_prompt_and_output(self):
        executor = Executor(FakeBackend(constant=True))
        base = executor.execute(self.question, self.full, 4)
        partial = executor.execute(self.question, self.changed, 4, base=base)
        self.assertEqual(partial.cost()["calls"], 2)
        self.assertNotEqual(base.nodes["aggregate"].generation.call_id, partial.nodes["aggregate"].generation.call_id)

    def test_physical_ledger_charges_unique_new_calls_once_and_boundary_messages(self):
        backend = FakeBackend()
        executor = Executor(backend)
        base = executor.execute(self.question, self.full, 7)
        partial = executor.execute(self.question, self.changed, 7, base=base)
        self.assertEqual(len(backend.calls), base.cost()["calls"] + partial.cost()["calls"])
        call_ids = {n.generation.call_id for r in (base, partial) for n in r.nodes.values()}
        self.assertEqual(len(call_ids), 11)
        self.assertEqual(base.cost()["input_tokens"] + partial.cost()["input_tokens"], sum(c.input_tokens for c in backend.calls))
        self.assertEqual(base.cost()["output_tokens"] + partial.cost()["output_tokens"], sum(c.output_tokens for c in backend.calls))
        self.assertGreater(partial.cost()["received_message_tokens"], 0)
        unchanged = executor.execute(self.question, self.changed, 7, base=partial)
        self.assertEqual(unchanged.cost()["calls"], 0)
        self.assertEqual(unchanged.cost()["received_message_tokens"], 0)
        self.compare_nodes(unchanged, partial)

    def test_message_cap_counts_wire_schema_and_preserves_raw_generation(self):
        backend = FakeBackend()
        text = canonical({"answer": "42", "evidence": "e" * 500, "uncertainty": "u" * 100})
        serialized, parsed, metadata = normalize_message(text, backend, 128)
        self.assertLessEqual(len(backend.encode(serialized)), 128)
        self.assertEqual(parsed["answer"], "42")
        self.assertTrue(metadata["format_ok"])
        self.assertTrue(metadata["message_trimmed"])
        self.assertEqual(json.loads(serialized), parsed)
        self.assertIn("e" * 500, text)

    def test_malformed_message_repair_is_explicit_and_model_free(self):
        backend = FakeBackend()
        serialized, parsed, metadata = normalize_message("Unstructured candidate: 42", backend, 128)
        self.assertFalse(metadata["format_ok"])
        self.assertFalse(metadata["type_normalized"])
        self.assertEqual(parsed["answer"], "")
        self.assertEqual(parsed["evidence"], "")
        self.assertEqual(parsed["uncertainty"], "FORMAT_ERROR")
        self.assertEqual(len(backend.calls), 0)
        self.assertEqual(json.loads(serialized), parsed)

    def test_complete_numerical_answers_are_normalized_without_fabricating_evidence(self):
        backend = FakeBackend()
        for value, expected in ((25, "25"), (-7, "-7"), (2.5, "2.5"), (25.0, "25.0")):
            raw = json.dumps({"answer": value, "evidence": "5 groups of 5.", "uncertainty": ""})
            serialized, parsed, metadata = normalize_message(raw, backend, 128)
            self.assertTrue(metadata["format_ok"])
            self.assertTrue(metadata["type_normalized"])
            self.assertIsNone(metadata["format_error"])
            self.assertEqual(parsed["answer"], expected)
            self.assertEqual(parsed["evidence"], "5 groups of 5.")
            self.assertEqual(json.loads(serialized), parsed)
        _, _, metadata = normalize_message('{"answer":"25","evidence":"checked","uncertainty":""}', backend, 128)
        self.assertFalse(metadata["type_normalized"])

    def test_complete_fenced_json_is_deterministically_accepted(self):
        backend = FakeBackend()
        for tag in ("json", "JSON", ""):
            raw = '```' + tag + '\n{"answer":25,"evidence":"5 times 5.","uncertainty":""}\n```'
            serialized, parsed, metadata = normalize_message(raw, backend, 128)
            self.assertTrue(metadata["format_ok"])
            self.assertTrue(metadata["type_normalized"])
            self.assertEqual(parsed["answer"], "25")
            self.assertEqual(json.loads(serialized), parsed)
            self.assertEqual(len(backend.calls), 0)

    def test_truncated_json_and_incomplete_schema_are_explicit_empty_errors(self):
        backend = FakeBackend()
        invalid = [
            '{"answer":"25","evidence":"5 times',
            '{"answer":"25","evidence":"checked"',
            '{"answer":"25"}',
            '{"answer":25,"uncertainty":""}',
            '```json\n{"answer":"25","evidence":"checked","uncertainty":""}',
            '{"answer":"25","evidence":null,"uncertainty":""}',
        ]
        for raw in invalid:
            serialized, parsed, metadata = normalize_message(raw, backend, 128)
            self.assertFalse(metadata["format_ok"], raw)
            self.assertFalse(metadata["type_normalized"], raw)
            self.assertEqual(parsed, {"answer": "", "evidence": "", "uncertainty": "FORMAT_ERROR"})
            self.assertTrue(metadata["format_error"])
            self.assertNotIn(raw, serialized)
        self.assertEqual(len(backend.calls), 0)

    def test_non_finite_boolean_and_ambiguous_answers_are_rejected(self):
        backend = FakeBackend()
        invalid = [
            '{"answer":NaN,"evidence":"checked","uncertainty":""}',
            '{"answer":Infinity,"evidence":"checked","uncertainty":""}',
            '{"answer":1e309,"evidence":"checked","uncertainty":""}',
            '{"answer":true,"evidence":"checked","uncertainty":""}',
            '{"answer":null,"evidence":"checked","uncertainty":""}',
            '{"answer":"25","answer":"26","evidence":"checked","uncertainty":""}',
            '{"answer":"25","evidence":"checked","uncertainty":""} {"answer":"26"}',
        ]
        for raw in invalid:
            _, parsed, metadata = normalize_message(raw, backend, 128)
            self.assertFalse(metadata["format_ok"], raw)
            self.assertEqual(parsed["answer"], "")
            self.assertEqual(parsed["evidence"], "")
            self.assertEqual(parsed["uncertainty"], "FORMAT_ERROR")

    def test_error_messages_preserve_original_raw_generation_and_full_cost(self):
        class TruncatedBackend(FakeBackend):
            def generate(self, messages, seed, max_new_tokens, config):
                raw = '{"answer":"25","evidence":"' + "x" * (max_new_tokens - 27)
                raw = raw[:max_new_tokens]
                result = Generation(raw, len(canonical(messages)), len(raw), .01, .008, 4096, "length")
                self.calls.append(result)
                return result
        backend = TruncatedBackend()
        result = Executor(backend).execute(self.question, self.full, 99)
        for name, node in result.nodes.items():
            self.assertFalse(node.message_metadata["format_ok"], name)
            self.assertTrue(node.generation.text.startswith('{"answer":"25","evidence":"'))
            self.assertEqual(node.generation.finish_reason, "length")
            self.assertEqual(node.parsed["answer"], "")
            self.assertEqual(node.parsed["evidence"], "")
            self.assertEqual(node.parsed["uncertainty"], "FORMAT_ERROR")
        self.assertEqual(result.cost()["calls"], len(backend.calls))
        self.assertEqual(result.cost()["output_tokens"], sum(c.output_tokens for c in backend.calls))
        self.assertGreater(result.cost()["output_tokens"], 0)

    def test_v2_prompts_require_quoted_compact_messages_and_internal_checks(self):
        self.assertEqual(GraphConfig().executor_version, "unrolled_closed_input_v3")
        self.assertIn('"answer":"25"', MESSAGE_INSTRUCTION)
        self.assertIn("at most 20 words", MESSAGE_INSTRUCTION)
        self.assertIn("at most 5 words", MESSAGE_INSTRUCTION)
        self.assertIn("single-line", MESSAGE_INSTRUCTION)
        self.assertTrue(all("internally" in prompt for prompt in ROLE_PROMPTS))

    def test_v3_aggregator_system_and_user_require_all_three_fields_in_order(self):
        template = '{"answer":"<final answer>","uncertainty":"","evidence":"<brief check>"}'
        self.assertIn(template, AGGREGATOR_PROMPT)
        self.assertIn("ALL THREE keys are mandatory", AGGREGATOR_PROMPT)
        self.assertIn('Never omit "uncertainty"', AGGREGATOR_PROMPT)
        result = Executor(FakeBackend()).execute(self.question, self.full, 91)
        user = result.nodes["aggregate"].messages[1]["content"]
        tail = user.split("Required final format:", 1)[1]
        self.assertIn(template, tail)
        self.assertIn("Include exactly all 3 keys in that order", tail)
        self.assertIn("Never omit uncertainty, even when empty", tail)
        self.assertTrue(tail.endswith("Return only the complete single-line JSON object."))
        for node, record in result.nodes.items():
            if node != "aggregate":
                self.assertNotIn("Required final format:", record.messages[1]["content"])
                self.assertTrue(record.messages[0]["content"].endswith(MESSAGE_INSTRUCTION))

    def test_source_order_and_no_within_round_dependencies(self):
        result = Executor(FakeBackend()).execute(self.question, self.full, 81)
        prompt = result.nodes["r2:w0"].messages[1]["content"]
        self.assertLess(prompt.index('"worker":1'), prompt.index('"worker":2'))
        self.assertLess(prompt.index('"worker":2'), prompt.index('"worker":3'))
        deps = unrolled_dependencies(self.full, 3)
        for node, parents in deps.items():
            if node != "aggregate":
                r = int(node.split(":")[0][1:])
                self.assertTrue(all(int(parent.split(":")[0][1:]) == r - 1 for parent in parents))

    def test_gold_fields_and_invalid_masks_fail_closed(self):
        for field in ("answer", "label", "gold", "target", "solution", "quality"):
            with self.assertRaises(ValueError):
                Executor.question_payload({**self.question, field: "4"})
        for mask in ([1] * 11, [1] * 13, [0] * 11 + [2], [0] * 11 + ["1"], [0] * 11 + [1.]):
            with self.assertRaises(ValueError):
                validate_mask(mask)
        self.assertEqual(logical_edges()[0], (0, 1))
        self.assertEqual(logical_edges()[4], (1, 2))

    def test_qualification_branch_records_both_round_counts_and_raw_format_counts(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_pilot.py"
        spec = importlib.util.spec_from_file_location("pilot_qualification_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            result = module.qualify(FakeBackend(), self.question, GraphConfig(), 1701, Path(temp))
            self.assertEqual(result["status"], "PASS")
            self.assertEqual([c["rounds"] for c in result["checks"]], [2, 3])
            self.assertFalse(result["scientific_completion"])
            self.assertTrue((Path(temp) / "qualification_executions" / "r3_empty_graph.json").is_file())
            self.assertEqual(result["checks"][0]["format_counts"]["full_first"]["nodes"], 9)

    def test_qualification_rejects_hidden_mutable_generation_state(self):
        class StatefulBackend(FakeBackend):
            def generate(self, messages, seed, max_new_tokens, config):
                result = super().generate(messages, seed, max_new_tokens, config)
                text = canonical({"answer": str(len(self.calls)), "evidence": "mutable", "uncertainty": ""})
                return dataclasses.replace(result, text=text, output_tokens=len(text))
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_pilot.py"
        spec = importlib.util.spec_from_file_location("pilot_qualification_failure_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            result = module.qualify(StatefulBackend(), self.question, GraphConfig(), 1701, Path(temp))
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["checks"][0]["same_input_seed_full_repeat_equal"])


if __name__ == "__main__":
    unittest.main()
