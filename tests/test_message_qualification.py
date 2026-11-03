"""Message validity must qualify separately from deterministic graph replay.

All generations in this file are synthetic CPU contract tests. A passing test
never supplies real-model S0, population-format, or scientific G1 evidence.
"""
from __future__ import annotations

import importlib.util
import dataclasses
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graph_credit.executor import Generation, GraphConfig, canonical

SPEC = importlib.util.spec_from_file_location("qualification_run_pilot", ROOT / "scripts" / "run_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


def record(*, valid=True, trimmed=False, reason="eos", normalized=False, reused=False):
    return SimpleNamespace(
        reused=reused,
        message_metadata={"format_ok": valid, "message_trimmed": trimmed,
                          "type_normalized": normalized},
        generation=SimpleNamespace(finish_reason=reason),
    )


def balanced_records(per_role=25, aggregators=10):
    return [(f"r{1 + i % 2}:w{w}", record()) for w in range(4) for i in range(per_role)] + [
        ("aggregate", record()) for _ in range(aggregators)]


class QualificationBackend:
    """Numeric JSON answers exercise normalization, not an LLM capability."""
    identity = {"backend": "synthetic_message_qualification_test", "weights": "none"}

    def __init__(self, invalid=False, length=False):
        self.invalid, self.length = invalid, length
        self.calls = []

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(chr(value) for value in tokens)

    def generate(self, messages, seed, max_new_tokens, config):
        text = '{"answer":4,"evidence":"2+2","uncertainty":""}'
        if self.invalid:
            text = '{"answer":4,"evidence":"unfinished'
        result = Generation(text, len(canonical(messages)), len(text), .001,
                            finish_reason="length" if self.length else "eos")
        self.calls.append(result)
        return result


class MessageGateTests(unittest.TestCase):
    def test_valid_records_report_each_role_and_accept_numeric_normalization(self):
        records = balanced_records()
        records[0][1].message_metadata["type_normalized"] = True
        result = pilot.evaluate_message_records(records)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["workers"]["generated"], 100)
        self.assertEqual(result["aggregator"]["generated"], 10)
        self.assertEqual(result["workers"]["type_normalized"], 1)
        self.assertEqual(result["workers_by_role"]["w0"]["generated"], 25)

    def test_aggregate_compliance_cannot_hide_one_role_failure(self):
        records = balanced_records()
        for _, node in records[:4]:
            node.message_metadata["format_ok"] = False
        result = pilot.evaluate_message_records(records)
        self.assertEqual(result["workers"]["format_valid_rate"], .96)
        self.assertEqual(result["workers_by_role"]["w0"]["format_valid_rate"], .84)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("w0: format_valid_rate" in reason for reason in result["reasons"]))

    def test_length_finish_and_wire_trimming_have_separate_limits(self):
        for failure in ("length", "trimmed"):
            with self.subTest(failure=failure):
                records = balanced_records()
                for _, node in records[:3]:
                    if failure == "length":
                        node.generation.finish_reason = "length"
                    else:
                        node.message_metadata["message_trimmed"] = True
                result = pilot.evaluate_message_records(records)
                self.assertEqual(result["workers"]["format_valid_rate"], 1)
                self.assertEqual(result["status"], "FAIL")

    def test_cached_invalid_occurrences_do_not_pollute_physical_denominator(self):
        records = balanced_records()
        records.extend(("r2:w0", record(valid=False, reason="length", reused=True)) for _ in range(100))
        result = pilot.evaluate_message_records(records)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["physical_generations"], 110)
        self.assertEqual(result["reused_node_occurrences_excluded"], 100)
        self.assertEqual(result["workers"]["format_invalid"], 0)

    def test_aggregator_is_qualified_and_no_aggregator_fails_closed(self):
        records = balanced_records()
        for name, node in records:
            if name == "aggregate":
                node.message_metadata["format_ok"] = False
        result = pilot.evaluate_message_records(records)
        self.assertEqual(result["workers"]["format_valid_rate"], 1)
        self.assertEqual(result["status"], "FAIL")
        result = pilot.evaluate_message_records(balanced_records(aggregators=0))
        self.assertEqual(result["status"], "FAIL")

    def test_missing_metadata_or_unknown_finish_reason_fails_closed(self):
        records = balanced_records()
        del records[0][1].message_metadata["format_ok"]
        records[1][1].generation.finish_reason = "unknown"
        result = pilot.evaluate_message_records(records)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["workers"]["missing_or_invalid_metadata"], 1)
        self.assertEqual(result["workers"]["unknown_finish_reasons"], 1)

    def test_threshold_config_rejects_typos_invalid_values_and_wrong_types(self):
        for config in ({"worker_min_valid_rate": .9}, {"worker_min_format_valid_rate": float("nan")},
                       {"worker_max_trimmed_rate": -1}, {"require_each_role": 1}, []):
            with self.subTest(config=config), self.assertRaises(ValueError):
                pilot.evaluate_message_records(balanced_records(), config)

    def test_frozen_limits_are_inclusive_at_exact_threshold(self):
        records = balanced_records(per_role=10, aggregators=10)
        for role in range(4):
            node = records[role * 10][1]
            node.message_metadata["format_ok"] = False
            node.message_metadata["message_trimmed"] = True
            node.generation.finish_reason = "length"
        result = pilot.evaluate_message_records(records)
        self.assertEqual(result["workers"]["format_valid_rate"], .9)
        self.assertEqual(result["workers"]["length_finish_rate"], .1)
        self.assertEqual(result["status"], "PASS")


class CombinedQualificationTests(unittest.TestCase):
    question = {"id": "synthetic-qual", "question": "What is 2+2?", "task": "gsm8k", "split": "qualification"}

    def run_qualification(self, backend):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            return pilot.qualify(backend, self.question, GraphConfig(), 1701, Path(tmp))

    def test_valid_numeric_schema_passes_execution_and_message_gates(self):
        backend = QualificationBackend()
        result = self.run_qualification(backend)
        self.assertEqual(len(backend.calls), 96)
        self.assertEqual(result["execution_gate"]["status"], "PASS")
        self.assertEqual(result["message_gate"]["status"], "PASS")
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["eligible_for_G1"])
        self.assertEqual(result["message_gate"]["physical_generations"], 96)
        self.assertEqual(result["message_gate"]["workers"]["type_normalized"], 86)
        self.assertFalse(result["scientific_completion"])
        self.assertEqual(result["qualification_question_count"], 1)

    def test_identical_malformed_outputs_pass_replay_but_cannot_release_G1(self):
        result = self.run_qualification(QualificationBackend(invalid=True))
        self.assertEqual(result["execution_gate"]["status"], "PASS")
        self.assertEqual(result["message_gate"]["status"], "FAIL")
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["eligible_for_G1"])
        self.assertTrue(all(check["execution_passed"] and not check["message_passed"] for check in result["checks"]))

    def test_schema_valid_length_terminated_outputs_still_fail_message_gate(self):
        result = self.run_qualification(QualificationBackend(length=True))
        self.assertEqual(result["execution_gate"]["status"], "PASS")
        self.assertEqual(result["message_gate"]["workers"]["format_valid_rate"], 1)
        self.assertEqual(result["message_gate"]["workers"]["length_finish_rate"], 1)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["eligible_for_G1"])

    def test_round_failure_cannot_be_hidden_by_pooled_message_rates(self):
        class FirstCallLength(QualificationBackend):
            def generate(self, messages, seed, max_new_tokens, config):
                generation = super().generate(messages, seed, max_new_tokens, config)
                return dataclasses.replace(generation, finish_reason="length") if len(self.calls) == 1 else generation
        result = self.run_qualification(FirstCallLength())
        # One failure in w0's eight fresh 2-round calls exceeds 0.1. In the
        # pooled 2/3-round report it would be diluted below the same threshold.
        self.assertEqual(result["execution_gate"]["status"], "PASS")
        self.assertLess(result["message_gate"]["workers_by_role"]["w0"]["length_finish_rate"], .1)
        self.assertEqual(result["message_gate"]["status"], "FAIL")
        self.assertEqual(result["message_gate"]["round_failures"][0]["rounds"], 2)
        self.assertFalse(result["eligible_for_G1"])


if __name__ == "__main__":
    unittest.main()
