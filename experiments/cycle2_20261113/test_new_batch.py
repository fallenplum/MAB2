"""Synthetic-only tests; no real questions, labels, torch imports or GPU jobs."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import new_batch as nb
from make_plan_template import make_plan

SOURCE = Path(__file__).parent/"source_audit"
PINS = {p: nb.file_sha(SOURCE/p) for p in nb.REQUIRED_SOURCES}
M, _ = nb.load_operator(SOURCE, PINS)


class FakeBackend:
    identity = {"test_only": "deterministic synthetic backend"}
    def __init__(self, fail_at=None): self.calls = 0; self.fail_at = fail_at
    def encode(self, text): return list(text.encode("utf-8"))
    def decode(self, tokens): return bytes(tokens).decode("utf-8")
    def generate(self, messages, seed, max_new_tokens, config):
        self.calls += 1
        if self.calls == self.fail_at: raise RuntimeError("synthetic injected failure")
        answer = str(int(nb.digest([messages, seed])[:6], 16) % 100)
        text = nb.canonical({"answer": answer, "evidence": "synthetic", "uncertainty": ""})
        return M.Generation(text, len(nb.canonical(messages)), len(text), .001, .0005, 1, "eos")


def question(i):
    text = "Synthetic question number "+str(i)
    return {"id": "gsm8k_"+nb.question_text_sha(text)[:20], "question": text, "source_split": "train"}


def episodes():
    return [{"episode": 2*k+c, "seed": 3701+2*k+c,
             "permutation": [(j+k)%4 for j in range(4)], "complement": bool(c)}
            for k in range(4) for c in range(2)]


def exclusion():
    return {"schema": "mab2_cycle2_exclusion_v1", "excluded_ids": [],
            "coverage": {**{k: "ALL_KNOWN_RECORDED" for k in nb.EXCLUSION_KINDS},
                         "literature_exposure": "KNOWN_RECORDS_REVIEWED"},
            "unknown_exposure_beyond_recorded": True}


class BatchTests(unittest.TestCase):
    def setUp(self):
        # Windows app sandbox redirects default TEMP through a reparse device;
        # use this writable stage for realistic same-filesystem atomic renames.
        test_root = Path(__file__).parent/"synthetic_test_tmp"; test_root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=test_root); self.root = Path(self.tmp.name)
        self.q = question(0)
        self.unit = {"unit_id": "ep0_C_00", "question_id": self.q["id"], "phase": "C",
                     "episode": 0, "seed": 3701, "worker_index": 0}
        self.plan = {"max_unit_attempts": 2, "episodes": [{**e, "menu": nb.episode_menu(e["permutation"], e["complement"])} for e in episodes()]}
        self.context = {"plan": self.plan, "plan_sha256": "f"*64, "module": M,
                        "questions": {self.q["id"]: self.q}, "output": self.root/"results"}
        self.worker = self.root/"worker_0"
    def tearDown(self): self.tmp.cleanup()
    def executor(self, backend=None, cap=1000):
        backend = backend or FakeBackend()
        journal = nb.JournalBackend(backend, self.worker/"call_journal.jsonl", cap)
        return M.Executor(journal), journal, backend
    def test_menu_transform_and_physical_costs(self):
        for ep in self.plan["episodes"]:
            nb.validate_menu(ep["menu"], M)
            self.assertEqual(nb.estimated_calls(ep["menu"], M), 37)
            self.assertEqual(nb.estimated_calls(ep["menu"][:7], M), 21)
        self.assertEqual(len({tuple(e["menu"][0]["mask"]) for e in self.plan["episodes"]}), 8)
    def test_collection_real_executor_and_completed_resume(self):
        ex, journal, backend = self.executor()
        receipt, fresh = nb.collect_one_unit(self.context, ex, journal, self.unit, self.worker)
        self.assertTrue(fresh); self.assertEqual(backend.calls, 37)
        self.assertEqual(receipt["logical_calls"], 126)
        self.assertEqual(receipt["physical_calls_referenced"], 37)
        before = {p.name: nb.file_sha(p) for p in (self.worker/"units"/self.unit["unit_id"]).glob("*.json")}
        same, fresh = nb.collect_one_unit(self.context, ex, journal, self.unit, self.worker)
        self.assertFalse(fresh); self.assertEqual(receipt, same); self.assertEqual(backend.calls, 37)
        self.assertEqual(before, {p.name: nb.file_sha(p) for p in (self.worker/"units"/self.unit["unit_id"]).glob("*.json")})
    def test_I_collects_only_A_menu(self):
        self.unit["phase"] = "I"
        ex, journal, backend = self.executor()
        receipt, _ = nb.collect_one_unit(self.context, ex, journal, self.unit, self.worker)
        self.assertEqual((backend.calls, receipt["logical_calls"]), (21, 63))
    def test_failed_partial_resume_retains_cost_and_reuses_success(self):
        ex, journal, backend = self.executor(FakeBackend(fail_at=10))
        with self.assertRaises(RuntimeError): nb.collect_one_unit(self.context, ex, journal, self.unit, self.worker)
        base = self.worker/"units"/self.unit["unit_id"]/"A_stay.json"
        pin = nb.file_sha(base)
        with self.assertRaises(nb.ProtocolError): nb.collect_one_unit(self.context, ex, journal, self.unit, self.worker)
        ex2, journal2, backend2 = self.executor()
        receipt, _ = nb.collect_one_unit(self.context, ex2, journal2, self.unit, self.worker, resume=True)
        self.assertEqual(nb.file_sha(base), pin); self.assertEqual(backend2.calls, 28)
        costs = nb.ledger_summary(self.worker)
        self.assertEqual((costs["attempted_calls"], costs["failed_wrapper_calls"]), (38, 1))
        self.assertIsNone(costs["unknown_failure_gpu_seconds"])
    def test_checkpoint_tamper_fails_before_inference(self):
        ex, journal, backend = self.executor(FakeBackend(fail_at=10))
        with self.assertRaises(RuntimeError): nb.collect_one_unit(self.context, ex, journal, self.unit, self.worker)
        path = self.worker/"units"/self.unit["unit_id"]/"A_stay.json"
        rec = nb.read_json(path); rec["execution"]["answer"] = "tampered"; nb.atomic_json(path, rec)
        ex2, j2, b2 = self.executor()
        with self.assertRaises(nb.ProtocolError): nb.collect_one_unit(self.context, ex2, j2, self.unit, self.worker, resume=True)
        self.assertEqual(b2.calls, 0)
    def test_failed_call_cap_includes_failures(self):
        ex, journal, backend = self.executor(FakeBackend(fail_at=1), cap=1)
        with self.assertRaises(RuntimeError): ex.execute(self.q, nb.default_menu()[0]["mask"], 3701)
        with self.assertRaises(nb.ProtocolError): ex.execute(self.q, nb.default_menu()[0]["mask"], 3701)
        self.assertEqual(backend.calls, 1)
    def test_qualification_full_replay_equal(self):
        menu = nb.default_menu()
        menu.append({"name": "independent", "mask": menu[-1]["mask"], "base": None, "replay_of": menu[-1]["name"]})
        self.plan["qualification_menu"] = menu; self.unit["phase"] = "qualification"; self.unit["seed"] = 3799
        ex, journal, backend = self.executor()
        receipt, _ = nb.collect_one_unit(self.context, ex, journal, self.unit, self.worker)
        self.assertEqual(backend.calls, 46); self.assertEqual(receipt["replay_checks"], [True])
    def test_exclusion_id_hash_alias_and_unknown_scope(self):
        excl = exclusion(); nb.validate_exclusions({self.q["id"]: self.q}, excl)
        for field, value in (("excluded_ids", self.q["id"]), ("excluded_normalized_question_sha256", nb.question_text_sha(self.q["question"])), ("excluded_normalized_question_sha256_prefix20", nb.question_text_sha(self.q["question"])[:20])):
            bad = copy.deepcopy(excl); bad[field] = [value]
            with self.assertRaises(nb.ProtocolError): nb.validate_exclusions({self.q["id"]: self.q}, bad)
        bad = copy.deepcopy(excl); bad["unknown_exposure_beyond_recorded"] = False
        with self.assertRaises(nb.ProtocolError): nb.validate_exclusions({self.q["id"]: self.q}, bad)
    def test_gold_payload_refused(self):
        with self.assertRaises(ValueError): M.Executor.question_payload({**self.q, "answer": "should never load"})
    def test_resident_fail_closed_and_additive(self):
        rows = [{"event": "START", "attempt_id": "a"}, {"event": "END", "attempt_id": "a", "resident_wall_seconds": 5.5}, {"event": "START", "attempt_id": "b"}, {"event": "END", "attempt_id": "b", "resident_wall_seconds": 3.5}]
        self.assertEqual(nb.resident_used(rows), 9)
        with self.assertRaises(nb.ProtocolError): nb.resident_used(rows[:-1])
        with self.assertRaises(nb.ProtocolError): nb.resident_used(rows+rows[:1])
    def test_worker_lease_prevents_second_owner(self):
        with nb.process_lease(self.root/"lock"):
            with self.assertRaises(nb.ProtocolError):
                with nb.process_lease(self.root/"lock"): pass
        with nb.process_lease(self.root/"lock"): pass
    def test_supervisor_budget_kills_only_own_mock_child(self):
        class Process:
            pid = 12345
            returncode = None
            terminated = False
            def poll(self): return self.returncode
            def terminate(self): self.terminated = True; self.returncode = -15
            def kill(self): self.returncode = -9
            def wait(self, timeout=None): return self.returncode
        child = Process()
        self.plan["workers"] = [{"physical_gpu": 2, "resident_wall_cap_seconds": 16}]
        args = SimpleNamespace(worker_index=0, phase="qualification", source_root=SOURCE, data_root=self.root,
                               plan=self.root/"plan.json", plan_sha256=self.context["plan_sha256"], resume=False)
        with patch.object(nb.subprocess, "Popen", return_value=child) as launch, patch.object(nb.time, "sleep"), patch.object(nb.time, "monotonic", side_effect=range(100)):
            with self.assertRaises(nb.ProtocolError): nb.supervise(args, self.context)
        self.assertTrue(child.terminated); self.assertEqual(launch.call_count, 1)
        rows = nb.read_jsonl(self.context["output"]/"worker_0"/"resident_wall.jsonl")
        self.assertEqual(rows[-1]["reason"], "BUDGET_STOP")
        self.assertTrue(rows[-1]["includes_model_load_and_failures"])
        self.assertLessEqual(nb.resident_used(rows), 16)
    def test_path_escape_and_immutable_output(self):
        for value in ("../else", "C:/else", "/absolute", "dir//file"):
            with self.assertRaises(nb.ProtocolError): nb.safe_path(self.root, value)
        nb.atomic_json(self.root/"one.json", {"a": 1}, immutable=True)
        with self.assertRaises(nb.ProtocolError): nb.atomic_json(self.root/"one.json", {"a": 2}, immutable=True)
        self.assertEqual(nb.read_json(self.root/"one.json"), {"a": 1})
    def test_allocation_preflight_synthetic_390_questions(self):
        qs = [question(i) for i in range(390)]; units = []; pos = 0
        for i in range(6):
            units.append({"unit_id": "qual_"+str(i), "question_id": qs[pos]["id"], "phase": "qualification", "seed": 3799, "worker_index": i%3}); pos += 1
        for e in episodes():
            for role, n in (("I", 12), ("C", 12), ("S", 24)):
                for i in range(n):
                    units.append({"unit_id": f"ep{e['episode']}_{role}_{i}", "question_id": qs[pos]["id"], "phase": role, "episode": e["episode"], "seed": e["seed"], "worker_index": pos%3}); pos += 1
        for q in qs: nb.append_json(self.root/"questions.jsonl", q)
        nb.atomic_json(self.root/"exclusions.json", exclusion())
        plan = make_plan(SOURCE, self.root, "synthetic", "questions.jsonl", "exclusions.json", "models/fake", FakeBackend.identity, episodes(), units)
        plan["status"] = "FROZEN"; nb.atomic_json(self.root/"plan.json", plan)
        ctx = nb.preflight(SOURCE, self.root, self.root/"plan.json", nb.file_sha(self.root/"plan.json"))
        self.assertEqual(len(ctx["questions"]), 390)
        self.assertEqual(sum(nb.estimated_calls(nb.unit_menu(plan, u), M) for u in units), 12948)
        plan["units"][-1]["question_id"] = plan["units"][-2]["question_id"]
        nb.atomic_json(self.root/"plan.json", plan)
        with self.assertRaises(nb.ProtocolError): nb.preflight(SOURCE, self.root, self.root/"plan.json", nb.file_sha(self.root/"plan.json"))


if __name__ == "__main__": unittest.main(verbosity=2)
