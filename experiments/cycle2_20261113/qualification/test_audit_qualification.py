"""Entirely synthetic, CPU-only qualification receipts and corruption tests."""
from pathlib import Path
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import new_batch as nb
from test_new_batch import FakeBackend, M, SOURCE, episodes, question
from audit_qualification import audit, forecast_progress, raw_format, original_execution_cost, producer_float_sum


class LoggedFake(FakeBackend):
    def __init__(self, path): super().__init__(); self.path = path
    def generate(self, messages, seed, max_new_tokens, config):
        started = time.time()
        g = super().generate(messages, seed, max_new_tokens, config)
        nb.append_json(self.path, {"status": "SUCCESS", "call_id": g.call_id, "started_unix": started,
            "seed": seed, "prompt_sha256": nb.digest(messages), "input_tokens": g.input_tokens,
            "output_tokens": g.output_tokens, "gpu_seconds": g.gpu_seconds,
            "gpu_peak_allocated_bytes": g.gpu_peak_allocated_bytes, "finish_reason": g.finish_reason,
            "elapsed_seconds": g.elapsed_seconds,
            "generation_config": {"do_sample": config.do_sample, "temperature": config.temperature,
                                  "top_p": config.top_p, "max_new_tokens": max_new_tokens}})
        return g


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        testroot = HERE.parent.parent/"qtest"; testroot.mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=testroot); cls.root = Path(cls.temp.name)
        cls.output = cls.root/"outputs"; cls.runner = Path(nb.__file__)
        q = [question(1000+i) for i in range(6)]
        eps = [{**e, "menu": nb.episode_menu(e["permutation"], e["complement"])} for e in episodes()]
        units = [{"unit_id": "qual_"+str(i), "question_id": row["id"], "phase": "qualification", "seed": 3799, "worker_index": i%3} for i, row in enumerate(q)]
        for i in range(3):
            units.append({"unit_id": "formal_"+str(i), "question_id": "not_read_"+str(i), "phase": "S", "seed": 3701, "episode": 0, "worker_index": i})
        menu = nb.default_menu(); menu.append({"name": "replay", "mask": menu[-1]["mask"], "base": None, "replay_of": "B_toggle_9"})
        import dataclasses
        cfg = M.GraphConfig(); model = FakeBackend.identity
        cls.plan = {"schema": nb.SCHEMA, "status": "FROZEN", "labels_read": False, "runner_sha256": nb.file_sha(cls.runner),
            "source_manifest": {p: nb.file_sha(SOURCE/p) for p in nb.REQUIRED_SOURCES}, "graph": dataclasses.asdict(cfg),
            "model": {"identity": model}, "executor_protocol_hash": M.Executor(FakeBackend()).protocol_hash,
            "episodes": eps, "units": units, "qualification_menu": menu, "max_unit_attempts": 2,
            "questions": {"payload_sha256": {r["id"]: M.digest(M.Executor.question_payload(r)) for r in q}},
            "workers": [{"index": i, "resident_wall_cap_seconds": 28800, "max_call_attempts": 6000} for i in range(3)],
            "budget": {"resident_wall_gpu_seconds": 86400, "max_call_attempts": 18000}}
        cls.planpath = cls.root/"plan.json"; nb.atomic_json(cls.planpath, cls.plan); cls.pin = nb.file_sha(cls.planpath)
        ctx = {"plan": cls.plan, "plan_sha256": cls.pin, "module": M, "questions": {r["id"]: r for r in q}, "output": cls.output}
        for w in range(3):
            folder = cls.output/("worker_"+str(w)); folder.mkdir(parents=True)
            native = LoggedFake(folder/"physical_calls.jsonl"); journal = nb.JournalBackend(native, folder/"call_journal.jsonl", 6000)
            ex = M.Executor(journal)
            nb.append_json(folder/"resident_wall.jsonl", {"event": "START", "attempt_id": "a", "phase": "qualification", "plan_sha256": cls.pin, "started_unix": time.time()})
            nb.atomic_json(folder/"RUNTIME_synthetic.json", {"model_identity": model, "plan_sha256": cls.pin, "labels_read": False})
            complete = []
            for unit in units:
                if unit["worker_index"] != w or unit["phase"] != "qualification": continue
                nb.collect_one_unit(ctx, ex, journal, unit, folder)
                complete.append({"unit_id": unit["unit_id"], "sha256": nb.file_sha(folder/"units"/unit["unit_id"]/"UNIT_COMPLETE.json")})
            nb.atomic_json(folder/"QUALIFICATION_COMPLETE.json", {"status": "COMPLETE_UNSCORED", "phase": "qualification",
                "worker_index": w, "plan_sha256": cls.pin, "executor_protocol_hash": ex.protocol_hash,
                "labels_read": False, "completed_units": complete, "ledger_summary": nb.ledger_summary(folder)})
            nb.append_json(folder/"resident_wall.jsonl", {"event": "END", "attempt_id": "a", "resident_wall_seconds": 10,
                "plan_sha256": cls.pin, "resident_cap_exceeded": False})
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def run_audit(self, path=None, pin=None):
        return audit(path or self.planpath, pin or self.pin, self.output, SOURCE, self.runner)
    def test_complete_no_auto_acceptance_and_readonly(self):
        before = {str(p.relative_to(self.root)): nb.file_sha(p) for p in self.root.rglob("*") if p.is_file()}
        result = self.run_audit()
        self.assertFalse(result["automatic_acceptance"]); self.assertFalse(result["question_file_read"])
        self.assertEqual(result["committed_physical_calls"], 276); self.assertEqual(result["logical_calls"], 810)
        self.assertEqual(result["format_counts"]["physical_outputs"], 276)
        self.assertEqual(result["format_counts"]["malformed"], 0)
        self.assertEqual(result["findings_for_manual_review"], [])
        self.assertEqual(before, {str(p.relative_to(self.root)): nb.file_sha(p) for p in self.root.rglob("*") if p.is_file()})
    def test_episode_swap_rejected_without_changing_frozen_runner(self):
        import copy
        p = copy.deepcopy(self.plan)
        for field in ("permutation", "complement", "menu"):
            p["episodes"][0][field], p["episodes"][2][field] = p["episodes"][2][field], p["episodes"][0][field]
        path = self.root/"bad_plan.json"; nb.atomic_json(path, p)
        with self.assertRaises(RuntimeError): self.run_audit(path, nb.file_sha(path))
    def test_pinned_plan_rejects_wrong_sha(self):
        with self.assertRaises(RuntimeError): self.run_audit(pin="0"*64)
    def test_native_cost_tamper_detected(self):
        path = self.output/"worker_0"/"physical_calls.jsonl"; raw = path.read_bytes()
        try:
            rows = nb.read_jsonl(path); rows[0]["gpu_seconds"] += 1
            path.write_text("".join(nb.canonical(r)+"\n" for r in rows), encoding="utf-8")
            with self.assertRaises(RuntimeError): self.run_audit()
        finally: path.write_bytes(raw)
    def test_timing_forecast_has_no_scientific_decision(self):
        result = forecast_progress(self.planpath, self.pin, self.output, SOURCE, self.runner)
        self.assertFalse(result["scientific_failure_inferred"])
        self.assertEqual(result["planned_formal_calls"], 111)
        self.assertEqual(len(result["workers"]), 3)
        self.assertTrue(all(w["including_load_resident_calls_per_second"] == 9.2 for w in result["workers"]))
    def test_independent_raw_format_cases(self):
        self.assertTrue(raw_format('{"answer":2,"evidence":"","uncertainty":""}'))
        self.assertFalse(raw_format('{"answer":true,"evidence":"","uncertainty":""}'))
        self.assertFalse(raw_format('{"answer":"2","answer":"3","evidence":"","uncertainty":""}'))
        self.assertFalse(raw_format('{"answer":"2"'))
    def test_json_node_order_and_python311_arithmetic_exact(self):
        # Positive timing numbers: changing JSON key order changes old float sum.
        from types import SimpleNamespace
        order = [f'r{r}:w{w}' for r in (1,2) for w in range(4)]+['aggregate']
        values = [0.1]*8+[0.9]
        nodes = {k: SimpleNamespace(reused=False, generation=SimpleNamespace(input_tokens=1, output_tokens=1,
                 elapsed_seconds=t, gpu_seconds=t, gpu_peak_allocated_bytes=1), received_message_tokens=0, self_state_tokens=0)
                 for k,t in zip(order,values)}
        restored = SimpleNamespace(nodes={k:nodes[k] for k in sorted(nodes)})
        original = producer_float_sum(values, '3.11.9')
        sorted_sum = producer_float_sum([nodes[k].generation.gpu_seconds for k in sorted(nodes)], '3.11.9')
        self.assertNotEqual(original, sorted_sum)
        cost = original_execution_cost(restored, False, '3.11.9')
        self.assertEqual(cost['gpu_seconds'], original)
        self.assertEqual(cost['calls'], 9)
        if sys.version_info >= (3, 12):
            self.assertNotEqual(producer_float_sum([1e16,1.,1.], '3.11.9'), producer_float_sum([1e16,1.,1.], '3.13.3'))
        else:
            with self.assertRaises(RuntimeError):
                producer_float_sum([1e16,1.,1.], '3.13.3')


if __name__ == "__main__": unittest.main(verbosity=2)
