"""Independent CPU boundary probes; no model imports and no production edits."""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from test_independent12_executor import make
from independent12_executor import Independent12ExecutionError


def main():
    question = {"id": "synthetic_external_edge_probe", "question": "Return the synthetic marker.",
                "context": "This is a synthetic CPU-only boundary fixture."}
    checks = []
    executor, router, state = make()
    with patch("independent12_executor.normalize_message", side_effect=RuntimeError("synthetic normalization failure")):
        try:
            executor.execute(question, 99191)
        except Independent12ExecutionError as error:
            result = error.partial_result
        else:
            raise AssertionError("Injected normalizer error did not abort")
    assert len(result.slots) == 13 and len(state["calls"]) == 1
    assert result.slots[0].status == "POSTPROCESS_ERROR"
    assert result.slots[0].generation is not None and result.slots[0].record is None
    assert all(s.status == "NOT_ATTEMPTED" for s in result.slots[1:])
    assert result.cost()["calls"] == result.cost()["returned_generations"] == 1
    assert result.cost()["output_tokens"] == result.slots[0].generation.output_tokens
    assert result.slots[0].route_events[0]["status"] == "SUCCESS"
    assert result.slots[0].physical_events[0]["status"] == "SUCCESS"
    assert not router._execution_lock.locked() and router._selected.get() is None
    checks.append({"name": "normalizer_exception_preserves_returned_generation_cost_and_13_slots", "pass": True,
                   "attempted_requests": 1, "retained_generations": 1, "unattempted_slots": 12})

    executor, router, state = make()
    original_encode = router.backends["mistral"].encode
    def encoding_failure(text):
        if router._selected.get() == "aggregate":
            raise RuntimeError("synthetic aggregator receiver-tokenizer failure")
        return original_encode(text)
    router.backends["mistral"].encode = encoding_failure
    try:
        executor.execute(question, 99191)
    except Independent12ExecutionError as error:
        result = error.partial_result
    else:
        raise AssertionError("Injected aggregate preparation error did not abort")
    assert len(result.slots) == 13 and len(state["calls"]) == 12
    assert all(s.record is not None and s.generation is not None for s in result.slots[:12])
    assert result.slots[12].status == "PREPARATION_ERROR"
    assert not result.slots[12].attempted_backend_request
    assert result.slots[12].generation is None and result.answer is None
    assert result.cost()["calls"] == result.cost()["returned_generations"] == 12
    assert result.cost()["unattempted_calls"] == 1 and not result.cost()["service_complete"]
    assert len(router.route_events) == 12
    assert not router._execution_lock.locked() and router._selected.get() is None
    checks.append({"name": "aggregator_receiver_preparation_failure_retains_all_12_candidates_without_extra_request", "pass": True,
                   "attempted_requests": 12, "retained_generations": 12, "unattempted_slots": 1})
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {"status": "PASS_INDEPENDENT_CPU_BOUNDARY_PROBES", "checks": checks,
               "reviewed_source_sha256": sha(ROOT / "independent12_executor.py"),
               "reviewed_tests_sha256": sha(ROOT / "test_independent12_executor.py"),
               "probe_script_sha256": sha(Path(__file__)),
               "new_GPU_calls": 0, "model_loads": 0, "production_sources_modified": False,
               "scope": "Synthetic mock failures only; no real-model engineering qualification or scientific admission"}
    (ROOT / "INDEPENDENT_EDGE_PROBE.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
