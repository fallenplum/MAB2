"""Run only synthetic unittest cases and retain sanitized validation artifacts."""
import hashlib
import io
import json
import platform
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import test_independent12_executor as tests
import test_cost_reconciliation as cost_tests


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def main():
    log = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordingResult).run(
        unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),
                            unittest.defaultTestLoader.loadTestsFromModule(cost_tests)]))
    elapsed = time.perf_counter() - started
    model_runtime_loaded = any(name in sys.modules for name in ("torch", "transformers"))
    receipt = {"status": "PASS_CPU_MOCK_CONTRACT_ONLY" if result.wasSuccessful() and not model_runtime_loaded else "FAIL",
               "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "passed_test_ids": result.passed_ids, "elapsed_seconds": elapsed,
               "python_version": platform.python_version(), "torch_or_transformers_imported": model_runtime_loaded,
               "control_sha256": hashlib.sha256((ROOT / "independent12_executor.py").read_bytes()).hexdigest(),
               "tests_sha256": hashlib.sha256((ROOT / "test_independent12_executor.py").read_bytes()).hexdigest(),
               "cost_tests_sha256": hashlib.sha256((ROOT / "test_cost_reconciliation.py").read_bytes()).hexdigest(),
               "test_counts": {"original_contract": 14, "new_cost_reconciliation": 9},
               "new_GPU_calls": 0, "real_model_calls": 0, "benchmark_or_R1_result_reads": 0,
               "scientific_admission": False, "real_model_engineering_qualification": "NOT_RUN_REQUIRED_SEPARATELY"}
    (ROOT / "TEST_RESULTS.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (ROOT / "CPU_TESTS.log").write_text(log.getvalue(), encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("status", "tests_run", "failures", "errors", "torch_or_transformers_imported")}))
    return 0 if receipt["status"] == "PASS_CPU_MOCK_CONTRACT_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
