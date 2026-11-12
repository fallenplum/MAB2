"""Record a local synthetic test attempt, refusing to overwrite receipts."""
import time
WALL_START = time.perf_counter()
import argparse
import hashlib
import io
import json
from pathlib import Path
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Keep prior test receipt; choose a new output name')
    stream = io.StringIO()
    tests = unittest.defaultTestLoader.loadTestsFromName('test_r1_policy')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    root = Path(__file__).resolve().parent
    source = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ('r1_policy.py', 'test_r1_policy.py', 'RUN_SYNTHETIC_TESTS.py')}
    receipt = {'schema': 'mab2_r1_synthetic_test_receipt_v1',
               'status': 'PASS_SOFTWARE_ONLY' if result.wasSuccessful() else 'FAIL',
               'tests': result.testsRun, 'failures': len(result.failures),
               'errors': len(result.errors), 'skips': len(result.skipped),
               'new_real_data_read': False, 'new_GPU_calls': 0,
               'main_process_cpu_seconds': time.process_time(),
               'wall_seconds_from_script_entry': time.perf_counter() - WALL_START,
               'other_agent_and_prior_attempt_CPU_seconds': None,
               'source_sha256': source, 'test_log': stream.getvalue()}
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('status', 'tests', 'failures', 'errors', 'skips', 'main_process_cpu_seconds')}))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
