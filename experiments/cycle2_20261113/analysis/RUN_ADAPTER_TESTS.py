"""Full artificial-archive tests plus a separate independent replay CLI."""
import time
START_WALL = time.perf_counter()
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-script', type=Path, required=True)
    parser.add_argument('--r2-script', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep earlier receipts; supply a new output')
    if not args.replay_script.is_file():
        raise FileNotFoundError('Independent replay script required')
    if not args.r2_script.is_file():
        raise FileNotFoundError('R2 I/C-only diagnostic script required')
    os.environ['MAB2_R1_REPLAY_SCRIPT'] = str(args.replay_script.resolve())
    os.environ['MAB2_R2_DIAGNOSTIC_SCRIPT'] = str(args.r2_script.resolve())
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(['test_r1_policy', 'test_r1_adapter'])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    here = Path(__file__).resolve().parent
    names = ['r1_adapter.py', 'r1_policy.py', 'test_r1_policy.py', 'test_r1_adapter.py',
             'synthetic_collector_fixture.py', 'RUN_ADAPTER_TESTS.py', 'test_support/original_scoring.py']
    receipt = {'schema': 'mab2_r1_adapter_synthetic_receipt_v1',
        'status': 'PASS_SOFTWARE_ONLY' if result.wasSuccessful() else 'FAIL',
        'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skips': len(result.skipped),
        'new_real_questions_or_labels_read': False, 'new_GPU_calls': 0,
        'main_process_CPU_seconds': time.process_time(), 'wall_seconds_from_script_entry': time.perf_counter() - START_WALL,
        'independent_subprocess_CPU_seconds': None,
        'independent_replay_source_sha256': hashlib.sha256(args.replay_script.read_bytes()).hexdigest(),
        'R2_diagnostic_source_sha256': hashlib.sha256(args.r2_script.read_bytes()).hexdigest(),
        'source_sha256': {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in names},
        'test_log': stream.getvalue()}
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('status', 'tests', 'failures', 'errors', 'skips', 'main_process_CPU_seconds')}))
    if not result.wasSuccessful():
        print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
