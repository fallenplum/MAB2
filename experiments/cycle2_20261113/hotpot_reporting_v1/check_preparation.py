"""CPU test receipt for artificial reporting fixtures; no benchmark IO."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def pins():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    before = pins(); started = time.monotonic()
    result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'test_reporting', '-v'], cwd=ROOT, capture_output=True)
    log = result.stdout + result.stderr; (args.output / 'TESTS.log').write_bytes(log)
    receipt = {'status': 'PASS_SYNTHETIC_REPORTING_TESTS' if result.returncode == 0 and pins() == before else 'FAILED',
        'returncode': result.returncode, 'elapsed_seconds': time.monotonic() - started, 'source_unchanged': pins() == before,
        'source_sha256': before, 'test_log_sha256': hashlib.sha256(log).hexdigest(), 'real_data_read': False,
        'new_frozen_statistics_runs': 0, 'new_model_calls': 0, 'GPU_hours': 0,
        'scoring_or_publication_admission': False}
    raw = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    (args.output / 'RECEIPT.json').write_bytes(raw)
    print(json.dumps({'status': receipt['status'], 'receipt_sha256': hashlib.sha256(raw).hexdigest()}))
    if receipt['status'] == 'FAILED': raise SystemExit(2)


if __name__ == '__main__': main()
