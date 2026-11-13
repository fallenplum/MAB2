"""Write a fresh CPU-only test receipt; uses temporary artificial fixtures only."""
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
    return {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(ROOT.rglob('*.py'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = pins(); start = time.monotonic()
    result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'test_postclosure', '-v'],
                            cwd=ROOT, capture_output=True)
    elapsed = time.monotonic() - start
    log = result.stdout + result.stderr
    (args.output / 'CPU_TESTS.log').write_bytes(log)
    unchanged = pins() == before
    receipt = {'status': 'PASS_ARTIFICIAL_CPU_PREPARATION' if result.returncode == 0 and unchanged else 'FAILED_PREPARATION',
        'returncode': result.returncode, 'elapsed_seconds': elapsed, 'source_unchanged': unchanged,
        'source_sha256': before, 'test_log_sha256': hashlib.sha256(log).hexdigest(),
        'real_dataset_or_archive_accessed': False, 'real_labels_read': False,
        'new_model_calls': 0, 'GPU_hours': 0, 'label_extraction_authorized': False,
        'scoring_authorized': False, 'publication_authorized': False}
    raw = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    (args.output / 'RECEIPT.json').write_bytes(raw)
    print(json.dumps({'status': receipt['status'], 'elapsed_seconds': elapsed,
                      'receipt_sha256': hashlib.sha256(raw).hexdigest()}))
    if receipt['status'] != 'PASS_ARTIFICIAL_CPU_PREPARATION': raise SystemExit(2)


if __name__ == '__main__': main()
