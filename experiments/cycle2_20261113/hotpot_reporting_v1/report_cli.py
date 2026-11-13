"""No-input default; synthetic fixture or explicitly released evaluation only."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import reporting as r


def read_release_metadata(evaluation, receipt, review, receipt_pin, review_pin):
    """Reject mistaken path aliases before any content read. Identity-only stat."""
    r.require(r.is_hash(receipt_pin) and r.is_hash(review_pin), 'Two explicit metadata pins required')
    for left, right in ((evaluation, receipt), (evaluation, review), (receipt, review)):
        r.require(not os.path.samefile(left, right), 'Evaluation and metadata files must be distinct')
    return receipt.read_bytes(), review.read_bytes()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run-released', action='store_true')
    mode.add_argument('--synthetic', action='store_true')
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--evaluation-receipt', type=Path)
    parser.add_argument('--receipt-sha256')
    parser.add_argument('--score-review', type=Path)
    parser.add_argument('--score-review-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.run_released and not args.synthetic:
        print(json.dumps({'status': 'PREFLIGHT_NO_EVALUATION_OR_LABEL_IO', 'new_model_calls': 0}))
        return
    r.require(args.output is not None and not args.output.exists(), 'Fresh output directory required')
    if args.synthetic:
        r.require(all(x is None for x in (args.evaluation, args.evaluation_receipt, args.receipt_sha256,
            args.score_review, args.score_review_sha256)), 'Synthetic mode cannot accept real-input paths or receipts')
        r.require(any('synthetic' in part.lower() for part in args.output.parts), 'Explicit synthetic output path required')
        from synthetic_fixture import make_fixture
        evaluation = make_fixture()
        private = {'source': 'IN_MEMORY_FABRICATED_FIXTURE_ONLY', 'real_data_read': False}
    else:
        r.require(all(x is not None for x in (args.evaluation, args.evaluation_receipt, args.receipt_sha256,
            args.score_review, args.score_review_sha256)), 'Two externally trusted pins and metadata inputs required')
        # Only filesystem identity is inspected to reject metadata aliases.
        # Evaluation CONTENT is not opened or hashed before the release gate.
        receipt_bytes, review_bytes = read_release_metadata(args.evaluation, args.evaluation_receipt,
            args.score_review, args.receipt_sha256, args.score_review_sha256)
        evaluation, private = r.load_released_evaluation(receipt_bytes=receipt_bytes,
            expected_receipt_sha256=args.receipt_sha256, score_review_bytes=review_bytes,
            expected_score_review_sha256=args.score_review_sha256, evaluation_reader=args.evaluation.read_bytes)
    summary = r.aggregate_evaluation(evaluation, synthetic=args.synthetic)
    receipt = r.write_reports(summary, args.output, private)
    print(json.dumps({'status': receipt['status'], 'public_aggregate_files': len(receipt['public_whitelist_sha256']),
        'full_physical_cost_coverage': False, 'new_statistics_run': False, 'publication_performed': False}))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(json.dumps({'status': 'BLOCKED_REPORTING', 'error_type': type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
