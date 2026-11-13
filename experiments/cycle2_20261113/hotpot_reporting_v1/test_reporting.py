"""Small reporting tests using fabricated schema values only, not statistics tests."""
import copy
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import reporting as r
from synthetic_fixture import make_fixture

ROOT = Path(__file__).resolve().parent


def released_fixture_arguments():
    result = make_fixture(); result.pop('synthetic_fixture')
    review = {'schema': 'hotpot_closed_archive_score_release_v1', 'decision': 'APPROVED_FOR_SCORING',
        'scientific_protocol_status': 'FROZEN', 'unique_questions': 384, 'terminal_conditions': 4224,
        'runtime_failure_count': 0, 'all_attempts_closed': True, 'physical_ledger_closed': True,
        'bindings': result['bindings'], 'archive_metadata_sha256': result['physical_research_ledger']['archive_metadata_sha256'],
        'test_only': 'ARTIFICIAL_RELEASE_NOT_REAL_AUTHORIZATION'}
    review_raw = r.encode(review); review_pin = r.sha(review_raw)
    result['archive_review_sha256'] = review_pin
    evaluation_raw = r.encode(result)
    receipt = {'status': 'COMPLETE_PRIVATE_FROZEN_EVALUATION', 'evaluation_adapter_sha256': r.FROZEN_PINS['evaluation_adapter_sha256'],
        'postclosure_source_sha256': r.POSTCLOSURE_PIN, 'evaluation_sha256': r.sha(evaluation_raw),
        'score_review_sha256': review_pin, 'new_model_calls': 0, 'scientific_or_GPU_admission_granted': False,
        'publication_authorized': False}
    receipt_raw = r.encode(receipt)
    return dict(receipt_bytes=receipt_raw, expected_receipt_sha256=r.sha(receipt_raw),
        score_review_bytes=review_raw, expected_score_review_sha256=review_pin), evaluation_raw


class ReportingTests(unittest.TestCase):
    def test_metadata_alias_refused_before_any_content_read(self):
        import report_cli
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / 'artificial_evaluation.json'
            review = root / 'artificial_review.json'
            evaluation.write_text('SYNTHETIC_ONLY', encoding='utf-8')
            review.write_text('SYNTHETIC_ONLY', encoding='utf-8')
            with mock.patch.object(Path, 'read_bytes') as reader:
                with self.assertRaises(r.ReportError):
                    report_cli.read_release_metadata(evaluation, evaluation, review, 'a' * 64, 'b' * 64)
                reader.assert_not_called()
            with mock.patch.object(report_cli.os.path, 'samefile', return_value=True), mock.patch.object(Path, 'read_bytes') as reader:
                with self.assertRaises(r.ReportError):
                    report_cli.read_release_metadata(evaluation, root / 'alias.json', review, 'a' * 64, 'b' * 64)
                reader.assert_not_called()

    def test_receipt_and_release_guards_precede_evaluation_reader(self):
        for change in ('missing_receipt_pin', 'wrong_receipt_pin', 'wrong_review_pin', 'wrong_status', 'unapproved', 'unclosed'):
            with self.subTest(change=change):
                args, raw = released_fixture_arguments()
                if change == 'missing_receipt_pin': args['expected_receipt_sha256'] = None
                if change == 'wrong_receipt_pin': args['expected_receipt_sha256'] = '0' * 64
                if change == 'wrong_review_pin': args['expected_score_review_sha256'] = '0' * 64
                if change in ('wrong_status', 'unapproved', 'unclosed'):
                    receipt = r.decode(args['receipt_bytes']); review = r.decode(args['score_review_bytes'])
                    if change == 'wrong_status': receipt['status'] = 'REVIEW_REQUIRED'
                    if change == 'unapproved': review['decision'] = 'REVIEW_REQUIRED'
                    if change == 'unclosed': review['all_attempts_closed'] = False
                    args['score_review_bytes'] = r.encode(review)
                    args['expected_score_review_sha256'] = r.sha(args['score_review_bytes'])
                    receipt['score_review_sha256'] = args['expected_score_review_sha256']
                    args['receipt_bytes'] = r.encode(receipt); args['expected_receipt_sha256'] = r.sha(args['receipt_bytes'])
                reader = mock.Mock(return_value=raw)
                with self.assertRaises(r.ReportError): r.load_released_evaluation(**args, evaluation_reader=reader)
                reader.assert_not_called()

    def test_released_bytes_and_physical_pins_remain_private(self):
        args, raw = released_fixture_arguments(); reader = mock.Mock(return_value=raw)
        result, private = r.load_released_evaluation(**args, evaluation_reader=reader)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(private['physical_archive_metadata_sha256'], result['physical_research_ledger']['archive_metadata_sha256'])
        summary = r.aggregate_evaluation(result)
        public = r.encode(summary).decode()
        self.assertIsNone(re.search(r'\b[0-9a-fA-F]{64}\b', public))
        self.assertNotIn('hotpotqa_', public)
        with self.assertRaises(r.ReportError): r.load_released_evaluation(**args, evaluation_reader=lambda: raw + b' ')

    def test_all_conditions_means_rejections_and_missing_diagnostics(self):
        summary = r.aggregate_evaluation(make_fixture(), synthetic=True)
        rows = {row['condition']: row for row in summary['conditions']}
        self.assertEqual(tuple(rows), r.CONDITIONS)
        self.assertEqual(rows['S1/SINGLE_M']['em_success_count'], 191)
        self.assertAlmostEqual(rows['S1/SINGLE_M']['em'], 191 / 384)
        self.assertAlmostEqual(rows['S1/NO_EDGE']['em'], 1 / 3)
        self.assertEqual(sum(row['input_rejections'] for row in rows.values()), 2)
        self.assertTrue(all(row['n'] == 384 and row['format_failures'] is None and row['finish_reasons'] is None for row in rows.values()))
        self.assertEqual(summary['global_format_failure_conditions'], 5)
        self.assertTrue(all(row['value'] is None for row in summary['physical_costs']))
        self.assertTrue(all(row['calls'] is None for row in summary['logical_costs'] if row['condition'] == 'S2/B'))

    def test_primary_statistics_are_copied_and_missing_stays_unavailable(self):
        fixture = make_fixture()
        key = r.PRIMARY[0][2]
        fixture['statistics']['primary_EM'][key].update(estimate=.123456, interval=[.111, .222], support_guard_pass=False)
        summary = r.aggregate_evaluation(fixture, synthetic=True)
        row = summary['primary'][0]
        self.assertEqual((row['estimate'], row['lower'], row['upper'], row['support_guard_pass']), (.123456, .111, .222, False))
        fixture.pop('statistics')
        summary = r.aggregate_evaluation(fixture, synthetic=True)
        self.assertTrue(all(row['estimate'] is None and row['lower'] is None for row in summary['primary']))
        self.assertIsNone(summary['gates']['L1'])
        self.assertTrue(summary['claims']['new_statistical_tests'] is False)

    def test_signed_distributions_preserve_negative_products_and_full_denominators(self):
        summary = r.aggregate_evaluation(make_fixture(), synthetic=True)
        for row in summary['signed']:
            self.assertEqual(row['negative'] + row['zero'] + row['positive'], 384)
        product = next(row for row in summary['signed'] if row['metric'] == 'EM' and row['quantity'] == 'delta_product')
        self.assertGreaterEqual(product['negative'], 48)
        groups = {}
        for row in summary['em_hist']:
            groups.setdefault((row['quantity'], row['seed']), 0)
            groups[(row['quantity'], row['seed'])] += row['count']
        self.assertEqual(set(groups.values()), {384})
        self.assertTrue(any(row['value'] < 0 and row['count'] > 0 for row in summary['em_hist'] if row['quantity'] == 'delta_product'))

    def test_incomplete_or_duplicate_conditions_refused(self):
        for mode in ('missing', 'duplicate', 'bad_rejection_zero'):
            fixture = make_fixture()
            if mode == 'missing': fixture['condition_statuses'].pop()
            if mode == 'duplicate': fixture['condition_statuses'][-1] = copy.deepcopy(fixture['condition_statuses'][0])
            if mode == 'bad_rejection_zero': fixture['em']['single'][0] = 1
            with self.assertRaises(r.ReportError): r.aggregate_evaluation(fixture, synthetic=True)

    def test_default_and_synthetic_cli_cannot_open_real_input(self):
        default = subprocess.run([sys.executable, '-B', str(ROOT / 'report_cli.py'), '--evaluation', 'missing_private.json'],
            capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(default.stdout)['status'], 'PREFLIGHT_NO_EVALUATION_OR_LABEL_IO')
        with tempfile.TemporaryDirectory() as temp:
            invalid = subprocess.run([sys.executable, '-B', str(ROOT / 'report_cli.py'), '--synthetic',
                '--evaluation', 'missing_private.json', '--output', str(Path(temp) / 'synthetic_rejected')], capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2)
            self.assertFalse((Path(temp) / 'synthetic_rejected').exists())

    def test_end_to_end_synthetic_files_privacy_and_no_overwrite(self):
        # One small integration render; does not call any frozen statistics code.
        from pypdf import PdfReader
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'synthetic_report'
            command = [sys.executable, '-B', str(ROOT / 'report_cli.py'), '--synthetic', '--output', str(output)]
            run = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(run.stdout)['status'], 'SYNTHETIC_REPORT_ONLY')
            public = output / 'public'; files = list(public.iterdir())
            self.assertEqual(len(files), 16)
            self.assertEqual(len(list(public.glob('*.png'))), 4)
            self.assertEqual(len(list(public.glob('*.pdf'))), 4)
            for path in files:
                if path.suffix == '.pdf': text = '\n'.join(page.extract_text() for page in PdfReader(path).pages)
                elif path.suffix == '.png':
                    with Image.open(path) as image:
                        image.verify()
                    continue
                else: text = path.read_text(encoding='utf-8')
                for sentinel in ('PRIVATE_SYNTHETIC_SENTINEL', 'PRIVATE_PROMPT_SENTINEL', 'PRIVATE_ANSWER_SENTINEL', 'hotpotqa_'):
                    self.assertNotIn(sentinel, text)
                self.assertIsNone(re.search(r'\b[0-9a-fA-F]{64}\b', text))
                self.assertIn('SYNTHETIC', text)
            with (public / 'conditions.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 11)
            self.assertTrue(all(row['format_failures'] == 'NA' and row['finish_reasons'] == 'NA' for row in rows))
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(rerun.returncode, 2)


if __name__ == '__main__': unittest.main()
