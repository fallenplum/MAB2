"""Independent CPU-only fault boundary review of the frozen working snapshot."""
import argparse
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(source, output):
    source, output = Path(source), Path(output)
    assert not output.exists()
    manifest_path = source/'WORKING_MANIFEST.json'
    manifest_pin = sha(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    for name, pin in manifest['files'].items(): assert sha(source/name) == pin, name
    helper = source.parent/'durable_backend_stage'
    sys.path[:0] = [str(source), str(helper)]
    import test_qualification_preparation as qualification_tests
    import test_durable_backend as durable_tests
    from durable_backend import DurableBackend, DurableBackendError, read_journal, extract_durable_error_evidence
    from qualify_hotpot_matrix import RequestCap

    class ExtraBoundaries(unittest.TestCase):
        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.path = Path(self.tmp.name)
        def tearDown(self): self.tmp.cleanup()

        def test_postwrite_report_failure_is_once_and_raw_survives(self):
            native = durable_tests.Native()
            wrapped = DurableBackend(native, 'qwen', self.path/'raw.jsonl')
            original = wrapped._append
            def after_write(event):
                original(event)
                if event['event'] == 'RETURNED': raise OSError('synthetic failure after fsync')
            with patch.object(wrapped, '_append', side_effect=after_write):
                with self.assertRaises(DurableBackendError) as caught:
                    wrapped.generate([], 17, 128, durable_tests.Config())
            journal = read_journal(self.path/'raw.jsonl')
            self.assertEqual(journal['pending_request_ids'], [])
            self.assertEqual(len(journal['requests']), 1)
            terminal = next(iter(journal['requests'].values()))['terminal']
            self.assertEqual(terminal['generation'], dataclasses.asdict(native.result))
            self.assertEqual(len(native.physical_calls), 1)
            with self.assertRaises(DurableBackendError): wrapped.generate([], 17, 128, durable_tests.Config())
            self.assertEqual(native.calls, 1)
            self.assertFalse(extract_durable_error_evidence(caught.exception)[0]['new_physical_cost_event'])

        def test_global_cap_across_models_rejects_without_second_native_call(self):
            cap = RequestCap(1)
            first, second = durable_tests.Native(), durable_tests.Native()
            a = DurableBackend(first, 'qwen', self.path/'q.jsonl', before_request=cap)
            b = DurableBackend(second, 'mistral', self.path/'m.jsonl', before_request=cap)
            a.generate([], 17, 128, durable_tests.Config())
            with self.assertRaises(DurableBackendError): b.generate([], 18, 128, durable_tests.Config())
            self.assertEqual((first.calls, second.calls, cap.admitted), (1, 0, 1))
            self.assertEqual(len(first.physical_calls)+len(second.physical_calls), 1)

        def test_admission_callback_cannot_mutate_recorded_or_caller_context_by_alias(self):
            context = {'unit': 5, 'nested': {'value': 'original'}}
            def mutate(value): value['nested']['value'] = 'callback copy only'
            native = durable_tests.Native()
            wrapped = DurableBackend(native, 'qwen', self.path/'raw.jsonl',
                context_provider=lambda: context, before_request=mutate)
            wrapped.generate([], 17, 128, durable_tests.Config())
            start = next(iter(read_journal(self.path/'raw.jsonl')['requests'].values()))['start']
            self.assertEqual(start['context']['nested']['value'], 'original')
            self.assertEqual(context['nested']['value'], 'original')

        def test_callback_overflow_text_cannot_be_misclassified_by_matrix(self):
            from matrix_executor import classify_failure
            native = durable_tests.Native()
            def callback(_): raise ValueError('Prompt has 8193 tokens; frozen cap is 8192')
            wrapped = DurableBackend(native, 'qwen', self.path/'raw.jsonl', before_request=callback)
            with self.assertRaises(DurableBackendError) as caught:
                wrapped.generate([], 17, 128, durable_tests.Config())
            self.assertEqual(classify_failure(caught.exception), 'INFRASTRUCTURE_OR_IMPLEMENTATION_FAILURE')
            self.assertEqual(native.calls, 0)

    stream = io.StringIO()
    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromModule(qualification_tests),
                               unittest.defaultTestLoader.loadTestsFromModule(durable_tests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(ExtraBoundaries)))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    for name, pin in manifest['files'].items(): assert sha(source/name) == pin, name
    assert sha(manifest_path) == manifest_pin
    helper_pin = sha(helper/'durable_backend.py')
    assert helper_pin == manifest['files']['deployment_sources/experiments/cycle2_20261113/durable_backend_v1/durable_backend.py']
    assert not any(name in sys.modules for name in ('torch', 'transformers'))
    output.mkdir(parents=True)
    (output/'CPU_TESTS.log').write_text(stream.getvalue(), encoding='utf-8')
    receipt = {'status': 'PASS_CPU_QUALIFICATION_PREPARATION_REVIEW' if result.wasSuccessful() else 'FAIL',
               'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'additional_independent_tests': 4,
               'manifest_sha256': manifest_pin, 'helper_sha256': helper_pin,
               'entry_sha256': sha(source/'qualify_hotpot_matrix.py'),
               'audit_sha256': sha(source/'matrix_qualification_audit.py'),
               'review_code_sha256': sha(__file__), 'sources_unchanged': True,
               'new_GPU_hours': 0, 'model_runtime_imported': False, 'benchmark_data_read': False,
               'actual_174_call_qualification': 'NOT_RUN_BY_THIS_REVIEW',
               'actual_Linux_guard_execution': 'NOT_RUN_BY_THIS_REVIEW; prior unchanged guard evidence is separate',
               'scientific_admission': False}
    (output/'RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    return int(not result.wasSuccessful())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.output))
