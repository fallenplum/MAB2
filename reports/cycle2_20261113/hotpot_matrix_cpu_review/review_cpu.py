"""Independent CPU boundary review; synthetic input only, no model runtime."""
import argparse
import copy
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True


def main(root, output):
    root, output = Path(root), Path(output)
    assert not output.exists()
    manifest_path = root / 'WORKING_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    pins = {row['path']: row['sha256'] for row in manifest['files']}
    for name, pin in pins.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == pin, name
    sys.path.insert(0, str(root))
    import test_matrix_executor as t
    import matrix_executor as m
    q = {'id': 'independent_synthetic_probe', 'question': 'What is the token?',
         'context': 'The complete synthetic text identifies the token as silver.'}
    seeds = (26111301, 26111302)

    class AdditionalChecks(unittest.TestCase):
        def test_full_context_and_unused_metadata_isolation(self):
            models, state = t.models()
            result = m.HotpotApplicabilityMatrixExecutor(models).execute(
                {**q, 'supporting_facts': 'SECRET_UNSEEN_METADATA', 'untrusted': {'answer': 'SECRET'}}, *seeds, 1)
            for request in state['requests']:
                raw = json.dumps(request['messages'])
                self.assertIn(q['context'], raw)
                self.assertNotIn('SECRET', raw)
            self.assertEqual(result.to_dict()['total_logical_service_cost']['calls'], 131)

        def test_single_overflow_does_not_cancel_second_seed(self):
            models, state = t.models(overflow_at={55})
            result = m.HotpotApplicabilityMatrixExecutor(models).execute(q, *seeds, 1)
            self.assertEqual(result.conditions[6].key, 'S1/SINGLE_M')
            self.assertEqual(result.conditions[6].status, 'INPUT_REJECTED')
            self.assertTrue(all(c.status == 'COMPLETE' for c in result.conditions[7:]))
            self.assertEqual(result.to_dict()['returned_generations'], 86)
            self.assertIsNone(result.to_dict()['total_logical_service_cost'])
            self.assertEqual(len(state['requests']), 87)

        def test_model_identity_mutation_keeps_first_paid_generation_and_stops(self):
            models, state = t.models()
            original = models['qwen'].generate
            def mutate(*args, **kwargs):
                generated = original(*args, **kwargs)
                models['mistral'].identity['version'] = 'changed_after_paid_first_request'
                return generated
            models['qwen'].generate = mutate
            with self.assertRaises(m.MatrixExecutionError) as caught:
                m.HotpotApplicabilityMatrixExecutor(models).execute(q, *seeds, 1)
            partial = caught.exception.partial_result
            self.assertEqual(len(state['requests']), 1)
            self.assertEqual(partial.conditions[0].physical_cost['returned_generations'], 1)
            self.assertEqual(len(partial.conditions[0].partial_execution['nodes']), 1)
            self.assertEqual(partial.conditions[0].status, 'INCOMPLETE')

        def test_postpaid_backend_output_violation_is_not_input_rejection(self):
            models, state = t.models()
            original = models['qwen'].generate
            def overlong(*args, **kwargs):
                generated = original(*args, **kwargs)
                changed = dataclasses.replace(generated, output_tokens=129)
                models['qwen'].physical_calls[-1]['output_tokens'] = 129
                return changed
            models['qwen'].generate = overlong
            with self.assertRaises(m.MatrixExecutionError) as caught:
                m.HotpotApplicabilityMatrixExecutor(models).execute(q, *seeds, 1)
            partial = caught.exception.partial_result
            self.assertEqual(len(state['requests']), 1)
            self.assertEqual(partial.conditions[0].status, 'INCOMPLETE')
            self.assertEqual(partial.conditions[0].request_trace[0]['generation']['output_tokens'], 129)
            self.assertEqual(partial.conditions[0].physical_cost['output_tokens'], 129)

        def test_fixed_model_composition_and_tokenizer_cost_for_controls(self):
            models, state = t.models()
            result = m.HotpotApplicabilityMatrixExecutor(models).execute(q, *seeds, 9)
            for name in ('A', 'B', 'NO_EDGE', 'A_MINUS_E', 'B_MINUS_E', 'INDEPENDENT12'):
                branch = next(c for c in result.conditions if c.seed_number == 1 and c.name == name)
                raw = branch.execution
                nodes = raw.get('nodes') or {s['node']: s['record'] for s in raw['slots']}
                counts = {'qwen': 0, 'mistral': 0}
                for node, record in nodes.items():
                    model = 'mistral' if node == 'aggregate' else ('qwen' if int(node[-1]) % 2 == 0 else 'mistral')
                    counts[model] += 1
                    for message in record['messages']:
                        if message['role'] == 'user': self.assertIn(q['context'], message['content'])
                self.assertEqual(counts, {'qwen': 6, 'mistral': 7})

    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromModule(t),
                               unittest.defaultTestLoader.loadTestsFromTestCase(AdditionalChecks)))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    assert not any(name in sys.modules for name in ('torch', 'transformers'))
    for name, pin in pins.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == pin, name
    output.mkdir(parents=True, exist_ok=False)
    (output/'CPU_TESTS.log').write_text(stream.getvalue(), encoding='utf-8')
    receipt = {'status': 'PASS_CPU_INDEPENDENT_BOUNDARY_REVIEW' if result.wasSuccessful() else 'FAIL',
               'tests_run': result.testsRun, 'author_tests_rerun': 15, 'additional_independent_boundary_tests': 5,
               'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
               'matrix_source_sha256': pins['matrix_executor.py'],
               'input_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'review_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'source_files_unchanged': True, 'scientific_admission': False,
               'models_or_GPU_used': False, 'benchmark_or_R1_data_read': False,
               'scope': 'Read-only reviewed source; rerun author synthetic tests plus independent boundary cases; no real-backend qualification.'}
    (output/'RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.root, args.output))
