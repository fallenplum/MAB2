"""Prospective rule review: synthetic cases only, never real outcomes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True


def run(source, output):
    source, output = Path(source), Path(output)
    assert not output.exists()
    filenames = ('point_frontier.py', 'test_point_frontier.py', 'PROSPECTIVE_RULE.md')
    pins = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in filenames}
    sys.path.insert(0, str(source))
    import test_point_frontier as tests
    from point_frontier import review_frontier

    class IndependentBoundaries(unittest.TestCase):
        def test_nonfinite_measurement_is_rejected(self):
            for value in (float('nan'), float('inf'), -1.0, True):
                quality, costs, audit = tests.data()
                costs['B']['mistral']['CUDA_event_seconds'] = value
                with self.assertRaises(ValueError): review_frontier(quality, costs, **audit)

        def test_all_coverage_and_integrity_conditions_are_required(self):
            for key, value in (('unique_questions', 383), ('complete_conditions', 4223),
                               ('archive_integrity_passed', False), ('runtime_failures', 1)):
                quality, costs, audit = tests.data()
                audit[key] = value
                self.assertEqual(review_frontier(quality, costs, **audit)['status'], 'NOT_ASSESSED')

        def test_unknown_in_any_model_or_method_is_not_zero(self):
            for method in ('B', 'NO_EDGE', 'INDEPENDENT12', 'SINGLE_M'):
                quality, costs, audit = tests.data()
                costs[method]['qwen']['input_tokens'] = None
                result = review_frontier(quality, costs, **audit)
                self.assertEqual(result['status'], 'NOT_ASSESSED')
                self.assertFalse(result['GPU_admission_granted'])

    stream = io.StringIO()
    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromModule(tests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(IndependentBoundaries)))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    for name, pin in pins.items(): assert hashlib.sha256((source/name).read_bytes()).hexdigest() == pin
    assert not any(name in sys.modules for name in ('torch', 'transformers'))
    output.mkdir(parents=True)
    (output/'CPU_TESTS.log').write_text(stream.getvalue(), encoding='utf-8')
    receipt = {'status': 'PASS_PROSPECTIVE_RULE_CPU_REVIEW' if result.wasSuccessful() else 'FAIL',
               'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'source_sha256': pins, 'source_unchanged': True, 'real_data_read': False,
               'new_GPU_hours': 0, 'scientific_admission': False,
               'scope': 'Author tests rerun plus three independent schema/coverage boundary tests; descriptive frontier only.'}
    (output/'RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    return int(not result.wasSuccessful())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.output))
