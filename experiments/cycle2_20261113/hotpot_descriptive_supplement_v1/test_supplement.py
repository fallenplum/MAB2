"""Small artificial matrices; no benchmark, answer metrics, or GPU."""
import copy
import importlib.util
import unittest
from unittest import mock
import supplement as s


def fixture(adapter):
    conditions = []; serial = 0
    for key in adapter.ORDER:
        name = key.split('/')[1]
        nodes = ('aggregate',) if name == 'SINGLE_M' else adapter.NODES
        records = {}
        for node in nodes:
            serial += 1
            record = {'node': node, 'input_fingerprint': 'synthetic', 'seed': serial,
                'messages': [{'role': 'user', 'content': 'PRIVATE_SYNTHETIC_SENTINEL'}],
                'generation': {'text': 'PRIVATE_SYNTHETIC_GENERATION', 'input_tokens': 100,
                    'output_tokens': 10, 'gpu_seconds': .25, 'elapsed_seconds': .5,
                    'gpu_peak_allocated_bytes': 1000, 'call_id': 'synthetic_' + str(serial),
                    'finish_reason': 'length' if node == 'r1:w0' else 'eos'},
                'parsed': {'answer': 'PRIVATE_SYNTHETIC_ANSWER', 'evidence': '', 'uncertainty': ''},
                'message_metadata': {'format_ok': not (node == 'aggregate' and key in ('S1/B', 'S2/A_MINUS_E'))},
                'received_message_tokens': 3, 'self_state_tokens': 2, 'reused': False}
            if key == 'S1/A' and node == 'r2:w0': record['generation'].pop('finish_reason')
            records[node] = record
        if key == 'S1/B':
            records['r1:w0'] = copy.deepcopy(conditions[0]['execution']['nodes']['r1:w0'])
            records['r1:w0']['reused'] = True
        if name in ('INDEPENDENT12', 'SINGLE_M'):
            order = ('aggregate',) if name == 'SINGLE_M' else tuple(f'r{sample}:w{role}' for role in range(4) for sample in range(1, 4)) + ('aggregate',)
            execution = {'slots': []}
            for index, node in enumerate(order):
                record = records[node]
                execution['slots'].append({'index': index, 'node': node, 'model': adapter._model(node),
                    'role': None if node == 'aggregate' else int(node[-1]),
                    'sample': None if node == 'aggregate' else int(node[1]),
                    'attempted_backend_request': True, 'status': 'SUCCESS', 'record': record,
                    'generation': record['generation'], 'seed': record['seed'],
                    'messages': record['messages'], 'input_fingerprint': record['input_fingerprint']})
        else: execution = {'nodes': records}
        traces = [{'node': node, 'model': adapter._model(node), 'generation': record['generation']}
            for node, record in records.items() if not record['reused']]
        conditions.append({'key': key, 'name': name, 'status': 'COMPLETE', 'execution': execution, 'request_trace': traces})
    return {'status': 'COMPLETE', 'conditions': conditions}


class SupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.adapter = s.frozen_adapter()

    def test_full_vs_physical_reuse_format_finish_and_private_projection(self):
        agg = s.Aggregator(self.adapter); agg.consume(fixture(self.adapter), 0)
        formats, logical, finish = agg.output_rows()
        self.assertEqual(sum(r['calls'] for r in logical), 131)
        self.assertEqual(agg.returned_count, 130)
        self.assertEqual(sum(r['reused_logical_calls_included'] for r in logical), 1)
        self.assertEqual(sum(r['aggregate_format_failed'] for r in formats), 2)
        self.assertEqual(sum(r['observed_records'] for r in finish if r['scope'] == 'logical'), 131)
        self.assertEqual(sum(r['observed_records'] for r in finish if r['scope'] == 'physical_returned'), 130)
        self.assertEqual(sum(r['unknown'] for r in finish if r['scope'] == 'logical'), 1)
        text = s.canonical([formats, logical, finish]).decode()
        for secret in ('PRIVATE_SYNTHETIC', 'synthetic_', 'input_fingerprint', 'call_id'):
            self.assertNotIn(secret, text)

    def test_four_method_costs_use_original_adapter_reduction(self):
        matrix = fixture(self.adapter); agg = s.Aggregator(self.adapter); agg.consume(matrix, 1)
        for condition in matrix['conditions']:
            if condition['key'] not in ('S1/B', 'S1/NO_EDGE', 'S1/INDEPENDENT12', 'S1/SINGLE_M'): continue
            original, _ = self.adapter._node_costs(self.adapter._records(condition['execution'], condition['name']))
            for model in s.MODELS:
                self.assertEqual({field: agg.logical[condition['key'], model][field] for field in s.MAIN_FIELDS}, original[model])
        self.assertEqual(agg.logical['S1/SINGLE_M', 'qwen']['calls'], 0)

    def test_format_predicate_matches_original_without_answer_metrics(self):
        path = s.ROOT / 'frozen_dependencies/hotpot_support.py'
        spec = importlib.util.spec_from_file_location('_synthetic_scorer_check', path)
        scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
        record = fixture(self.adapter)['conditions'][1]['execution']['nodes']['aggregate']
        with mock.patch.object(scorer, 'answer_metrics', side_effect=AssertionError('No scoring allowed')):
            self.assertFalse(s.strict_format_flag(record))
            self.assertFalse(scorer.score_aggregate_node(record, None)['format_ok'])
            record['message_metadata']['format_ok'] = True
            self.assertTrue(s.strict_format_flag(record))
        record['parsed']['extra'] = ''
        with self.assertRaises(ValueError): s.strict_format_flag(record)

    def test_rejected_full_cost_unknown_without_partial_imputation(self):
        matrix = fixture(self.adapter); condition = matrix['conditions'][-1]
        condition.update(status='INPUT_REJECTED', execution=None,
            request_trace=[{'generation': None}], known_partial_logical_cost={'calls': 12})
        matrix['status'] = 'COMPLETE_WITH_INPUT_REJECTIONS'
        agg = s.Aggregator(self.adapter); agg.consume(matrix, 0)
        self.assertIsNone(agg.logical['S2/B_MINUS_E', 'mistral']['calls'])
        self.assertEqual(agg.formats['S2/B_MINUS_E']['input_rejected'], 1)
        self.assertEqual(agg.unreturned_requests, 1)

    def test_duplicate_returned_physical_identity_refused(self):
        matrix = fixture(self.adapter)
        traces = matrix['conditions'][0]['request_trace']
        traces[1]['generation']['call_id'] = traces[0]['generation']['call_id']
        with self.assertRaises(ValueError): s.Aggregator(self.adapter).consume(matrix, 0)


if __name__ == '__main__': unittest.main()
