"""Only fabricated metadata, temporary artificial Parquet and reader spies."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import postclosure as p
import postclosure_cli as cli

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / 'frozen_dependencies' / 'evaluation_adapter.py'


def h(value):
    return p.sha(p.canonical(value))


def metadata_fixture(shards=None):
    descriptor = {'schema_version': '1.0', 'status': 'FROZEN', 'dataset_id': 'synthetic/hotpot',
        'revision': p.REVISION, 'config': 'distractor', 'split': 'train', 'source_audit_sha256': h('source-audit'),
        'read_columns': ['id', 'question', 'context', 'type', 'level'], 'parquet_files': shards or [
            {'path': f'train-{i}.parquet', 'sha256': h(['shard', i]), 'num_rows': 194} for i in range(2)]}
    questions = [{'id': f'artificial_{i:03d}', 'edge_index': (1, 5, 7, 9)[i // 96],
        'generation_seed1': 26111301, 'generation_seed2': 26111302} for i in range(p.N)]
    bindings = {key: h(key) for key in p.BINDINGS}
    bindings.update(evaluation_adapter_sha256=p.ADAPTER_PIN, selector_sha256=p.SELECTOR_PIN)
    # Required by the unchanged evaluation adapter even in the intentional
    # missing-archive test, which must reach file binding before label callback.
    bindings.update(scorer_sha256='60d056c6c74080ff1186762074b6fb85b9ea7e229143bedd74299d87a1864c0e',
        statistics_sha256='3417dc11bc583d72a2ec8c8cc139c6ee490aa039d8eb386c97f7e2aa52060bfe',
        frontier_sha256='6f641a3ff9c00afc2b045bc08435e42c602df5756747c58a7852f017da7bf7f9')
    for i, q in enumerate(questions):
        q.update(question_group_sha256=h(['g', i]), input_sha256=h(['input', i]), content_sha256=h(['content', i]))
    manifest = {'classification': 'PRIVATE_NO_LABELS', 'pins': {
        'source_descriptor_sha256': h(descriptor), 'protocol_sha256': bindings['scientific_protocol_sha256'],
        'selector_sha256': p.SELECTOR_PIN}, 'questions': questions}
    bindings['selection_manifest_sha256'] = h(manifest)
    template = {'schema': 'hotpot_closed_archive_score_release_v1', 'decision': 'REVIEW_REQUIRED',
        'bindings': bindings, 'scientific_protocol_status': 'FROZEN', 'reviewed_scopes': sorted(p.SCOPES),
        'unique_questions': 384, 'terminal_conditions': 4224, 'runtime_failure_count': 0,
        'all_attempts_closed': True, 'physical_ledger_closed': True, 'resident_ledger_closed': True,
        'label_vault_sha256': None, 'input_overflow_conditions': 0, 'unknown_preflight_component_requests': 0,
        'all_native_cost_fields_complete': True, 'root_must_acknowledge_unknown_preflight_components_before_release': False,
        'archive_metadata_sha256': {name: h(name) for name in p.META_NAMES},
        'units': [{'index': i, 'directory': f'worker_{i % 4}/unit_{i:03d}',
            'sha256': {name: h([i, name]) for name in ('INPUT.json', 'MATRIX.json', 'UNIT_RECEIPT.json')}} for i in range(384)]}
    contract = {'bindings': bindings, 'execution_protocol_sha256': h('later-resource-contract')}
    audit = {'status': 'REVIEW_REQUIRED', 'synthetic_tokenizer_witness': False,
        'no_label_vault_read': True, 'label_vault_binding_pending': True, 'label_release_authorized': False,
        'GPU_admission_granted': False, 'template_sha256': h(template), 'contract_sha256': h(contract),
        'preflight_rejections_with_no_native_record': 0,
        'audit_source_sha256': {name: h(name) for name in p.AUDIT_SOURCES}}
    evidence = {key: p.canonical(value) for key, value in dict(audit=audit, template=template,
        audit_contract=contract, selection_manifest=manifest, source_descriptor=descriptor).items()}
    approval = {'schema': 'hotpot_postclosure_label_extraction_release_v1',
        'decision': 'APPROVED_FOR_LABEL_EXTRACTION_ONLY', 'postclosure_source_sha256': p.source_sha(),
        'expected_questions': 384, 'expected_terminal_conditions': 4224, 'allowed_columns': ['id', 'answer'],
        'output_purpose': 'PRIVATE_LABEL_VAULT', 'scoring_authorized': False, 'publication_authorized': False,
        'evidence_sha256': {k: p.sha(v) for k, v in evidence.items()}, 'audit_source_sha256': audit['audit_source_sha256'],
        'source_identity': {k: descriptor[k] for k in ('dataset_id', 'revision', 'config', 'split')},
        'unknown_preflight_components_acknowledged': False,
        'test_only': 'FABRICATED_AUTHORITY_NOT_VALID_FOR_REAL_FILES'}
    return {'approval_bytes': p.canonical(approval), 'expected_approval_sha256': h(approval),
        'evidence': evidence, 'paths_by_sha256': {s['sha256']: s['path'] for s in descriptor['parquet_files']}}


def fake_rows(**kwargs):
    return [{'id': qid, 'answer': 'fabricated answer'} for qid in kwargs['selected_ids']]


def resign_fixture(arguments, evidence_key=None, mutate=None, approval_mutate=None):
    result = copy.deepcopy(arguments)
    approval = p.decode(result['approval_bytes'])
    if evidence_key:
        obj = p.decode(result['evidence'][evidence_key]); mutate(obj)
        result['evidence'][evidence_key] = p.canonical(obj)
    if approval_mutate:
        approval_mutate(approval)
    approval['evidence_sha256'] = {k: p.sha(v) for k, v in result['evidence'].items()}
    result.update(approval_bytes=p.canonical(approval), expected_approval_sha256=h(approval))
    return result


def scoring_arguments(extraction):
    vault, receipt = p.extract_vault(**extraction, reader=fake_rows)
    evidence = extraction['evidence']
    review = p.decode(evidence['template'])
    review.update(decision='APPROVED_FOR_SCORING', label_vault_sha256=p.sha(vault),
        postclosure={'schema': 'hotpot_postclosure_scoring_binding_v1',
            'extraction_approval_sha256': extraction['expected_approval_sha256'],
            'extraction_receipt_sha256': h(receipt), 'audit_sha256': p.sha(evidence['audit']),
            'template_sha256': p.sha(evidence['template']), 'postclosure_source_sha256': p.source_sha(),
            'unknown_preflight_components_acknowledged': False})
    args = {k: v for k, v in extraction.items() if k != 'paths_by_sha256'}
    args.update(extraction_receipt_bytes=p.canonical(receipt), score_review_bytes=p.canonical(review),
        expected_score_review_sha256=h(review), archive_files={}, adapter_path=ADAPTER)
    return args, vault


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.args = metadata_fixture()

    def test_unapproved_or_bad_sha_never_invokes_reader(self):
        for change in ('missing_pin', 'bad_pin', 'decision', 'source_pin', 'wrong_evidence'):
            with self.subTest(change=change):
                args = copy.deepcopy(self.args)
                if change == 'missing_pin': args['expected_approval_sha256'] = None
                if change == 'bad_pin': args['expected_approval_sha256'] = '0' * 64
                if change == 'decision': args = resign_fixture(args, approval_mutate=lambda a: a.update(decision='REVIEW_REQUIRED'))
                if change == 'source_pin': args = resign_fixture(args, approval_mutate=lambda a: a.update(postclosure_source_sha256='0' * 64))
                if change == 'wrong_evidence': args['evidence']['audit'] += b' '
                reader = mock.Mock(side_effect=AssertionError('Reader must never run'))
                with self.assertRaises(p.ReleaseError): p.extract_vault(**args, reader=reader)
                reader.assert_not_called()

    def test_synthetic_witness_unclosed_counts_or_costs_refused_before_reader(self):
        mutations = [('audit', lambda a: a.update(synthetic_tokenizer_witness=True)),
            ('audit', lambda a: a.update(synthetic_tokenizer_witness=0)),
            ('audit', lambda a: a.update(status='BLOCKED_EXTERNAL_ARCHIVE_AUDIT')),
            ('template', lambda t: t.update(unique_questions=383)),
            ('template', lambda t: t.update(terminal_conditions=4223)),
            ('template', lambda t: t.update(physical_ledger_closed=False)),
            ('template', lambda t: t.update(resident_ledger_closed=False)),
            ('template', lambda t: t.update(runtime_failure_count=1)),
            ('template', lambda t: t['units'].pop())]
        for key, mutate in mutations:
            args = resign_fixture(self.args, key, mutate)
            if key == 'template':
                pin = p.sha(args['evidence']['template'])
                args = resign_fixture(args, 'audit', lambda a: a.update(template_sha256=pin))
            reader = mock.Mock()
            with self.assertRaises(p.ReleaseError): p.extract_vault(**args, reader=reader)
            reader.assert_not_called()

    def test_wrong_source_split_revision_or_projection_refused(self):
        for key, value in (('split', 'validation'), ('revision', '1' * 40), ('read_columns', ['id', 'answer'])):
            args = resign_fixture(self.args, 'source_descriptor', lambda s: s.update({key: value}))
            reader = mock.Mock()
            with self.assertRaises(p.ReleaseError): p.extract_vault(**args, reader=reader)
            reader.assert_not_called()

    def test_unknown_preflight_requires_acknowledgement_without_fabricating_cost(self):
        args = resign_fixture(self.args, 'template', lambda t: t.update(unknown_preflight_component_requests=1,
            input_overflow_conditions=1, all_native_cost_fields_complete=False,
            root_must_acknowledge_unknown_preflight_components_before_release=True))
        pin = p.sha(args['evidence']['template'])
        args = resign_fixture(args, 'audit', lambda a: a.update(template_sha256=pin, preflight_rejections_with_no_native_record=1))
        reader = mock.Mock()
        with self.assertRaises(p.ReleaseError): p.extract_vault(**args, reader=reader)
        reader.assert_not_called()
        args = resign_fixture(args, approval_mutate=lambda a: a.update(unknown_preflight_components_acknowledged=True))
        vault, receipt = p.extract_vault(**args, reader=fake_rows)
        self.assertEqual(len(p.decode(vault)), 384)
        self.assertFalse(receipt['scoring_authorized'])

    def test_exact_384_answer_strings_no_normalization_no_extra_or_duplicate(self):
        def reader(**kwargs):
            return [{'id': qid, 'answer': '  Fabricated\n答案  '} for qid in kwargs['selected_ids']]
        vault, receipt = p.extract_vault(**self.args, reader=reader)
        self.assertEqual(set(p.decode(vault).values()), {'  Fabricated\n答案  '})
        self.assertEqual(receipt['selected_label_count'], 384)
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.append(rows[0]),
                       lambda rows: rows[0].update(id='not_selected'),
                       lambda rows: rows[0].update(answer=None),
                       lambda rows: rows[0].update(supporting_facts=[])):
            def invalid(**kwargs):
                rows = fake_rows(**kwargs); mutate(rows); return rows
            with self.assertRaises(p.ReleaseError): p.extract_vault(**self.args, reader=invalid)

    def test_artificial_parquet_two_shards_projection_filter_and_changed_sha(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            paths = []; shards = []
            for part in range(2):
                rows = [{'id': f'artificial_{i:03d}', 'answer': f'answer {i}', 'question': 'UNREAD QUESTION',
                    'context': 'UNREAD CONTEXT', 'type': 'bridge', 'level': 'easy',
                    'supporting_facts': 'UNREAD SUPPORT'} for i in range(part * 192, (part + 1) * 192)]
                rows.append(dict(rows[0], id=f'unselected_{part}', answer='UNSELECTED ARTIFICIAL GOLD'))
                path = Path(temp) / f'train-{part}.parquet'; pq.write_table(pa.Table.from_pylist(rows), path)
                paths.append(path); shards.append({'path': path.name, 'sha256': p.sha(path.read_bytes()), 'num_rows': len(rows)})
            args = metadata_fixture(shards)
            args['paths_by_sha256'] = {item['sha256']: path for item, path in zip(shards, paths)}
            original = pq.read_table
            with mock.patch.object(pq, 'read_table', wraps=original) as scan:
                vault, receipt = p.extract_vault(**args)
            self.assertEqual(len(p.decode(vault)), 384)
            self.assertEqual(p.decode(vault)['artificial_383'], 'answer 383')
            self.assertNotIn('UNSELECTED ARTIFICIAL GOLD', p.decode(vault).values())
            for call in scan.call_args_list:
                self.assertEqual(call.kwargs['columns'], ['id', 'answer'])
                self.assertEqual(call.kwargs['filters'][0][:2], ('id', 'in'))
                self.assertEqual(len(call.kwargs['filters'][0][2]), 384)
            self.assertFalse(receipt['row_level_byte_isolation'])
            paths[0].write_bytes(paths[0].read_bytes() + b'changed')
            with mock.patch.object(pq, 'read_table', side_effect=AssertionError('No decode after wrong hash')) as scan:
                with self.assertRaises(p.ReleaseError): p.extract_vault(**args)
                scan.assert_not_called()


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.args, self.vault = scoring_arguments(metadata_fixture())

    def test_missing_bad_or_wrong_scope_score_release_never_calls_vault_loader(self):
        for mode in ('missing', 'bad', 'wrong_decision'):
            args = dict(self.args)
            if mode == 'missing': args['expected_score_review_sha256'] = None
            if mode == 'bad': args['expected_score_review_sha256'] = '0' * 64
            if mode == 'wrong_decision':
                review = p.decode(args['score_review_bytes']); review['decision'] = 'APPROVED_FOR_LABEL_EXTRACTION_ONLY'
                args.update(score_review_bytes=p.canonical(review), expected_score_review_sha256=h(review))
            loader = mock.Mock()
            with self.assertRaises(p.ReleaseError): p.score_after_release(**args, vault_loader=loader)
            loader.assert_not_called()

    def test_changed_audited_template_or_vault_lineage_refused(self):
        for change in ('bindings', 'units', 'receipt', 'vault_pin'):
            args = dict(self.args); review = p.decode(args['score_review_bytes'])
            if change == 'bindings': review['bindings']['scientific_protocol_sha256'] = h('new science')
            if change == 'units': review['units'].pop()
            if change == 'receipt': review['postclosure']['extraction_receipt_sha256'] = h('wrong receipt')
            if change == 'vault_pin': review['label_vault_sha256'] = h('different vault')
            args.update(score_review_bytes=p.canonical(review), expected_score_review_sha256=h(review))
            loader = mock.Mock()
            with self.assertRaises(p.ReleaseError): p.score_after_release(**args, vault_loader=loader)
            loader.assert_not_called()

    def test_frozen_real_adapter_incomplete_archive_never_calls_loader(self):
        # Uses actual frozen source, no monkeypatch: no 384 model mock rebuild.
        loader = mock.Mock()
        with self.assertRaises(KeyError) as caught:
            p.score_after_release(**self.args, vault_loader=loader)
        self.assertIn(str(caught.exception).strip("'"), p.META_NAMES)
        loader.assert_not_called()

    def test_delegation_requests_default_statistics_and_only_adapter_calls_loader(self):
        # Function-call spy only; does not claim real archive validation/scoring.
        adapter = p._load_adapter(ADAPTER)
        order = []; loader = mock.Mock(side_effect=lambda request: order.append('vault') or self.vault)
        def delegated(**kwargs):
            order.append('adapter')
            self.assertNotIn('run_statistics', kwargs)
            self.assertEqual(kwargs['selection_manifest_bytes'], self.args['evidence']['selection_manifest'])
            labels = kwargs['label_loader']({'archive_review_sha256': self.args['expected_score_review_sha256'],
                'label_vault_sha256': p.sha(self.vault),
                'raw_question_ids': tuple(q['id'] for q in p.decode(self.args['evidence']['selection_manifest'])['questions'])})
            self.assertEqual(labels, self.vault)
            return {'test_only': 'DELEGATION_SPY_NO_SCORE'}
        with mock.patch.object(p, '_load_adapter', return_value=adapter), mock.patch.object(adapter, 'evaluate_after_release', side_effect=delegated):
            result = p.score_after_release(**self.args, vault_loader=loader)
        self.assertEqual(order, ['adapter', 'vault']); self.assertEqual(loader.call_count, 1)
        self.assertEqual(result['test_only'], 'DELEGATION_SPY_NO_SCORE')


class InterfaceTests(unittest.TestCase):
    def test_real_json_cli_extracts_artificial_vault_and_refuses_overwrite(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); shards = []
            for part in range(2):
                rows = [{'id': f'artificial_{i:03d}', 'answer': 'fake answer', 'question': 'fake question',
                    'context': 'fake context', 'type': 'bridge', 'level': 'easy', 'supporting_facts': 'unused'}
                    for i in range(part * 192, (part + 1) * 192)]
                path = root / f'train-{part}.parquet'; pq.write_table(pa.Table.from_pylist(rows), path)
                shards.append({'path': path.name, 'sha256': p.sha(path.read_bytes()), 'num_rows': 192})
            args = metadata_fixture(shards)
            (root / 'approval.json').write_bytes(args['approval_bytes'])
            for key, raw in args['evidence'].items(): (root / (key + '.json')).write_bytes(raw)
            io_spec = {'approval': 'approval.json', 'evidence': {k: k + '.json' for k in p.EVIDENCE_NAMES},
                'output': 'private_vault', 'parquet_paths_by_sha256': {s['sha256']: s['path'] for s in shards}}
            (root / 'io.json').write_bytes(p.canonical(io_spec))
            argv = ['--mode', 'extract', '--io', str(root / 'io.json'),
                '--extraction-approval-sha256', args['expected_approval_sha256']]
            with mock.patch('sys.stdout', new_callable=io.StringIO) as stdout: cli.main(argv)
            self.assertEqual(json.loads(stdout.getvalue())['status'], 'COMPLETE_PRIVATE_LABEL_EXTRACTION')
            vault = (root / 'private_vault/LABEL_VAULT.json').read_bytes()
            receipt = p.decode((root / 'private_vault/EXTRACTION_RECEIPT.json').read_bytes())
            self.assertEqual(receipt['label_vault_sha256'], p.sha(vault))
            self.assertEqual(len(p.decode(vault)), 384)
            self.assertFalse(receipt['scoring_authorized'])
            with mock.patch.object(p, 'read_selected_parquet', side_effect=AssertionError('No second extraction')) as reader:
                with self.assertRaises(p.ReleaseError): cli.main(argv)
                reader.assert_not_called()

    def test_default_cli_does_not_open_nonexistent_io(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'postclosure_cli.py'), '--io', 'nonexistent.json'],
            capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        self.assertEqual(data['status'], 'PREFLIGHT_NO_DATA_IO')
        self.assertFalse(data['labels_read'])

    def test_private_file_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'vault.json'; p.write_new(path, b'original')
            with self.assertRaises(FileExistsError): p.write_new(path, b'replacement')
            self.assertEqual(path.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
