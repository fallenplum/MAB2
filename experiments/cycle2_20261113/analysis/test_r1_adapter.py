"""Synthetic archive tests. The fixed protocol loader is mocked only for fixtures.

Real protocol byte identity is independently checked by policy tests/integration;
no original question, answer, or service output is opened here.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import r1_adapter as a
import r1_policy as p
from synthetic_collector_fixture import make_fixture, write, write_lines


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Path(__file__).resolve().parent / 'synthetic_test_tmp'
        cls.base.mkdir(exist_ok=True)
        cls.folder = Path(tempfile.mkdtemp(dir=cls.base))
        cls.fixture = make_fixture(cls.folder / 'fixture')
        cls.f = cls.fixture
        cls.scorer = staticmethod(a.load_scorer(Path(__file__).parent / 'test_support' / 'original_scoring.py'))
        with patch.object(p, 'specs_from_protocol_bytes', return_value=cls.f['specs']):
            a.owner_split_labels(cls.f['labels'], cls.f['plan_path'], cls.f['pins_path'], cls.f['protocol_path'],
                                 cls.f['data_receipt_path'], cls.f['selection_path'], cls.folder / 'released')
            cls.archive = a.audit_archive(cls.f['plan_path'], cls.f['pins_path'], cls.f['protocol_path'], cls.f['run'])
        cls.result = a.analyze_validated_archive(cls.archive, cls.folder / 'released' / 'PHASE_LABELS.json',
                                               cls.scorer, cls.folder / 'analysis', {'kind': 'SYNTHETIC_FIXTURE_ONLY'},
                                               label_descriptor_sha256=a.file_sha(cls.folder / 'released' / 'PHASE_LABELS.json'))

    @classmethod
    def tearDownClass(cls):
        # Delete only this known newly created synthetic fixture subtree.
        if hasattr(cls, 'folder'):
            assert cls.folder.resolve().is_relative_to(cls.base.resolve()) and cls.folder != cls.base
            shutil.rmtree(cls.folder)

    def test_full_390_units_4704_formal_outputs_and_call_denominator(self):
        self.assertEqual(len(self.archive.graphs), 390)
        formal = sum(len(self.archive.graphs[u['unit_id']]) for u in self.f['plan']['units'] if u['phase'] != 'qualification')
        self.assertEqual(formal, 4704)
        self.assertEqual(self.f['call_count'], 12948)
        self.assertEqual(self.result['physical_research_costs']['attempted_calls'], 12948)

    def test_owner_split_only_phase_files_no_qualification_labels(self):
        descriptor = a.parse((self.folder / 'released' / 'PHASE_LABELS.json').read_bytes())
        self.assertEqual({k: v['count'] for k, v in descriptor['phases'].items()}, {'I': 96, 'C': 96, 'S': 192})
        self.assertTrue(descriptor['owner_read_all_390_labels'])
        self.assertTrue(descriptor['owner_access_is_not_policy_access'])
        all_ids = []
        for phase, d in descriptor['phases'].items():
            rows = a._jsonl((self.folder / 'released' / d['path']).read_bytes())
            all_ids.extend(x['id'] for x in rows)
            self.assertTrue(all('_' + phase + '_' in x['id'] for x in rows))
        self.assertEqual(len(set(all_ids)), 384)

    def test_phase_receipts_bind_correct_predecessors(self):
        private = self.folder / 'analysis' / 'private'
        for phase, predecessor in [('I', 'OUTPUTS_FROZEN.json'), ('C', 'PURCHASES_FROZEN.json'), ('S', 'ACTIONS_FROZEN.json')]:
            event = a.parse((private / f'FIRST_{phase}_LABEL_ACCESS.json').read_bytes())
            self.assertEqual(event['preceding_freeze_sha256'], a.file_sha(private / predecessor))
            self.assertEqual(event['event'], 'HANDLE_OPENED_BEFORE_FIRST_READ')
        complete = a.parse((private / 'COMPLETE.json').read_bytes())
        self.assertTrue(complete['S_released_after_all_actions_frozen'])

    def test_original_scorer_consumes_raw_text_not_parsed_answer(self):
        unit = self.archive.unit_order[(0, 'I')][0]
        calls = []
        def spy(text, label, task):
            calls.append(text); return self.scorer(text, label, task)
        store = a.ScoringStore(self.archive, spy)
        store.open_phase('I', {unit['question_id']: {'answer': '0'}})
        store.score(unit, 'A', None)
        self.assertIn('"answer":', calls[0])
        self.assertIn('"evidence":', calls[0])

    def test_future_phase_score_blocked(self):
        store = a.ScoringStore(self.archive, self.scorer)
        store.open_phase('I', {})
        with self.assertRaisesRegex(ValueError, 'currently released phase'):
            store.score(self.archive.unit_order[(0, 'S')][0], 'A', None)
        with self.assertRaisesRegex(ValueError, 'phase transition'):
            store.open_phase('S', {})

    def test_S_unqueried_slots_stay_nan_and_required_slots_exist(self):
        path = self.folder / 'analysis' / 'private' / 'REPLAY_INPUTS.npz'
        with np.load(path) as data:
            self.assertEqual(data['I_delta'].shape, (8, 12, 6))
            self.assertEqual(data['C_delta'].shape, (8, 2, 12, 6))
            self.assertEqual(data['S_quality'].shape, (8, 2, 24, 7))
            self.assertTrue(np.isfinite(data['C_delta']).all())
            actions = a.parse((path.parent / 'ACTIONS_FROZEN.json').read_bytes())['payload']['records']
            selected = set()
            for rec in actions:
                plan = rec['plan']; e = plan['episode']; ci = p.CONDITIONS.index(plan['condition'])
                edges = tuple(data['editable_edges'][e]); slot = 0 if rec['action'] is None else 1 + edges.index(rec['action'])
                selected.add((e, ci, slot))
            for e in range(8):
                for ci in range(2):
                    for slot in range(7):
                        self.assertEqual(np.isfinite(data['S_quality'][e, ci, :, slot]).all(), (e, ci, slot) in selected)

    def test_physical_edge_mapping_on_reversed_fixture_menu(self):
        with np.load(self.folder / 'analysis' / 'private' / 'REPLAY_INPUTS.npz') as data:
            for e, spec in enumerate(self.f['specs']):
                for j, edge in enumerate(spec.editable):
                    for i, unit in enumerate(self.archive.unit_order[(e, 'I')]):
                        graphs = self.archive.graphs[unit['unit_id']].values()
                        base = next(g for g in graphs if g.request['action'] == 'stay')
                        flip = next(g for g in graphs if g.request.get('edge') == edge)
                        expected = self.scorer(flip.raw_aggregate_text, '0') - self.scorer(base.raw_aggregate_text, '0')
                        self.assertEqual(data['I_delta'][e, i, j], expected)

    def test_logical_C_B_base_and_S_toggle_use_nine_calls(self):
        with np.load(self.folder / 'analysis' / 'private' / 'REPLAY_INPUTS.npz') as data:
            np.testing.assert_array_equal(data['I_cost'][:, 2], np.full(8, 12 * 21))
            np.testing.assert_array_equal(data['C_base_cost'][:, :, 2], np.full((8, 2), 12 * 9))
            np.testing.assert_array_equal(data['C_increment_cost'][:, :, :, 2], np.full((8, 2, 6), 12 * 2))
            np.testing.assert_array_equal(data['S_cost'][:, :, :, 2], np.full((8, 2, 7), 24 * 9))
        for row in self.result['logical_policy_costs']:
            if row['method'] == 'uniform_pair_expectation':
                expected = 12 * 21 + 12 * (9 + 2 * row['k']) + 24 * 9
                self.assertAlmostEqual(row['mean_episode_actual_cost'][2], expected)

    def test_token_counts_exact_and_unknown_prior_cost_not_zero(self):
        for row in self.result['logical_policy_costs']:
            self.assertLessEqual(sum(row['mean_episode_actual_cost'][:2]), row['conservative_cap_per_episode']['total_token_cap'])
        self.assertIsNone(self.result['physical_research_costs']['prior_runs_GPUh'])
        self.assertFalse(self.result['physical_research_costs']['entire_project_cost_complete'])

    def test_producer_float_order_explicit_not_tolerant(self):
        self.assertEqual(a.legacy_float_sum([1., 1e-16, 1e-16]), 1.)
        correct = {'gpu_seconds': 1.}
        a.compare_cost(correct, correct)
        with self.assertRaisesRegex(ValueError, 'Floating graph cost mismatch'):
            a.compare_cost({'gpu_seconds': float(np.nextafter(1., 2.))}, correct)

    def test_missing_unit_refuses_before_phase_labels(self):
        marker = self.f['run'] / 'worker_0' / 'DEVELOPMENT_COMPLETE.json'
        original = marker.read_bytes()
        try:
            value = a.parse(original); value['completed_units'].pop(); write(marker, value)
            with patch.object(p, 'specs_from_protocol_bytes', return_value=self.f['specs']):
                with self.assertRaisesRegex(ValueError, 'unit coverage differs'):
                    a.audit_archive(self.f['plan_path'], self.f['pins_path'], self.f['protocol_path'], self.f['run'])
        finally:
            marker.write_bytes(original)

    def test_tampered_graph_bytes_refuse_before_scoring(self):
        unit = self.archive.unit_order[(0, 'I')][0]
        path = self.f['run'] / 'worker_0' / 'units' / unit['unit_id'] / 'A_stay.json'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b' ')
            with patch.object(p, 'specs_from_protocol_bytes', return_value=self.f['specs']):
                with self.assertRaisesRegex(ValueError, 'Graph bytes changed'):
                    a.audit_archive(self.f['plan_path'], self.f['pins_path'], self.f['protocol_path'], self.f['run'])
        finally:
            path.write_bytes(original)

    def test_owner_source_hash_mismatch_fails(self):
        bad = self.folder / 'wrong_labels.jsonl'; bad.write_bytes(self.f['labels'].read_bytes() + b' ')
        with patch.object(p, 'specs_from_protocol_bytes', return_value=self.f['specs']):
            with self.assertRaisesRegex(ValueError, 'original pinned data receipt'):
                a.owner_split_labels(bad, self.f['plan_path'], self.f['pins_path'], self.f['protocol_path'],
                                     self.f['data_receipt_path'], self.f['selection_path'], self.folder / 'bad_owner')

    def test_IC_export_is_exact_allowlist_and_excludes_S(self):
        private = self.folder / 'analysis' / 'private'
        expected = {'I_A_gain', 'C_A_quality', 'C_B_quality', 'I_A_menu_tokens', 'C_A_menu_tokens',
                    'C_B_menu_tokens', 'C_AB_joint_tokens', 'C_AB_joint_calls', 'I_qid_hash', 'C_qid_hash',
                    'I_seed', 'C_seed', 'editable_edges', 'dirty', 'A_masks', 'B_masks'}
        with np.load(private / 'R2_IC_INPUTS.npz') as data:
            self.assertEqual(set(data.files), expected)
            np.testing.assert_array_equal(data['C_AB_joint_calls'], np.full((8, 12), 37))
            self.assertEqual(len(set(data['I_qid_hash'].ravel()) | set(data['C_qid_hash'].ravel())), 192)
            self.assertTrue(all(data[k].dtype == np.int64 for k in expected if k.endswith('tokens') or k.endswith('calls')))
        receipt = a.parse((private / 'R2_IC_RECEIPT.json').read_bytes())
        self.assertEqual(receipt['npz_sha256'], a.file_sha(private / 'R2_IC_INPUTS.npz'))
        self.assertFalse(receipt['S_arrays_exported']); self.assertFalse(receipt['S_labels_accessed_at_export'])
        self.assertEqual(receipt['R1_actions_file_sha256'], a.file_sha(private / 'ACTIONS_FROZEN.json'))

    def test_independent_summary_cli_on_artificial_fixture(self):
        script = os.environ.get('MAB2_R1_REPLAY_SCRIPT')
        self.assertTrue(script and Path(script).is_file(), 'Run via RUN_ADAPTER_TESTS.py --replay-script ...')
        output = self.folder / 'independent_artificial_replay.json'
        result = subprocess.run([sys.executable, script, '--analysis-dir', str(self.folder / 'analysis'),
                                 '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        review = a.parse(output.read_bytes())
        self.assertTrue(str(review.get('status', '')).startswith('PASS'), str(review))

    def test_R2_IC_only_cli_on_artificial_fixture(self):
        script = os.environ.get('MAB2_R2_DIAGNOSTIC_SCRIPT')
        self.assertTrue(script and Path(script).is_file(), 'Run via RUN_ADAPTER_TESTS.py --r2-script ...')
        output = self.folder / 'R2_artificial_IC_diagnostic.json'
        result = subprocess.run([sys.executable, script, '--private-dir', str(self.folder / 'analysis' / 'private'),
                                 '--output', str(output), '--synthetic-fixture'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        review = a.parse(output.read_bytes())
        self.assertEqual(review['input_mode'], 'SYNTHETIC_FIXTURE_ONLY')
        self.assertEqual(review['evidence_scope'], 'synthetic exporter integration only; no real method outcome')

    def test_public_summary_contains_no_question_ids_gold_or_machine_path(self):
        text = (self.folder / 'analysis' / 'public' / 'R1_RESULTS.json').read_text(encoding='utf-8')
        self.assertNotIn('artificial_ep', text)
        self.assertNotIn('artificial_qual', text)
        self.assertNotIn(str(self.folder), text)
        self.assertNotIn('"answer":', text)


class ResourceAmendmentTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent / 'synthetic_test_tmp'; base.mkdir(exist_ok=True)
        self.base = base; self.root = Path(tempfile.mkdtemp(dir=base))
        old = {'workers': [{'index': i, 'resident_wall_cap_seconds': 28800} for i in range(3)],
               'runner_sha256': 'a' * 64, 'executor_protocol_hash': 'b' * 64,
               'science': {'frozen': True}}
        self.new = copy.deepcopy(old)
        for worker, cap in zip(self.new['workers'], (21600, 35280, 29520)):
            worker['resident_wall_cap_seconds'] = cap
        old_pin = write(self.root / 'old.json', old); self.new_pin = write(self.root / 'new.json', self.new)
        reports = {str(i): str(i) * 64 for i in range(3)}
        evidence = {'schema': 'mab2_cycle2_qualification_evidence_audit_v1',
                    'status': 'EVIDENCE_CHECKS_COMPLETE_REVIEW_REQUIRED', 'plan_sha256': old_pin,
                    'runner_sha256': old['runner_sha256'], 'executor_protocol_hash': old['executor_protocol_hash'],
                    'worker_report_hashes': reports}
        evidence_pin = write(self.root / 'evidence.json', evidence)
        accepted = {'status': 'PASS', 'plan_sha256': old_pin, 'formal_admitted': False,
                    'evidence_sha256': evidence_pin, 'worker_reports': reports}
        accepted_pin = write(self.root / 'accepted_old.json', accepted)
        normalized = copy.deepcopy(old)
        for worker in normalized['workers']: worker.pop('resident_wall_cap_seconds')
        self.cert = {'schema': 'mab2_cycle2_resource_only_amendment_v1', 'status': 'FROZEN',
            'scientific_protocol_sha256': p.PROTOCOL_SHA256, 'formal_rounds_started_at_amendment': 0,
            'scientific_outcomes_read': False, 'old_qualification_costs_remain_charged': True,
            'old_plan': {'path': 'old.json', 'sha256': old_pin},
            'new_plan': {'path': 'new.json', 'sha256': self.new_pin},
            'qualification_evidence': {'path': 'evidence.json', 'sha256': evidence_pin},
            'qualification_acceptance_v1': {'path': 'accepted_old.json', 'sha256': accepted_pin},
            'unchanged_fields_sha256': a.digest(normalized), 'both_total_seconds': 86400,
            'qualification_resident_seconds': [500., 600., 700.],
            'changes': [{'path': f'workers[{i}].resident_wall_cap_seconds', 'old': 28800, 'new': cap}
                        for i, cap in enumerate((21600, 35280, 29520))],
            'phase_plan_binding': {'qualification': old_pin, 'I_C_S': self.new_pin}}
        self.qa = {'worker_reports': reports}

    def tearDown(self):
        assert self.root.resolve().is_relative_to(self.base.resolve())
        shutil.rmtree(self.root)

    def validate(self):
        pin = write(self.root / 'cert.json', self.cert); self.qa['resource_amendment_sha256'] = pin
        return a.validate_resource_amendment(self.root / 'cert.json', self.root, self.new, self.new_pin, self.qa)

    def test_exact_resource_change_accepted(self):
        result = self.validate()
        self.assertEqual(result['new_worker_caps_seconds'], [21600, 35280, 29520])

    def test_scientific_or_assignment_change_rejected_even_if_rehashed(self):
        self.new['science']['frozen'] = False
        self.new_pin = write(self.root / 'new.json', self.new)
        self.cert['new_plan']['sha256'] = self.new_pin; self.cert['phase_plan_binding']['I_C_S'] = self.new_pin
        with self.assertRaisesRegex(ValueError, 'non-cap field'):
            self.validate()

    def test_new_budget_total_increase_rejected(self):
        self.new['workers'][0]['resident_wall_cap_seconds'] += 1
        self.new_pin = write(self.root / 'new.json', self.new)
        self.cert['new_plan']['sha256'] = self.new_pin; self.cert['phase_plan_binding']['I_C_S'] = self.new_pin
        self.cert['changes'][0]['new'] += 1
        with self.assertRaisesRegex(ValueError, 'exact total budget'):
            self.validate()

    def test_outcome_driven_amendment_rejected(self):
        self.cert['scientific_outcomes_read'] = True
        with self.assertRaisesRegex(ValueError, 'Invalid resource-only'):
            self.validate()

    def test_qualification_wrong_old_pin_rejected(self):
        self.cert['phase_plan_binding']['qualification'] = self.new_pin
        with self.assertRaisesRegex(ValueError, 'phase binding'):
            self.validate()

    def test_omitted_qualification_expense_rejected_by_cap(self):
        self.cert['qualification_resident_seconds'][0] = 21601
        with self.assertRaisesRegex(ValueError, 'already spent qualification'):
            self.validate()


class LedgerAndGitGuardsTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parent / 'synthetic_test_tmp'; self.base.mkdir(exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(dir=self.base)); self.worker = self.folder / 'worker_0'
        self.worker.mkdir()
        self.journal = [
            {'event': 'CALL_START', 'reservation_id': 'q1', 'unit_id': 'q', 'request_name': 'A_stay',
             'plan_sha256': 'old', 'seed': 11, 'prompt_sha256': 'prompt'},
            {'event': 'CALL_FAILED', 'reservation_id': 'q1', 'unit_id': 'q', 'request_name': 'A_stay', 'plan_sha256': 'old'},
            {'event': 'CALL_START', 'reservation_id': 'd1', 'unit_id': 'd', 'request_name': 'A_stay',
             'plan_sha256': 'new', 'seed': 12, 'prompt_sha256': 'prompt'},
            {'event': 'CALL_FAILED', 'reservation_id': 'd1', 'unit_id': 'd', 'request_name': 'A_stay', 'plan_sha256': 'new'}]
        self.residence = [
            {'event': 'START', 'attempt_id': 'q', 'phase': 'qualification', 'plan_sha256': 'old'},
            {'event': 'END', 'attempt_id': 'q', 'resident_wall_seconds': 1.25, 'plan_sha256': 'old'},
            {'event': 'START', 'attempt_id': 'd', 'phase': 'development', 'plan_sha256': 'new'},
            {'event': 'END', 'attempt_id': 'd', 'resident_wall_seconds': 2.5, 'plan_sha256': 'new'}]
        # A native failure still remains in the physical ledger; its unknown cost is not zero.
        write_lines(self.worker / 'physical_calls.jsonl', [{'call_id': 'failure', 'status': 'FAILED'}])

    def tearDown(self):
        assert self.folder.resolve().is_relative_to(self.base.resolve())
        shutil.rmtree(self.folder)

    def read(self):
        write_lines(self.worker / 'call_journal.jsonl', self.journal)
        write_lines(self.worker / 'resident_wall.jsonl', self.residence)
        return a.read_worker_costs(self.folder, 0, {}, {'q': 'old', 'd': 'new'},
                                   {'qualification': 'old', 'development': 'new'})

    def test_failed_calls_keep_phase_pins_and_unknown_cost(self):
        _, costs = self.read()
        self.assertEqual(costs['attempted_calls'], 2); self.assertEqual(costs['failed_calls'], 2)
        self.assertIsNone(costs['unmeasured_failure_tokens'])
        self.assertEqual(costs['qualification_resident_wall_seconds'], 1.25)

    def test_failed_call_cannot_use_other_valid_phase_pin(self):
        self.journal[-1]['plan_sha256'] = 'old'
        with self.assertRaisesRegex(ValueError, 'unit/phase'):
            self.read()

    def test_call_unit_must_belong_to_worker_and_reservation(self):
        self.journal[-1]['unit_id'] = 'other_worker_unit'
        with self.assertRaisesRegex(ValueError, 'unit/phase'):
            self.read()

    def test_residence_phase_cannot_use_other_valid_pin(self):
        self.residence[2]['plan_sha256'] = self.residence[3]['plan_sha256'] = 'old'
        with self.assertRaisesRegex(ValueError, 'Residence ledger phase'):
            self.read()

    def test_worker_and_global_attempt_caps_include_failures(self):
        costs = [{'worker': 0, 'attempted_calls': 3}, {'worker': 1, 'attempted_calls': 4}]
        workers = [{'index': 0, 'max_call_attempts': 3}, {'index': 1, 'max_call_attempts': 4}]
        a.require_attempt_caps(costs, workers, {'max_call_attempts': 7})
        with self.assertRaisesRegex(ValueError, 'Global call-attempt'):
            a.require_attempt_caps(costs, workers, {'max_call_attempts': 6})
        costs[0]['attempted_calls'] = 4
        with self.assertRaisesRegex(ValueError, 'Worker call-attempt'):
            a.require_attempt_caps(costs, workers, {'max_call_attempts': 8})

    def test_git_exact_safe_directory_per_command_and_raw_byte_equality(self):
        repo = Path(a.__file__).resolve().parent
        calls = []
        def mock_run(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(argv[:5], ['git', '-c', 'safe.directory=' + repo.as_posix(), '-C', str(repo)])
            if argv[5] == 'rev-parse': value = 'a' * 40
            elif argv[5] == 'status': value = ''
            else: value = (repo / argv[-1].split(':', 1)[1]).read_bytes()
            return subprocess.CompletedProcess(argv, 0, stdout=value)
        with patch.object(a.subprocess, 'run', side_effect=mock_run):
            identity = a.code_identity(repo)
        self.assertEqual(len(calls), 4)
        self.assertTrue(identity['tracked_source_bytes_equal_HEAD'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
