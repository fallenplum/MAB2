"""Synthetic tests for the frozen I/C screen. No real experiment file access."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import r2_ic_diagnostic as d


def fixture():
    s = d.specs()
    arrays = {
        'I_A_gain': np.zeros((8, 12, 6), dtype=np.float64),
        'C_A_quality': np.zeros((8, 12, 7), dtype=np.float64),
        'C_B_quality': np.zeros((8, 12, 7), dtype=np.float64),
        'I_A_menu_tokens': np.full((8, 12), 21000, dtype=np.int64),
        'C_A_menu_tokens': np.full((8, 12), 21000, dtype=np.int64),
        'C_B_menu_tokens': np.full((8, 12), 21000, dtype=np.int64),
        'C_AB_joint_tokens': np.full((8, 12), 37000, dtype=np.int64),
        'C_AB_joint_calls': np.full((8, 12), 37, dtype=np.int64),
        'I_qid_hash': np.array([[hashlib.sha256(f'synthetic-I-{e}-{q}'.encode()).hexdigest()
                                 for q in range(12)] for e in range(8)], dtype='U64'),
        'C_qid_hash': np.array([[hashlib.sha256(f'synthetic-C-{e}-{q}'.encode()).hexdigest()
                                 for q in range(12)] for e in range(8)], dtype='U64'),
        'I_seed': np.array([[3701 + e] * 12 for e in range(8)], dtype=np.int64),
        'C_seed': np.array([[3701 + e] * 12 for e in range(8)], dtype=np.int64),
    }
    for key in ('editable_edges', 'dirty', 'A_masks', 'B_masks'):
        arrays[key] = np.array([row[key] for row in s], dtype=np.int64)
    return arrays


def fluctuating_identical_worlds():
    a = fixture()
    pattern = np.array([0, 1] * 6, dtype=float)
    a['I_A_gain'][:, :, 0] = pattern
    a['C_A_quality'][:, :, 1] = pattern
    a['C_B_quality'][:, :, 1] = pattern
    return a


def write_synthetic_export(root):
    a = fixture()
    initial = [{'episode': e, 'I_deltas_sha256': d.canonical_digest(a['I_A_gain'][e].tolist())} for e in range(8)]
    records = []
    for e in range(8):
        for condition, quality in [('change', a['C_B_quality']), ('no_change', a['C_A_quality'])]:
            edges = a['editable_edges'][e].tolist()
            evidence = [[edge, d.canonical_digest((quality[e, :, j + 1] - quality[e, :, 0]).tolist())]
                        for j, edge in enumerate(edges)]
            for replica in range(37):
                records.append({'plan': {'episode': e, 'condition': condition, 'purchased': edges,
                                          'synthetic_index': replica}, 'feedback_sha256': evidence})
    payload = {'purchases': {'protocol_sha256': d.PROTOCOL, 'initial': initial}, 'records': records}
    action = {'payload': payload, 'digest': d.canonical_digest(payload)}
    action_raw = json.dumps(action, sort_keys=True).encode()
    (root / 'ACTIONS_FROZEN.json').write_bytes(action_raw)
    with (root / 'R2_IC_INPUTS.npz').open('wb') as handle:
        np.savez_compressed(handle, **a)
    sources = {'worker_0/units/SYNTHETIC_ONLY_I/A_stay.json': '1' * 64}
    receipt = {'schema': 'mab2_r2_ic_export_v1', 'status': 'FROZEN_IC_ONLY_BEFORE_S_READ',
        'R1_protocol_sha256': d.PROTOCOL, 'R1_collection_plan_sha256': '2' * 64,
        'R1_actions_file_sha256': d.sha(action_raw), 'R1_action_payload_digest': action['digest'],
        'npz_sha256': d.sha((root / 'R2_IC_INPUTS.npz').read_bytes()), 'executor_protocol_hash': '3' * 64,
        'model_identity_sha256': '4' * 64, 'model_identity_digest': '4' * 64,
        'quality_scorer_sha256': '5' * 64, 'source_manifest_digest': '6' * 64,
        'analysis_source_sha256': {}, 'analysis_source_version': 'SYNTHETIC_FIXTURE_ONLY',
        'analysis_source_commit_bytes_verified': False, 'array_keys': sorted(a),
        'I_C_source_artifact_hashes': sources, 'I_C_source_graphs_sha256': d.canonical_digest(sources),
        'S_arrays_exported': False, 'S_labels_accessed_at_export': False,
        'new_labels_or_outputs_scored_for_this_export': False,
        'data_role': 'R1_development_I_C_only_not_independent_confirmation'}
    (root / 'R2_IC_RECEIPT.json').write_text(json.dumps(receipt), encoding='utf-8')
    return a, receipt


class R2Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = d.load_contract()

    def test_frozen_thresholds_and_sizes(self):
        c = self.contract
        self.assertEqual((c['estimators']['m'], c['estimators']['n_transport'], c['estimators']['n_direct']), (12, 3, 5))
        self.assertEqual(c['bootstrap']['replicates'], 10000)
        self.assertEqual(c['bootstrap']['seed'], 72031)
        self.assertEqual(c['gates']['G1']['nonzero_B_questions_min'], 12)
        self.assertEqual(len(c['input_schema']), 16)

    def test_extra_S_input_rejected(self):
        a = fixture()
        a['S_quality'] = np.zeros((1,))
        with self.assertRaisesRegex(ValueError, 'S and unregistered'):
            d.validate_arrays(a, self.contract)

    def test_raw_question_key_rejected(self):
        a = fixture()
        a['question_text'] = np.array(['synthetic text'])
        with self.assertRaisesRegex(ValueError, 'Extra/missing'):
            d.validate_arrays(a, self.contract)

    def test_wrong_qid_dtype_rejected(self):
        a = fixture()
        a['I_qid_hash'] = a['I_qid_hash'].astype(object)
        with self.assertRaisesRegex(ValueError, 'dtype mismatch'):
            d.validate_arrays(a, self.contract)

    def test_question_overlap_rejected(self):
        a = fixture()
        a['C_qid_hash'][0, 0] = a['I_qid_hash'][0, 0]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            d.validate_arrays(a, self.contract)

    def test_wrong_seed_rejected(self):
        a = fixture()
        a['C_seed'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'seed mismatch'):
            d.validate_arrays(a, self.contract)

    def test_wrong_graph_rejected(self):
        a = fixture()
        a['dirty'][0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'graph mismatch'):
            d.validate_arrays(a, self.contract)

    def test_wrong_joint_calls_rejected(self):
        a = fixture()
        a['C_AB_joint_calls'][0, 0] = 42
        with self.assertRaisesRegex(ValueError, 'invalid values'):
            d.validate_arrays(a, self.contract)

    def test_cached_B_token_surrogate_detected_by_union_bounds(self):
        a = fixture()
        a['C_B_menu_tokens'][:] = 10000
        with self.assertRaisesRegex(ValueError, 'union bounds'):
            d.validate_arrays(a, self.contract)

    def test_noninteger_or_nonfinite_inputs_rejected(self):
        a = fixture()
        a['C_A_quality'][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            d.validate_arrays(a, self.contract)
        a = fixture()
        a['I_A_menu_tokens'] = a['I_A_menu_tokens'].astype(float)
        with self.assertRaisesRegex(ValueError, 'dtype mismatch'):
            d.validate_arrays(a, self.contract)

    def test_zero_drift_does_not_remove_bank_noise(self):
        a = fluctuating_identical_worlds()
        _, _, _, _, va, vb, vd = d.moments(a)
        self.assertTrue((vd == 0).all())
        self.assertGreater(va.mean() / 12, 0)
        self.assertAlmostEqual(float((va / 12).mean() / (vb / 5).mean()), 5 / 12)

    def test_same_panel_identity(self):
        a = fluctuating_identical_worlds()
        a['C_B_quality'][:, :4, 2] = 1
        _, A, B, D, *_ = d.moments(a)
        np.testing.assert_allclose(A.mean(1) + D.mean(1), B.mean(1), rtol=0, atol=1e-15)

    def test_near_zero_and_zero_risk_not_success(self):
        self.assertIsNone(d.ratio(0., 0.))
        self.assertIsNone(d.ratio(0., 1e-4))
        self.assertIsNone(d.ratio(1., 1e-5))

    def test_all_zero_bootstrap_preserves_invalid_replicates(self):
        a = fixture()
        I, A, B, *_ = d.moments(a)
        out = d.bootstrap_risk(I, A, B)
        self.assertEqual(out['undefined_replicates'], 10000)
        self.assertIsNone(out['ci90'])
        self.assertFalse(out['invalid_replicates_dropped'])

    def test_bootstrap_paired_worlds_and_determinism(self):
        a = fluctuating_identical_worlds()
        I, A, B, *_ = d.moments(a)
        first = d.bootstrap_risk(I, A, B)
        second = d.bootstrap_risk(I, A, B)
        self.assertEqual(first, second)
        self.assertEqual(first['undefined_replicates'], 0)
        self.assertLess(first['ci90'][1], 1.)

    def test_cv_optimal_identical_world_scalar(self):
        a = fluctuating_identical_worlds()
        _, A, B, *_ = d.moments(a)
        out = d.fit_beta(A, B, 0)
        self.assertAlmostEqual(out['beta'], .8)
        self.assertEqual(out['fit_episode_count'], 7)

    def test_cv_excludes_heldout_episode(self):
        a = fluctuating_identical_worlds()
        _, A, B, *_ = d.moments(a)
        before = d.fit_beta(A, B, 2)
        A[2] = 1 - A[2]
        B[2] = -B[2]
        after = d.fit_beta(A, B, 2)
        self.assertEqual(before, after)

    def test_cv_degeneracy_and_no_clipping(self):
        zero = np.zeros((8, 12, 6))
        self.assertTrue(d.fit_beta(zero, zero)['degenerate'])
        a = fluctuating_identical_worlds()
        _, A, _, *_ = d.moments(a)
        self.assertAlmostEqual(d.fit_beta(A, -A)['beta'], -.8)

    def test_nochange_identity_and_pooling_control(self):
        a = fluctuating_identical_worlds()
        I, A, B, *_ = d.moments(a)
        result = d.decision_diagnostic(a, I, A, B, self.contract)
        self.assertEqual(result['nochange_by_construction']['old_bank'], result['nochange_by_construction']['transport_identity'])
        self.assertIn('pooled_old_plus_new3', result['nochange_by_construction'])
        self.assertEqual(len(result['fold_actions_frozen_before_validation']), 16)

    def test_editor_strict_sign_and_physical_tie(self):
        edges = [0, 3, 4, 6, 8, 9]
        self.assertEqual(d.choose_slot([0] * 6, edges), 0)
        self.assertEqual(d.choose_slot([-1] * 6, edges), 0)
        self.assertEqual(d.choose_slot([0, .5, 0, .5, 0, 0], edges), 2)

    def test_training_prefixes_three_vs_five(self):
        a = fixture()
        a['C_B_quality'][:, [3, 4, 9, 10], 1] = 1
        I, A, B, *_ = d.moments(a)
        out = d.decision_diagnostic(a, I, A, B, self.contract)
        for fold in out['fold_actions_frozen_before_validation']:
            self.assertEqual(fold['action_slots']['transport_beta1'], 0)
            self.assertEqual(fold['action_slots']['direct_C5'], 1)

    def test_cost_includes_bank_and_excludes_future_service(self):
        a = fluctuating_identical_worlds()
        *_, va, vb, vd = d.moments(a)
        out = d.cost_diagnostic(a, va, vb, vd, va / 12 + vd / 3, vb / 5)
        self.assertEqual(out['bank_episode_tokens'], [252000.] * 8)
        self.assertEqual(out['transport_initialization_plus_feedback_tokens'], [363000.] * 8)
        self.assertEqual(out['direct_initialization_plus_feedback_tokens'], [357000.] * 8)
        self.assertAlmostEqual(out['initialization_plus_feedback_cost_weighted_risk_ratio'], (5 / 12) * 363 / 357)
        self.assertFalse(out['includes_future_service'])
        self.assertFalse(out['future_common_service_cap_added_sensitivity']['measured_service_cost'])

    def test_gates_cannot_launch_gpu_even_all_pass(self):
        support = {'nonzero_B_questions': 20, 'episodes_with_nonzero_B_variance': 8, 'mean_direct_risk': .02}
        bs = {'ci90': [.4, .7], 'undefined_replicates': 0}
        costs = {'initialization_plus_feedback_cost_weighted_risk_ratio': .8}
        decision = {'mean_transport_minus_direct': .02, 'mean_transport_minus_old': .01,
                    'episodes_nonnegative_transport_minus_direct': 7}
        out = d.evaluate_gates(support, .6, bs, costs, decision)
        self.assertEqual(out['overall'], 'ELIGIBLE_FOR_G4_RESOURCE_SCREEN')
        self.assertFalse(out['GPU_launch_authorized_by_this_program'])
        self.assertFalse(out['project_stop'])

    def test_invalid_bootstrap_cannot_pass_favorable_point(self):
        support = {'nonzero_B_questions': 20, 'episodes_with_nonzero_B_variance': 8, 'mean_direct_risk': .02}
        costs = {'initialization_plus_feedback_cost_weighted_risk_ratio': .1}
        decision = {'mean_transport_minus_direct': .02, 'mean_transport_minus_old': .01,
                    'episodes_nonnegative_transport_minus_direct': 8}
        out = d.evaluate_gates(support, .1, {'ci90': None, 'undefined_replicates': 1}, costs, decision)
        self.assertTrue(out['G2'].startswith('INCONCLUSIVE'))

    def test_low_support_precedes_apparent_gain(self):
        support = {'nonzero_B_questions': 2, 'episodes_with_nonzero_B_variance': 1, 'mean_direct_risk': 1e-5}
        out = d.evaluate_gates(support, None, {'ci90': None, 'undefined_replicates': 10000},
                              {'initialization_plus_feedback_cost_weighted_risk_ratio': None},
                              {'mean_transport_minus_direct': .5, 'mean_transport_minus_old': .5,
                               'episodes_nonnegative_transport_minus_direct': 8})
        self.assertEqual(out['overall'], 'INCONCLUSIVE_LOW_SUPPORT')

    def test_full_zero_fixture_is_not_scientific_success(self):
        out = d.diagnose(fixture(), self.contract)
        self.assertEqual(out['gates']['overall'], 'INCONCLUSIVE_LOW_SUPPORT')
        self.assertIsNone(out['primary_risk']['ratio'])
        self.assertEqual(out['new_model_calls'], 0)
        self.assertFalse(out['S_access'])
        json.dumps(out, allow_nan=False)

    def test_synthetic_receipt_hash_and_feedback_binding(self):
        with tempfile.TemporaryDirectory(prefix='r2-synthetic-') as temp:
            root = Path(temp)
            expected, _ = write_synthetic_export(root)
            actual, provenance = d.load_inputs(root, self.contract, synthetic_fixture=True)
            np.testing.assert_array_equal(actual['I_A_gain'], expected['I_A_gain'])
            self.assertTrue(provenance['synthetic_fixture_mode'])

    def test_formal_mode_rejects_synthetic_source_identity(self):
        with tempfile.TemporaryDirectory(prefix='r2-synthetic-') as temp:
            root = Path(temp)
            write_synthetic_export(root)
            with self.assertRaisesRegex(ValueError, 'verified analysis commit'):
                d.load_inputs(root, self.contract)

    def test_npz_tampering_rejected(self):
        with tempfile.TemporaryDirectory(prefix='r2-synthetic-') as temp:
            root = Path(temp)
            write_synthetic_export(root)
            with (root / 'R2_IC_INPUTS.npz').open('ab') as handle:
                handle.write(b'SYNTHETIC_CORRUPTION')
            with self.assertRaisesRegex(ValueError, 'NPZ hash mismatch'):
                d.load_inputs(root, self.contract, synthetic_fixture=True)

    def test_self_consistent_npz_tamper_still_fails_frozen_credit_binding(self):
        with tempfile.TemporaryDirectory(prefix='r2-synthetic-') as temp:
            root = Path(temp)
            arrays, receipt = write_synthetic_export(root)
            arrays['C_B_quality'][0, 0, 1] = 1
            with (root / 'R2_IC_INPUTS.npz').open('wb') as handle:
                np.savez_compressed(handle, **arrays)
            receipt['npz_sha256'] = d.sha((root / 'R2_IC_INPUTS.npz').read_bytes())
            (root / 'R2_IC_RECEIPT.json').write_text(json.dumps(receipt), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'C gain differs'):
                d.load_inputs(root, self.contract, synthetic_fixture=True)

    def test_after_S_export_guard_rejected_before_array_read(self):
        with tempfile.TemporaryDirectory(prefix='r2-synthetic-') as temp:
            root = Path(temp)
            _, receipt = write_synthetic_export(root)
            receipt['S_labels_accessed_at_export'] = True
            (root / 'R2_IC_RECEIPT.json').write_text(json.dumps(receipt), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'pre-S'):
                d.load_inputs(root, self.contract, synthetic_fixture=True)


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(R2Checks)
    names = [case.id().split('.')[-1] for case in suite]
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests_run': result.testsRun,
              'tests': names, 'failures': [(case.id(), msg) for case, msg in result.failures],
              'errors': [(case.id(), msg) for case, msg in result.errors],
              'contract_sha256': d.CONTRACT_SHA256,
              'program_sha256': d.sha(Path(d.__file__).read_bytes()),
              'tests_sha256': d.sha(Path(__file__).read_bytes()),
              'real_I_C_access': False, 'real_S_access': False, 'new_model_calls': 0,
              'evidence_scope': 'controlled synthetic verification only'}
    (Path(__file__).parent / 'SYNTHETIC_DIAGNOSTIC_CHECKS.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
