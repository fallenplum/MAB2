"""Frozen CPU-only I/C diagnostic. No raw text, labels, S inputs, or model calls."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL = '5336eacc6713d4e444ddb631032409cd2718ad3b4aaee1387aff5baac40f5ecf'
CONTRACT_SHA256 = '3e85f15ff4c653f533d7f4483000c3fcf8c8c2f11013619d18d3d191f3e2ff02'
DENOM_MIN = 1e-4
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 72031
METHODS = ('transport_beta1', 'direct_C5', 'old_bank', 'cv_leave_one_episode_out', 'beta_half', 'beta_zero_C3')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_contract(path=None):
    path = Path(path) if path is not None else Path(__file__).parent / 'FROZEN_DIAGNOSTIC_CONTRACT.json'
    raw = path.read_bytes()
    require(sha(raw) == CONTRACT_SHA256, 'Frozen diagnostic contract bytes changed')
    return json.loads(raw)


def specs():
    pairs = [(u, v) for u in range(4) for v in range(4) if u != v]
    editable0 = {(0, 1), (1, 0), (1, 2), (2, 0), (2, 3), (3, 0)}
    on0 = {(0, 1), (1, 0), (2, 0)}
    changed0 = {(0, 2), (1, 3), (3, 1)}
    rows = []
    for e in range(8):
        shift, complement = divmod(e, 2)
        before = [((u - shift) % 4, (v - shift) % 4) for u, v in pairs]
        edit = [j for j, pair in enumerate(before) if pair in editable0]
        changed = [j for j, pair in enumerate(before) if pair in changed0]
        A = [int((pair in on0) ^ bool(complement)) if pair in editable0 else 1 for pair in before]
        B = [0 if j in changed else bit for j, bit in enumerate(A)]
        recv = {pairs[j][1] for j in changed}
        rows.append({'editable_edges': edit, 'dirty': [int(pairs[j][1] in recv) for j in edit],
                     'A_masks': A, 'B_masks': B})
    return rows


def validate_arrays(arrays, contract):
    schema = contract['input_schema']
    require(set(arrays) == set(schema), 'Extra/missing input keys; S and unregistered data are forbidden')
    for name, declaration in schema.items():
        x = arrays[name]
        require(isinstance(x, np.ndarray) and list(x.shape) == declaration['shape'], f'{name}: shape mismatch')
        expected_dtype = np.dtype(declaration['dtype'])
        require(x.dtype == expected_dtype, f'{name}: dtype mismatch')
        if x.dtype.kind == 'U':
            require(all(len(v) == 64 and set(str(v)) <= set('0123456789abcdef') for v in x.flat),
                    f'{name}: must contain only SHA256 hex identifiers')
        else:
            require(np.isfinite(x).all(), f'{name}: nonfinite values')
            if isinstance(declaration.get('values'), list):
                require(np.isin(x, declaration['values']).all(), f'{name}: invalid values')
            if 'bounds' in declaration:
                lo, hi = declaration['bounds']
                require(((x >= lo) & (x <= hi)).all(), f'{name}: cost outside protocol bounds')
    hashes = list(arrays['I_qid_hash'].flat) + list(arrays['C_qid_hash'].flat)
    require(len(set(hashes)) == 192, 'I/C question hashes overlap or repeat')
    for e, graph in enumerate(specs()):
        for name, value in graph.items():
            require(np.array_equal(arrays[name][e], value), f'{name}: graph mismatch')
        for name in ('I_seed', 'C_seed'):
            require((arrays[name][e] == 3701 + e).all(), f'{name}: episode seed mismatch')
    a, b, joint = (arrays[name] for name in ('C_A_menu_tokens', 'C_B_menu_tokens', 'C_AB_joint_tokens'))
    require(((joint >= np.maximum(a, b)) & (joint <= a + b)).all(), 'Joint token accounting violates union bounds')


def moments(arrays):
    I = arrays['I_A_gain']
    A = arrays['C_A_quality'][..., 1:] - arrays['C_A_quality'][..., :1]
    B = arrays['C_B_quality'][..., 1:] - arrays['C_B_quality'][..., :1]
    D = B - A
    return I, A, B, D, I.var(axis=1, ddof=1), B.var(axis=1, ddof=1), D.var(axis=1, ddof=1)


def ratio(numerator, denominator, minimum=DENOM_MIN):
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or denominator <= minimum:
        return None
    return float(numerator / denominator)


def bootstrap_risk(I, A, B):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ratios = np.empty(N_BOOTSTRAP, dtype=float)
    invalid = 0
    for index in range(N_BOOTSTRAP):
        episodes = rng.integers(0, 8, size=8)
        irows = rng.integers(0, 12, size=(8, 12))
        crows = rng.integers(0, 12, size=(8, 12))
        old = I[episodes[:, None], irows]
        aa = A[episodes[:, None], crows]
        bb = B[episodes[:, None], crows]
        numerator = float((old.var(axis=1, ddof=1) / 12 + (bb - aa).var(axis=1, ddof=1) / 3).mean())
        denominator = float((bb.var(axis=1, ddof=1) / 5).mean())
        value = ratio(numerator, denominator)
        if value is None:
            invalid += 1
            ratios[index] = np.nan
        else:
            ratios[index] = value
    ci = None if invalid else np.quantile(ratios, [.05, .95], method='linear').tolist()
    return {'replicates': N_BOOTSTRAP, 'seed': BOOTSTRAP_SEED,
            'ci90': ci, 'undefined_replicates': invalid, 'undefined_fraction': invalid / N_BOOTSTRAP,
            'invalid_replicates_dropped': False, 'near_zero_denominator_threshold': DENOM_MIN,
            'interval_scope': 'exploratory eight-episode nested bootstrap; no coverage guarantee'}


def fit_beta(A, B, excluded_episode=None):
    episodes = [e for e in range(8) if e != excluded_episode]
    aa, bb = A[episodes], B[episodes]
    centered_a = aa - aa.mean(axis=1, keepdims=True)
    centered_b = bb - bb.mean(axis=1, keepdims=True)
    variance = float((centered_a ** 2).sum(axis=1).mean() / 11)
    covariance = float((centered_a * centered_b).sum(axis=1).mean() / 11)
    degenerate = variance <= 1e-12
    return {'beta': 0. if degenerate else covariance / (variance * (1 + 3 / 12)),
            'degenerate': degenerate, 'mean_within_cell_variance': variance,
            'mean_within_cell_covariance': covariance, 'excluded_episode': excluded_episode,
            'fit_episode_count': len(episodes), 'clipped': False}


def choose_slot(gains, edges):
    require(len(gains) == 6 and all(math.isfinite(float(v)) for v in gains), 'Invalid editor gains')
    j = min(range(6), key=lambda index: (-float(gains[index]), int(edges[index])))
    return j + 1 if float(gains[j]) > 0 else 0


def decision_diagnostic(arrays, I, A, B, contract):
    fold_actions = []
    episode_accuracy = {name: [] for name in METHODS}
    nochange = {'old_bank': [], 'transport_identity': [], 'pooled_old_plus_new3': []}
    betas = [fit_beta(A, B, e) for e in range(8)]
    for e in range(8):
        source = I[e].mean(axis=0)
        edge = arrays['editable_edges'][e]
        folds = {name: [] for name in METHODS}
        identity_folds = {name: [] for name in nochange}
        for fold_id, fold in enumerate(contract['folds']):
            tr3 = fold['train_rows'][:3]
            tr5 = fold['train_rows'][:5]
            heldout = fold['validation_rows']
            mean_a, mean_b = A[e, tr3].mean(axis=0), B[e, tr3].mean(axis=0)
            tables = {'transport_beta1': source + (B[e, tr3] - A[e, tr3]).mean(axis=0),
                      'direct_C5': B[e, tr5].mean(axis=0), 'old_bank': source,
                      'cv_leave_one_episode_out': mean_b - betas[e]['beta'] * (mean_a - source),
                      'beta_half': mean_b - .5 * (mean_a - source), 'beta_zero_C3': mean_b}
            # Complete all actions first, then inspect the heldout scores.
            actions = {name: choose_slot(tables[name], edge) for name in METHODS}
            samegraph = {'old_bank': choose_slot(source, edge),
                         'transport_identity': choose_slot(source + (A[e, tr3] - A[e, tr3]).mean(axis=0), edge),
                         'pooled_old_plus_new3': choose_slot((I[e].sum(axis=0) + A[e, tr3].sum(axis=0)) / 15, edge)}
            require(samegraph['old_bank'] == samegraph['transport_identity'], 'Exact-graph identity failed')
            fold_actions.append({'episode': e, 'fold': fold_id, 'action_slots': actions,
                                 'nochange_action_slots': samegraph})
            for name, slot in actions.items():
                folds[name].append(float(arrays['C_B_quality'][e, heldout, slot].mean()))
            for name, slot in samegraph.items():
                identity_folds[name].append(float(arrays['C_A_quality'][e, heldout, slot].mean()))
        for name in METHODS:
            episode_accuracy[name].append(sum(folds[name]) / 2)
        for name in nochange:
            nochange[name].append(sum(identity_folds[name]) / 2)
    delta_direct = [a - b for a, b in zip(episode_accuracy['transport_beta1'], episode_accuracy['direct_C5'])]
    delta_old = [a - b for a, b in zip(episode_accuracy['transport_beta1'], episode_accuracy['old_bank'])]
    return {'episode_accuracy': episode_accuracy,
            'mean_accuracy': {name: sum(values) / 8 for name, values in episode_accuracy.items()},
            'episode_transport_minus_direct': delta_direct, 'episode_transport_minus_old': delta_old,
            'mean_transport_minus_direct': sum(delta_direct) / 8,
            'mean_transport_minus_old': sum(delta_old) / 8,
            'episodes_nonnegative_transport_minus_direct': sum(v >= 0 for v in delta_direct),
            'fold_actions_frozen_before_validation': fold_actions, 'cv_coefficients': betas,
            'all_R1_development_beta_proposal_only': fit_beta(A, B),
            'nochange_by_construction': nochange,
            'nochange_is_new_independent_drift_evidence': False,
            'all_metrics_are_exploratory_R1_C_reuse': True}


def cost_diagnostic(arrays, vA, vB, vD, rT, rB):
    bank = arrays['I_A_menu_tokens'].sum(axis=1).astype(float)
    cA = arrays['C_A_menu_tokens'].mean(axis=1)
    cB = arrays['C_B_menu_tokens'].mean(axis=1)
    cD = arrays['C_AB_joint_tokens'].mean(axis=1)
    new_t, new_b = 3 * cD, 5 * cB
    total_t, total_b = bank + new_t, bank + new_b
    rt, rb = rT.mean(axis=1), rB.mean(axis=1)
    def product_ratio(a, b):
        # Risk support is checked separately; here denominator is in token-risk units.
        return ratio(float((rt * a).mean()), float((rb * b).mean()), minimum=0)
    construction_t = (np.sqrt(vA * cA[:, None]) + np.sqrt(vD * cD[:, None])) ** 2
    construction_b = vB * cB[:, None]
    return {'label': '初始化+反馈的实测token成本', 'includes_future_service': False,
            'bank_episode_tokens': bank.tolist(), 'transport_new_episode_tokens': new_t.tolist(),
            'direct_new_episode_tokens': new_b.tolist(),
            'transport_initialization_plus_feedback_tokens': total_t.tolist(),
            'direct_initialization_plus_feedback_tokens': total_b.tolist(),
            'initialization_plus_feedback_cost_weighted_risk_ratio': product_ratio(total_t, total_b),
            'incremental_cost_weighted_risk_ratio': product_ratio(new_t, new_b),
            'future_common_service_cap_added_sensitivity': {'common_cap': 1800192,
                'ratio': product_ratio(total_t + 1800192, total_b + 1800192), 'measured_service_cost': False},
            'cold_direct_without_bank_episode_tokens': new_b.tolist(),
            'continuous_from_scratch_coefficient_ratio': ratio(float(construction_t.mean()), float(construction_b.mean()), minimum=0),
            'joint_actual_calls_per_question': 37, 'standalone_cold_B_time_measured': False,
            'same_actual_spend_claim': False}


def evaluate_gates(support, risk_ratio, bootstrap, costs, decision):
    G1 = (support['nonzero_B_questions'] >= 12 and support['episodes_with_nonzero_B_variance'] >= 4
          and support['mean_direct_risk'] > DENOM_MIN)
    ci = bootstrap['ci90']
    cost_ratio = costs['initialization_plus_feedback_cost_weighted_risk_ratio']
    G2 = (risk_ratio is not None and risk_ratio <= .80 and ci is not None and ci[1] < 1
          and bootstrap['undefined_replicates'] == 0 and cost_ratio is not None and cost_ratio <= 1)
    G3 = (decision['mean_transport_minus_direct'] >= .01 and decision['mean_transport_minus_old'] >= 0
          and decision['episodes_nonnegative_transport_minus_direct'] >= 6)
    g2status = 'PASS' if G2 else ('NO_GO_CURRENT_BANK_AND_QUOTA' if ci is not None and ci[0] >= 1
                                else 'INCONCLUSIVE_CURRENT_BANK_AND_QUOTA')
    g3status = 'PASS' if G3 else ('NO_GO_DECISION_VALUE' if decision['mean_transport_minus_direct'] < 0
                                or decision['mean_transport_minus_old'] < 0 else 'INCONCLUSIVE_DECISION_VALUE')
    if not G1:
        overall = 'INCONCLUSIVE_LOW_SUPPORT'
    elif G1 and G2 and G3:
        overall = 'ELIGIBLE_FOR_G4_RESOURCE_SCREEN'
    elif g2status.startswith('NO_GO') or g3status.startswith('NO_GO'):
        overall = 'NO_GO_CURRENT_BANK_AND_QUOTA_OR_DECISION_VALUE'
    else:
        overall = 'INCONCLUSIVE_CURRENT_CANDIDATE'
    return {'G0': 'PASS', 'G1': 'PASS' if G1 else 'INCONCLUSIVE_LOW_SUPPORT',
            'G2': g2status, 'G3': g3status, 'G4': 'NOT_EVALUATED_RESOURCE_RECEIPT_REQUIRED',
            'overall': overall, 'GPU_launch_authorized_by_this_program': False, 'project_stop': False}


def diagnose(arrays, contract):
    validate_arrays(arrays, contract)
    I, A, B, D, vA, vB, vD = moments(arrays)
    rT, rB = vA / 12 + vD / 3, vB / 5
    primary_ratio = ratio(float(rT.mean()), float(rB.mean()))
    support = {'nonzero_B_questions': int(np.any(B != 0, axis=2).sum()),
               'episodes_with_nonzero_B_variance': int(np.any(vB > 0, axis=1).sum()),
               'mean_direct_risk': float(rB.mean()),
               'total_questions': 96, 'cells': 48,
               'zero_target_variance_cells': int((vB == 0).sum()),
               'zero_drift_variance_cells': int((vD == 0).sum())}
    bs = bootstrap_risk(I, A, B)
    costs = cost_diagnostic(arrays, vA, vB, vD, rT, rB)
    decision = decision_diagnostic(arrays, I, A, B, contract)
    projections = []
    for m in (12, 24, 48):
        projections.append({'m': m, 'n_transport': 3, 'n_direct': 5,
                            'ratio': ratio(float((vA / m + vD / 3).mean()), float(rB.mean())),
                            'bank_observed_at_this_size': m == 12, 'gate_substitution_allowed': False})
    slices = {}
    for label, mask in [('dirty', arrays['dirty'].astype(bool)), ('non_dirty', ~arrays['dirty'].astype(bool))]:
        slices[label] = {'cells': int(mask.sum()),
                         'risk_ratio': ratio(float(rT[mask].mean()), float(rB[mask].mean())),
                         'is_validity_certificate': False}
    identity_error = float(np.max(np.abs(A.mean(axis=1) + D.mean(axis=1) - B.mean(axis=1))))
    require(identity_error <= 1e-12, 'Same-panel identity numerical check failed')
    return {'schema': 'mab2_r2_ic_diagnostic_result_v1', 'contract_sha256': CONTRACT_SHA256,
            'r1_protocol_sha256': PROTOCOL, 'support': support,
            'primary_risk': {'ratio': primary_ratio, 'mean_transport_risk': float(rT.mean()),
                             'mean_direct_risk': float(rB.mean()),
                             'strict_bank_necessary_condition_cells': int((vA / 12 < vB / 5).sum()),
                             'v_A': vA.tolist(), 'v_B': vB.tolist(), 'v_D': vD.tolist(),
                             'transport_risk_cells': rT.tolist(), 'direct_risk_cells': rB.tolist()},
            'bootstrap': bs, 'costs': costs, 'decision': decision,
            'gates': evaluate_gates(support, primary_ratio, bs, costs, decision),
            'secondary': {'bank_projection': projections,
                          'm12_n12_failure_boundary_ratio': ratio(float((vA / 12 + vD / 12).mean()), float((vB / 12).mean())),
                          'dirty_slices': slices, 'same_panel_identity_max_error': identity_error},
            'evidence_scope': 'R1 I/C reused development diagnostics, not new independent confirmation',
            'S_access': False, 'new_model_calls': 0, 'new_formal_candidate_rounds': 0,
            'limitations': ['I-bank variance is estimated from only 12 questions per episode.',
                            'Eight graph episodes and six-row folds do not establish calibrated confidence guarantees.',
                            'Token cost diagnostic excludes actual future service cost; a later independent pilot must measure it.',
                            'Control-variate and paired-difference estimators are established methods, not new theory.']}


def canonical_digest(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8'))


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def load_inputs(private_dir, contract, *, synthetic_fixture=False):
    """Never opens COMPLETE, S receipts, public R1 summaries, or source graph text."""
    root = Path(private_dir)
    receipt_raw = (root / 'R2_IC_RECEIPT.json').read_bytes()
    receipt = json.loads(receipt_raw)
    require(receipt['schema'] == 'mab2_r2_ic_export_v1' and
            receipt['status'] == 'FROZEN_IC_ONLY_BEFORE_S_READ', 'Wrong I/C release status/schema')
    require(receipt['R1_protocol_sha256'] == PROTOCOL, 'Wrong R1 protocol')
    require(receipt['S_arrays_exported'] is False and receipt['S_labels_accessed_at_export'] is False
            and receipt['new_labels_or_outputs_scored_for_this_export'] is False,
            'Export violates pre-S/no-new-feedback information boundary')
    require(receipt['data_role'] == 'R1_development_I_C_only_not_independent_confirmation', 'Wrong data role')
    require(receipt['array_keys'] == sorted(contract['input_schema']), 'Receipt input allowlist differs')
    for field in ('R1_collection_plan_sha256', 'R1_actions_file_sha256', 'R1_action_payload_digest',
                  'npz_sha256', 'executor_protocol_hash', 'model_identity_sha256', 'model_identity_digest',
                  'quality_scorer_sha256', 'source_manifest_digest', 'I_C_source_graphs_sha256'):
        require(valid_sha(receipt[field]), f'{field}: invalid provenance digest')
    require(receipt['model_identity_sha256'] == receipt['model_identity_digest'], 'Model receipts disagree')
    version = receipt['analysis_source_version']
    if synthetic_fixture:
        require(version == 'SYNTHETIC_FIXTURE_ONLY' and receipt['analysis_source_commit_bytes_verified'] is False,
                'Synthetic mode is only for explicitly synthetic exporter receipts')
    else:
        require(receipt['analysis_source_commit_bytes_verified'] is True and isinstance(version, str)
                and len(version) in (40, 64) and set(version) <= set('0123456789abcdef'),
                'Formal input requires actual verified analysis commit')
        require(receipt['analysis_source_sha256'] and all(valid_sha(v) for v in receipt['analysis_source_sha256'].values()),
                'Formal analysis source manifest missing/invalid')
    source_pins = receipt['I_C_source_artifact_hashes']
    require(isinstance(source_pins, dict) and source_pins and all(valid_sha(v) for v in source_pins.values()),
            'I/C source artifact manifest invalid')
    require(canonical_digest(source_pins) == receipt['I_C_source_graphs_sha256'], 'I/C source manifest digest mismatch')
    action_raw = (root / 'ACTIONS_FROZEN.json').read_bytes()
    require(sha(action_raw) == receipt['R1_actions_file_sha256'], 'Action freeze file hash mismatch')
    actions = json.loads(action_raw)
    require(canonical_digest(actions['payload']) == actions['digest'] == receipt['R1_action_payload_digest'],
            'Action freeze payload binding mismatch')
    require(actions['payload']['purchases']['protocol_sha256'] == PROTOCOL, 'Actions use different protocol')
    npz_raw = (root / 'R2_IC_INPUTS.npz').read_bytes()
    require(sha(npz_raw) == receipt['npz_sha256'], 'I/C NPZ hash mismatch')
    with np.load(io.BytesIO(npz_raw), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    validate_arrays(arrays, contract)
    # Bind exported differences to the earlier immutable decision artifacts.
    initial = actions['payload']['purchases']['initial']
    require(len(initial) == 8, 'Action freeze lacks eight initial banks')
    for e in range(8):
        require(initial[e]['episode'] == e and
                canonical_digest(arrays['I_A_gain'][e].tolist()) == initial[e]['I_deltas_sha256'],
                'Exported I gains differ from frozen R1 bank')
    require(len(actions['payload']['records']) == 592, 'Action freeze replica coverage mismatch')
    C_by_condition = {'change': arrays['C_B_quality'], 'no_change': arrays['C_A_quality']}
    seen = set()
    for record in actions['payload']['records']:
        plan = record['plan']
        e, condition = plan['episode'], plan['condition']
        require(condition in C_by_condition and e in range(8), 'Action condition or episode invalid')
        require([edge for edge, _ in record['feedback_sha256']] == plan['purchased'], 'Frozen feedback coverage differs')
        quality = C_by_condition[condition][e]
        for edge, pin in record['feedback_sha256']:
            positions = np.flatnonzero(arrays['editable_edges'][e] == edge)
            require(len(positions) == 1, 'Frozen feedback edge not editable')
            j = int(positions[0])
            gain = quality[:, j + 1] - quality[:, 0]
            require(canonical_digest(gain.tolist()) == pin, 'Exported C gain differs from frozen R1 feedback')
            seen.add((e, condition, j))
    require(len(seen) == 96, 'Frozen feedback does not cover all I/C diagnostic cells')
    provenance = {key: receipt[key] for key in ('R1_collection_plan_sha256', 'R1_actions_file_sha256',
                  'R1_action_payload_digest', 'npz_sha256', 'executor_protocol_hash', 'model_identity_sha256',
                  'quality_scorer_sha256', 'source_manifest_digest', 'analysis_source_sha256',
                  'analysis_source_version', 'analysis_source_commit_bytes_verified', 'I_C_source_graphs_sha256')}
    provenance['IC_export_receipt_sha256'] = sha(receipt_raw)
    provenance['raw_source_artifacts_reread_by_diagnostic'] = False
    provenance['synthetic_fixture_mode'] = synthetic_fixture
    return arrays, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--synthetic-fixture', action='store_true', help='Only accepts SYNTHETIC_FIXTURE_ONLY receipts')
    args = parser.parse_args()
    require(not args.output.exists(), 'Refuse to overwrite a diagnostic result')
    contract = load_contract(args.contract)
    arrays, provenance = load_inputs(args.private_dir, contract, synthetic_fixture=args.synthetic_fixture)
    result = diagnose(arrays, contract)
    result['provenance'] = provenance
    result['diagnostic_source_sha256'] = sha(Path(__file__).read_bytes())
    result['input_mode'] = 'SYNTHETIC_FIXTURE_ONLY' if args.synthetic_fixture else 'R1_DEVELOPMENT_IC_ONLY'
    if args.synthetic_fixture:
        result['evidence_scope'] = 'synthetic exporter integration only; no real method outcome'
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'input_mode': result['input_mode'], 'gates': result['gates'],
                      'contract_sha256': CONTRACT_SHA256, 'S_access': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
