"""Independent post-release arithmetic review; no inference/raw text/gold IO.

This is a review implementation written after final summaries were released.
It does not add or change a scientific gate. Outputs contain aggregate checks.
"""
import argparse
import datetime
import hashlib
import io
import itertools
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--postprocess', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    started = time.process_time()
    pins, checks = {}, []
    def read(path):
        raw = path.read_bytes()
        pins[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw
    def jread(path):
        return json.loads(read(path))
    def check(ok, label):
        if not ok:
            raise AssertionError(label)
        checks.append(label)
    def equal(a, b, label, tol=1e-12):
        check(np.allclose(a, b, rtol=1e-12, atol=tol, equal_nan=False), label)
    status = jread(args.postprocess / 'STATUS.json')
    check(status['status'] == 'POSTPROCESS_COMPLETE_ROOT_SCIENTIFIC_AND_VISUAL_REVIEW_REQUIRED', 'exact_release_gate')
    args.output.mkdir(parents=True, exist_ok=False)
    private = args.analysis / 'private'
    result_path = args.analysis / 'public/R1_RESULTS.json'
    result = jread(result_path)
    complete = jread(private / 'COMPLETE.json')
    replay = jread(args.postprocess / 'R1_INDEPENDENT_REPLAY.json')
    protocol = jread(args.repo / 'protocol/cycle2_20261113/R1_PROTOCOL.json')
    check(complete['public_result_sha256'] == pins[str(result_path)] == replay['official_result_sha256'], 'R1_result_hash_binding')
    check(result['protocol_sha256'] == pins[str(args.repo / 'protocol/cycle2_20261113/R1_PROTOCOL.json')], 'R1_protocol_hash')
    check(complete['source_archive_unchanged'] and complete['S_released_after_all_actions_frozen'], 'complete_release_attestations')
    check(replay['status'] == 'PASS' and not replay['mismatches'], 'prior_independent_replay_PASS')
    for name, expected in result['provenance']['code_identity']['analysis_source_sha256'].items():
        path = args.repo / 'experiments/cycle2_20261113/analysis' / name
        read(path)
        check(pins[str(path)] == expected, 'source_' + name)
    frozen = {}
    for name, field in [('OUTPUTS_FROZEN.json', 'output_freeze_sha256'),
                        ('PURCHASES_FROZEN.json', 'purchases_file_sha256'),
                        ('ACTIONS_FROZEN.json', 'actions_file_sha256')]:
        frozen[name] = jread(private / name)
        check(pins[str(private / name)] == result['provenance'][field], 'freeze_hash_' + name)
    for phase, previous in [('I', 'OUTPUTS_FROZEN.json'), ('C', 'PURCHASES_FROZEN.json'), ('S', 'ACTIONS_FROZEN.json')]:
        access = jread(private / ('FIRST_' + phase + '_LABEL_ACCESS.json'))
        check(access['preceding_freeze_sha256'] == pins[str(private / previous)], 'phase_dependency_' + phase)
    raw = read(private / 'REPLAY_INPUTS.npz')
    check(pins[str(private / 'REPLAY_INPUTS.npz')] == complete['replay_inputs_sha256'] == replay['replay_inputs_sha256'], 'R1_array_hash')
    with np.load(io.BytesIO(raw), allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    expected = {'I_delta','C_delta','S_quality','editable_edges','dirty','A_masks','B_masks','I_cost','C_base_cost','C_increment_cost','S_cost'}
    check(set(arrays) == expected, 'R1_anonymous_array_allowlist')
    methods = {(x['condition'], x['k'], x['method']): x for x in result['methods']}
    rebuilt = {key: np.zeros((8, 24)) for key in methods}
    costs = {key: np.zeros((8, 5)) for key in methods}
    groups = {}
    records = frozen['ACTIONS_FROZEN.json']['payload']['records']
    check(len(records) == 592, '592_frozen_replicas')
    for index, record in enumerate(records):
        plan = record['plan']
        e, k, condition, method = (plan[z] for z in ('episode','k','condition','method'))
        key = (condition, k, method)
        ci = ('change','no_change').index(condition)
        edges = arrays['editable_edges'][e].tolist()
        gain = arrays['I_delta'][e].mean(axis=0)
        width = np.sqrt(arrays['I_delta'][e].var(axis=0, ddof=1)/12)+1/12
        gap = width / (max(0, max(gain))-gain+width)
        dirty = arrays['dirty'][e] if ci == 0 else np.zeros(6)
        def rank(v):
            return [edges[j] for j in sorted(range(6), key=lambda j: (-v[j], edges[j]))]
        expected_buy = {
            'candidate': rank((1+dirty)*gap)[:k], 'no_graph_ablation': rank(gap)[:k],
            'max_uncertainty': rank(width)[:k], 'dirty_only': rank(dirty)[:k],
            'cyclic_pair': [edges[(e+j)%6] for j in range(k)],
            'source_only': [], 'current_stay': [], 'full_revalidation_high_cost': edges}
        purchased = plan['purchased']
        if method == 'uniform_pair_expectation':
            expected_sets = list(itertools.combinations(edges, k))
            check(tuple(purchased) == expected_sets[plan['replica']] and plan['weight_denominator'] == len(expected_sets), 'uniform_subset_' + str(index))
        else:
            check(purchased == expected_buy[method] and plan['weight_denominator'] == 1, 'I_only_purchase_' + str(index))
        groups.setdefault((e, *key), []).append(tuple(purchased))
        updated = gain.copy()
        for edge in purchased:
            j = edges.index(edge)
            updated[j] = arrays['C_delta'][e, ci, :, j].mean()
        equal(updated, record['updated_gain'], 'purchased_C_update_' + str(index))
        winner = min(range(6), key=lambda j: (-updated[j], edges[j]))
        action = None if method == 'current_stay' or updated[winner] <= 0 else edges[winner]
        check(action == record['action'], 'action_' + str(index))
        slot = 0 if action is None else edges.index(action)+1
        weight = plan['weight_denominator']
        check(plan['weight_numerator'] == 1, 'uniform_numerator_' + str(index))
        rebuilt[key][e] += arrays['S_quality'][e, ci, :, slot] / weight
        actual = arrays['S_cost'][e, ci, slot].copy()
        if method != 'current_stay':
            actual += arrays['I_cost'][e]
        if purchased:
            actual += arrays['C_base_cost'][e, ci]
            for edge in purchased:
                actual += arrays['C_increment_cost'][e, ci, edges.index(edge)]
        costs[key][e] += actual / weight
    for key, values in rebuilt.items():
        row = methods[key]
        equal(values.mean(axis=1), row['episode_accuracy'], 'episode_means_' + str(key))
        equal(values.mean(), row['accuracy'], 'accuracy_' + str(key))
        check(row['learning_episodes'] == 8 and row['service_questions'] == 192, 'unit_' + str(key))
    indices = np.random.default_rng(69173).integers(0, 8, size=(10000, 8))
    for name, contrast in result['contrasts'].items():
        condition, kval, rest = name.split('/')
        k = int(kval[1:])
        if condition == 'change-minus-no_change':
            delta = (rebuilt['change',k,'candidate'].mean(axis=1) - rebuilt['change',k,'uniform_pair_expectation'].mean(axis=1)
                     - rebuilt['no_change',k,'candidate'].mean(axis=1) + rebuilt['no_change',k,'uniform_pair_expectation'].mean(axis=1))
        else:
            baseline = rest.removeprefix('candidate-minus-')
            delta = rebuilt[condition,k,'candidate'].mean(axis=1) - rebuilt[condition,k,baseline].mean(axis=1)
        equal(delta, contrast['episode_values'], 'contrast_values_' + name)
        equal(delta.mean(), contrast['mean'], 'contrast_mean_' + name)
        equal(np.quantile(delta[indices].mean(axis=1), [.025,.975], method='linear'), contrast['ci95'], 'cluster_bootstrap_' + name)
    for row in result['logical_policy_costs']:
        key = (row['condition'], row['k'], row['method'])
        equal(costs[key], row['episode_actual_costs'], 'full_logical_cost_' + str(key), tol=1e-8)
        cap = row['conservative_cap_per_episode']
        check(np.all(costs[key][:,:2].sum(axis=1) <= cap['total_token_cap']), 'token_cap_' + str(key))
        equal(costs[key][:,3], cap['score_count_cap'], 'score_cap_' + str(key))
    primary = result['primary']['mean']
    source = result['contrasts']['change/k2/candidate-minus-source_only']['mean']
    check(result['point_rule_satisfied'] == (primary >= .01 and source >= 0), 'R1_point_gate')
    r2 = jread(args.postprocess / 'R2_IC_DIAGNOSTIC.json')
    contract_path = args.repo / 'experiments/cycle2_20261113/r2/FROZEN_DIAGNOSTIC_CONTRACT.json'
    contract = jread(contract_path)
    check(r2['contract_sha256'] == pins[str(contract_path)], 'R2_contract_hash')
    icreceipt = jread(private / 'R2_IC_RECEIPT.json')
    check(icreceipt['S_arrays_exported'] is False and icreceipt['S_labels_accessed_at_export'] is False, 'R2_preS_receipt')
    check(icreceipt['R1_actions_file_sha256'] == pins[str(private / 'ACTIONS_FROZEN.json')], 'R2_actions_binding')
    raw = read(private / 'R2_IC_INPUTS.npz')
    check(pins[str(private / 'R2_IC_INPUTS.npz')] == icreceipt['npz_sha256'] == r2['provenance']['npz_sha256'], 'R2_array_hash')
    with np.load(io.BytesIO(raw), allow_pickle=False) as bundle:
        z = {key: bundle[key] for key in bundle.files}
    check(set(z) == set(contract['input_schema']), 'R2_exact_IC_allowlist_no_S')
    check(len(set(z['I_qid_hash'].ravel()) | set(z['C_qid_hash'].ravel())) == 192, 'R2_disjoint_question_hashes')
    I = z['I_A_gain']; A = z['C_A_quality'][...,1:]-z['C_A_quality'][...,:1]
    B = z['C_B_quality'][...,1:]-z['C_B_quality'][...,:1]; D = B-A
    vA, vB, vD = (v.var(axis=1, ddof=1) for v in (I,B,D))
    rt, rb = vA/12 + vD/3, vB/5
    for key, value in [('v_A',vA),('v_B',vB),('v_D',vD),('transport_risk_cells',rt),('direct_risk_cells',rb)]:
        equal(value, r2['primary_risk'][key], 'R2_' + key)
    risk = float(rt.mean()/rb.mean())
    equal(risk, r2['primary_risk']['ratio'], 'R2_ratio_of_means')
    rng = np.random.default_rng(72031); ratios = []; invalid = 0
    for _ in range(10000):
        ee = rng.integers(0,8,size=8); ii = rng.integers(0,12,size=(8,12)); cc = rng.integers(0,12,size=(8,12))
        va = I[ee[:,None],ii].var(axis=1,ddof=1)
        vb = B[ee[:,None],cc].var(axis=1,ddof=1)
        vd = D[ee[:,None],cc].var(axis=1,ddof=1)
        den = (vb/5).mean()
        if not np.isfinite(den) or den <= .0001:
            invalid += 1
        else:
            ratios.append(float((va/12+vd/3).mean()/den))
    check(invalid == r2['bootstrap']['undefined_replicates'], 'R2_degenerate_bootstrap_count')
    if not invalid:
        equal(np.quantile(ratios,[.05,.95]), r2['bootstrap']['ci90'], 'R2_nested_bootstrap_CI')
    else:
        check(r2['bootstrap']['ci90'] is None, 'R2_any_invalid_null_interval')
    bank = z['I_A_menu_tokens'].sum(axis=1); ct = 3*z['C_AB_joint_tokens'].mean(axis=1); cb = 5*z['C_B_menu_tokens'].mean(axis=1)
    costrisk = float(np.mean(rt.mean(axis=1)*(bank+ct))/np.mean(rb.mean(axis=1)*(bank+cb)))
    equal(costrisk, r2['costs']['initialization_plus_feedback_cost_weighted_risk_ratio'], 'R2_full_acquisition_token_risk')
    direct_acc, transport_acc, old_acc = [], [], []
    for e in range(8):
        source_gain = I[e].mean(axis=0); fold_scores = [[],[],[]]
        for fold in contract['folds']:
            tr, valid = fold['train_rows'], fold['validation_rows']
            gains = [B[e,tr[:5]].mean(axis=0), source_gain+D[e,tr[:3]].mean(axis=0), source_gain]
            for j, g in enumerate(gains):
                best = min(range(6),key=lambda n:(-g[n],z['editable_edges'][e,n]))
                slot = best+1 if g[best]>0 else 0
                fold_scores[j].append(z['C_B_quality'][e,valid,slot].mean())
        direct_acc.append(np.mean(fold_scores[0])); transport_acc.append(np.mean(fold_scores[1])); old_acc.append(np.mean(fold_scores[2]))
    td = np.array(transport_acc)-direct_acc; to = np.array(transport_acc)-old_acc
    equal(td,r2['decision']['episode_transport_minus_direct'],'R2_heldout_direct_contrast')
    equal(to,r2['decision']['episode_transport_minus_old'],'R2_heldout_old_contrast')
    g1 = np.any(B!=0,axis=2).sum()>=12 and np.any(vB>0,axis=1).sum()>=4 and rb.mean()>.0001
    g2 = risk<=.8 and not invalid and max(ratios)>0 and np.quantile(ratios,.95)<1 and costrisk<=1
    g3 = td.mean()>=.01 and to.mean()>=0 and (td>=0).sum()>=6
    check(g1 == (r2['gates']['G1']=='PASS') and g2 == (r2['gates']['G2']=='PASS') and g3 == (r2['gates']['G3']=='PASS'), 'R2_G1_G2_G3')
    out = {'status':'PASS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'check_count':len(checks),'checks':checks,'input_sha256':pins,
           'implementation_authored_after_results_release':True,
           'no_new_statistical_gate':True,'R1_primary':result['primary'],'R1_point_rule_satisfied':result['point_rule_satisfied'],
           'R2_gates':r2['gates'],'R2_risk_ratio':risk,'R2_acquisition_token_risk_ratio':costrisk,
           'R2_undefined_bootstrap_replicates':invalid,'new_model_calls':0,'new_GPU_hours':0,
           'raw_questions_gold_or_model_text_read':False,'CPU_seconds':time.process_time()-started,
           'limits':['Does not independently reread raw execution artifacts or gold/scorer correctness.',
                     'Physical research costs and project budget require root ledger reconciliation.',
                     'Numeric checks verify frozen development arithmetic, not statistical coverage or future service benefit.']}
    out['review_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:out[k] for k in ('status','check_count','R1_point_rule_satisfied','R2_gates','CPU_seconds','review_source_sha256')},ensure_ascii=False))


if __name__ == '__main__':
    main()
