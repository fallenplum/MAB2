"""Independent scalar R1 audit. No production-policy imports or raw-label reads.

Real replay requires the existing official COMPLETE receipt and hash bindings.
This is an evaluator after scoring, never a policy input or a new experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL = '5336eacc6713d4e444ddb631032409cd2718ad3b4aaee1387aff5baac40f5ecf'
CONDITIONS = ('change', 'no_change')
METHODS = ('candidate', 'uniform_pair_expectation', 'cyclic_pair',
           'max_uncertainty', 'no_graph_ablation', 'dirty_only',
           'source_only', 'current_stay', 'full_revalidation_high_cost')
FIELDS = ('input_tokens', 'output_tokens', 'calls', 'score_count', 'score_cpu_seconds')
PRIMARY = 'change/k2/candidate-minus-uniform_pair_expectation'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def payload_digest(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode('utf-8'))


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def graph_specs():
    """Construct endpoint matrices, independently of the production dataclasses."""
    endpoints = [(u, v) for u in range(4) for v in range(4) if u != v]
    editable_template = {(0, 1), (1, 0), (1, 2), (2, 0), (2, 3), (3, 0)}
    editable_on = {(0, 1), (1, 0), (2, 0)}
    changed_template = {(0, 2), (1, 3), (3, 1)}
    result = []
    for episode in range(8):
        shift, complement = divmod(episode, 2)
        local_edges = [((u - shift) % 4, (v - shift) % 4) for u, v in endpoints]
        editable = [e for e, pair in enumerate(local_edges) if pair in editable_template]
        changed = [e for e, pair in enumerate(local_edges) if pair in changed_template]
        A = [int((pair in editable_on) ^ bool(complement))
             if pair in editable_template else 1 for pair in local_edges]
        B = [0 if e in changed else bit for e, bit in enumerate(A)]
        receivers = {endpoints[e][1] for e in changed}
        dirty = [int(endpoints[e][1] in receivers) for e in editable]
        result.append({'episode': episode, 'editable': editable, 'dirty': dirty,
                       'A': A, 'B': B, 'changed': changed})
    return result


def validate_arrays(arrays):
    shapes = {'I_delta': (8, 12, 6), 'C_delta': (8, 2, 12, 6),
              'S_quality': (8, 2, 24, 7), 'editable_edges': (8, 6),
              'dirty': (8, 6), 'A_masks': (8, 12), 'B_masks': (8, 12),
              'I_cost': (8, 5), 'C_base_cost': (8, 2, 5),
              'C_increment_cost': (8, 2, 6, 5), 'S_cost': (8, 2, 7, 5)}
    require(set(arrays) == set(shapes), 'NPZ keys differ from frozen replay schema')
    for name, shape in shapes.items():
        require(np.shape(arrays[name]) == shape, f'{name}: wrong shape')
    for name in ('I_delta', 'C_delta'):
        require(np.isfinite(arrays[name]).all() and
                np.isin(arrays[name], [-1, 0, 1]).all(), f'{name}: invalid differences')
    service = arrays['S_quality']
    require(not np.isinf(service).any(), 'Infinite service quality')
    require(np.isin(service[np.isfinite(service)], [0, 1]).all(), 'Nonbinary service quality')
    for spec in graph_specs():
        e = spec['episode']
        for name, field in [('editable_edges', 'editable'), ('dirty', 'dirty'),
                            ('A_masks', 'A'), ('B_masks', 'B')]:
            require(np.array_equal(arrays[name][e], spec[field]), f'{name}: frozen graph mismatch')
    for name, calls, scores in [('I_cost', 252, 84), ('C_base_cost', 108, 12),
                                ('C_increment_cost', 24, 12), ('S_cost', 216, 24)]:
        x = arrays[name]
        # Unselected S score-CPU entries may be NaN; counts/tokens are still known.
        require(np.isfinite(x[..., :4]).all(), f'{name}: missing token/call/count cost')
        require((x[..., :4] >= 0).all() and (x[..., :4] == np.floor(x[..., :4])).all(),
                f'{name}: negative/noninteger count')
        require((x[..., 2] == calls).all() and (x[..., 3] == scores).all(),
                f'{name}: wrong fixed calls or score count')
        cpu = x[..., 4]
        require(not np.isinf(cpu).any() and (cpu[np.isfinite(cpu)] >= 0).all(),
                f'{name}: invalid scoring CPU cost')
        if name != 'S_cost':
            require(np.isfinite(cpu).all(), f'{name}: missing scoring CPU cost')


def budget(k, method):
    require(k in (1, 2) and method in METHODS, 'Unknown budget setting')
    n = 0 if method in ('source_only', 'current_stay') else 6 if method == 'full_revalidation_high_cost' else k
    initial = 0 if method == 'current_stay' else 12 * (75008 + 6 * 16768)
    audit = 12 * (75008 + n * 16768) if n else 0
    return {'initial_token_cap': initial, 'audit_token_cap': audit,
            'service_token_cap': 24 * 75008, 'total_token_cap': initial + audit + 24 * 75008,
            'score_count_cap': (0 if method == 'current_stay' else 84) + (12 * (1 + n) if n else 0) + 24,
            'counterfactual_increment_calls': 24 * n,
            'uniform_cap_scope': 'per_replica_not_sum_over_evaluator_enumeration',
            'actual_tokens_or_GPU_occupancy': None,
            'current_stay_scope': 'practical_graph_reference_without_I_not_matched_initialization'}


def rank_columns(values, editable):
    return sorted(range(6), key=lambda column: (-values[column], editable[column]))


def scalar_plans_and_actions(arrays):
    """I-only acquisition followed by purchased-column-only C updates; no S reads."""
    result = []
    for spec in graph_specs():
        e, edges = spec['episode'], spec['editable']
        columns = [[float(arrays['I_delta'][e, q, j]) for q in range(12)] for j in range(6)]
        gains = [mean(column) for column in columns]
        widths = [math.sqrt(sum((x - g) ** 2 for x in column) / 11 / 12) + 1 / 12
                  for g, column in zip(gains, columns)]
        incumbent = max(0., max(gains))
        gaps = [w / (incumbent - g + w) for g, w in zip(gains, widths)]
        for ci, condition in enumerate(CONDITIONS):
            dirty = spec['dirty'] if ci == 0 else [0] * 6
            ranks = {'candidate': rank_columns([(1 + d) * g for d, g in zip(dirty, gaps)], edges),
                     'no_graph_ablation': rank_columns(gaps, edges),
                     'max_uncertainty': rank_columns(widths, edges),
                     'dirty_only': rank_columns(dirty, edges)}
            require(ci == 0 or ranks['candidate'] == ranks['no_graph_ablation'], 'No-change ranking identity')
            for k in (1, 2):
                for method in METHODS:
                    if method == 'uniform_pair_expectation':
                        sets = list(itertools.combinations(range(6), k))
                    elif method in ranks:
                        sets = [ranks[method][:k]]
                    elif method == 'cyclic_pair':
                        sets = [[(e + j) % 6 for j in range(k)]]
                    elif method in ('source_only', 'current_stay'):
                        sets = [[]]
                    else:
                        sets = [list(range(6))]
                    for replica, purchased_columns in enumerate(sets):
                        updated = gains.copy()
                        for j in purchased_columns:
                            updated[j] = mean(float(arrays['C_delta'][e, ci, q, j]) for q in range(12))
                        winner = rank_columns(updated, edges)[0]
                        slot = 0 if method == 'current_stay' or updated[winner] <= 0 else 1 + winner
                        result.append({'plan_key': [e, condition, k, method, replica],
                                       'purchased': [edges[j] for j in purchased_columns],
                                       'purchased_columns': list(purchased_columns),
                                       'denominator': len(sets), 'updated_gain': updated,
                                       'action': None if slot == 0 else edges[slot - 1], 'slot': slot})
    require(len(result) == 592, 'Replica count differs from fixed 8-episode design')
    indexed = {tuple(row['plan_key']): row for row in result}
    for e in range(8):
        for k in (1, 2):
            a = indexed[e, 'no_change', k, 'candidate', 0]
            b = indexed[e, 'no_change', k, 'no_graph_ablation', 0]
            require(all(a[key] == b[key] for key in ('purchased', 'updated_gain', 'action')),
                    'No-change full-policy identity')
    return result


def bootstrap(values, indices):
    require(len(values) == 8 and all(math.isfinite(v) for v in values), 'Eight finite episode contrasts required')
    samples = sorted(sum(values[int(i)] for i in row) / 8 for row in indices)
    def quantile(p):
        position = (len(samples) - 1) * p
        low = math.floor(position)
        high = math.ceil(position)
        fraction = position - low
        return samples[low] * (1 - fraction) + samples[high] * fraction
    return {'mean': mean(values), 'ci95': [quantile(.025), quantile(.975)],
            'episode_values': list(values), 'resamples': 10000, 'seed': 69173,
            'unit': 'episode', 'independent_episode_count': 8,
            'scope': 'exploratory_eight_cluster_percentile_not_confirmation'}


def replay(arrays, *, with_bootstrap=True):
    validate_arrays(arrays)
    records = scalar_plans_and_actions(arrays)
    keys = [(condition, k, method) for condition in CONDITIONS for k in (1, 2) for method in METHODS]
    quality = {key: [[0.] * 24 for _ in range(8)] for key in keys}
    costs = {key: [[0.] * 5 for _ in range(8)] for key in keys}
    counts = {key: dict(stay=0., add=0., delete=0.) for key in keys}
    specs = graph_specs()
    replica_costs = []
    for row in records:
        e, condition, k, method, _ = row['plan_key']
        ci, slot = CONDITIONS.index(condition), row['slot']
        key, denominator = (condition, k, method), row['denominator']
        y = [float(arrays['S_quality'][e, ci, q, slot]) for q in range(24)]
        require(all(v in (0., 1.) for v in y), 'Chosen action has missing/nonbinary service score')
        actual = [float(x) for x in arrays['S_cost'][e, ci, slot]]
        if method != 'current_stay':
            actual = [a + float(b) for a, b in zip(actual, arrays['I_cost'][e])]
        if row['purchased_columns']:
            actual = [a + float(b) for a, b in zip(actual, arrays['C_base_cost'][e, ci])]
            for j in row['purchased_columns']:
                actual = [a + float(b) for a, b in zip(actual, arrays['C_increment_cost'][e, ci, j])]
        require(all(math.isfinite(v) and v >= 0 for v in actual), 'Chosen action has missing actual cost')
        cap = budget(k, method)
        require(actual[0] + actual[1] <= cap['total_token_cap'], 'Logical token cap exceeded')
        require(actual[3] == cap['score_count_cap'], 'Logical score quota differs')
        for q in range(24):
            quality[key][e][q] += y[q] / denominator
        for j in range(5):
            costs[key][e][j] += actual[j] / denominator
        mask = specs[e]['B' if ci == 0 else 'A']
        category = 'stay' if row['action'] is None else 'delete' if mask[row['action']] else 'add'
        counts[key][category] += 1 / denominator
        replica_costs.append({'plan_key': row['plan_key'], 'cost_vector': actual})
    methods, cost_rows, episode_means = [], [], {}
    for key in keys:
        condition, k, method = key
        episode_means[key] = [mean(questions) for questions in quality[key]]
        require(math.isclose(sum(counts[key].values()), 8., abs_tol=1e-12), 'Action weights do not sum to eight')
        methods.append({'condition': condition, 'k': k, 'method': method,
                        'accuracy': mean(episode_means[key]), 'episode_accuracy': episode_means[key],
                        'expected_episode_action_counts': counts[key], 'service_questions': 192, 'learning_episodes': 8})
        totals = [sum(row[j] for row in costs[key]) for j in range(5)]
        cost_rows.append({'condition': condition, 'k': k, 'method': method,
                          'mean_episode_actual_cost': [v / 8 for v in totals],
                          'all_eight_episode_actual_cost': totals, 'episode_actual_costs': costs[key],
                          'conservative_cap_per_episode': budget(k, method), 'same_actual_expenditure_claim': False})
    contrasts = {}
    indices = np.random.default_rng(69173).integers(0, 8, size=(10000, 8)) if with_bootstrap else None
    def describe(values):
        return bootstrap(values, indices) if with_bootstrap else {'mean': mean(values), 'episode_values': values}
    for k in (1, 2):
        for condition in CONDITIONS:
            candidate = episode_means[condition, k, 'candidate']
            for method in METHODS[1:]:
                values = [a - b for a, b in zip(candidate, episode_means[condition, k, method])]
                contrasts[f'{condition}/k{k}/candidate-minus-{method}'] = describe(values)
        values = [a - b for a, b in zip(
            contrasts[f'change/k{k}/candidate-minus-uniform_pair_expectation']['episode_values'],
            contrasts[f'no_change/k{k}/candidate-minus-uniform_pair_expectation']['episode_values'])]
        contrasts[f'change-minus-no_change/k{k}/candidate-minus-uniform'] = describe(values)
    require(len(methods) == 36 and len(contrasts) == 34, 'Summary coverage differs')
    summary = {'protocol_sha256': PROTOCOL, 'primary_key': PRIMARY, 'primary': contrasts[PRIMARY],
               'methods': methods, 'contrasts': contrasts, 'logical_policy_costs': cost_rows,
               'cost_vector_fields': list(FIELDS),
               'point_rule_satisfied': contrasts[PRIMARY]['mean'] >= .01 and
                   contrasts['change/k2/candidate-minus-source_only']['mean'] >= 0}
    return {'summary': summary, 'records': records, 'replica_costs': replica_costs}


class Comparator:
    def __init__(self):
        self.checks = 0
        self.numeric_checks = 0
        self.max_absolute_error = 0.
        self.mismatches = []

    def compare(self, expected, observed, path='root', *, absolute=1e-12, relative=1e-12):
        self.checks += 1
        if isinstance(expected, dict):
            if not isinstance(observed, dict) or set(expected) != set(observed):
                self.mismatches.append(path + ': dictionary keys differ')
                return
            for key in expected:
                self.compare(expected[key], observed[key], path + '/' + str(key), absolute=absolute, relative=relative)
        elif isinstance(expected, (list, tuple)):
            if not isinstance(observed, (list, tuple)) or len(expected) != len(observed):
                self.mismatches.append(path + ': sequence shape differs')
                return
            for i, (a, b) in enumerate(zip(expected, observed)):
                self.compare(a, b, f'{path}/{i}', absolute=absolute, relative=relative)
        elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
            self.numeric_checks += 1
            if not isinstance(observed, (float, int)) or isinstance(observed, bool) or not math.isfinite(observed):
                self.mismatches.append(path + ': nonnumeric/nonfinite value')
            else:
                error = abs(expected - observed)
                self.max_absolute_error = max(self.max_absolute_error, error)
                if not math.isclose(expected, observed, abs_tol=absolute, rel_tol=relative):
                    self.mismatches.append(path + ': numeric mismatch')
        elif expected != observed:
            self.mismatches.append(path + ': exact value mismatch')


def compare_result(rebuilt, official, purchase_freeze, action_freeze, official_replica_costs):
    c = Comparator()
    for field in ('protocol_sha256', 'primary_key', 'primary', 'methods', 'contrasts',
                  'cost_vector_fields', 'point_rule_satisfied'):
        c.compare(rebuilt['summary'][field], official[field], field)
    c.compare(rebuilt['summary']['logical_policy_costs'], official['logical_policy_costs'],
              'logical_policy_costs', absolute=1e-8, relative=1e-12)
    c.compare(rebuilt['replica_costs'], official_replica_costs['replicas'],
              'replica_costs', absolute=1e-8, relative=1e-12)
    c.compare(list(FIELDS), official_replica_costs['cost_fields'], 'replica_cost_fields')
    plans = purchase_freeze['payload']['plans']
    records = action_freeze['payload']['records']
    c.compare(592, len(plans), 'purchase_count', absolute=0, relative=0)
    c.compare(592, len(records), 'action_count', absolute=0, relative=0)
    require(len(plans) == len(records) == len(rebuilt['records']), 'Frozen replica coverage mismatch')
    for index, (ours, plan, record) in enumerate(zip(rebuilt['records'], plans, records)):
        e, condition, k, method, replica = ours['plan_key']
        expected_plan = {'episode': e, 'condition': condition, 'k': k, 'method': method,
                         'replica': replica, 'purchased': ours['purchased'],
                         'weight_numerator': 1, 'weight_denominator': ours['denominator']}
        c.compare(expected_plan, plan, f'purchase/{index}', absolute=0, relative=0)
        c.compare(expected_plan, record['plan'], f'action_plan/{index}', absolute=0, relative=0)
        c.compare(ours['updated_gain'], record['updated_gain'], f'updated_gain/{index}')
        c.compare(ours['action'], record['action'], f'action/{index}', absolute=0, relative=0)
        provenance = [('current_B' if condition == 'change' else 'current_A')
                      if j in ours['purchased_columns'] else 'source_A' for j in range(6)]
        c.compare(provenance, record['cell_provenance'], f'provenance/{index}')
    return {'status': 'PASS' if not c.mismatches else 'FAIL', 'checks': c.checks,
            'numeric_checks': c.numeric_checks, 'max_absolute_error': c.max_absolute_error,
            'mismatches': c.mismatches, 'replicas': 592, 'method_rows': 36, 'contrasts': 34,
            'new_model_calls': 0, 'bootstrap_resamples': 10000, 'bootstrap_clusters': 8,
            'scope': 'independent scalar replay of already-scored development pilot; not independent scientific confirmation',
            'limits': ['Does not verify raw questions, gold correctness, scorer implementation, or raw graph execution.',
                       'Physical GPU accounting must be merged independently by the root budget ledger.',
                       'Scalar arithmetic is independent; bootstrap index generator is the protocol-specified NumPy generator.',
                       'Rank ties follow exact floating values and physical-edge ordering; disagreements fail rather than relax ties.']}


def load_bound_inputs(analysis_dir):
    """Read only completion/public metadata before allowing the private NPZ load."""
    root = Path(analysis_dir)
    private = root / 'private'
    receipt_bytes = (private / 'COMPLETE.json').read_bytes()
    receipt = json.loads(receipt_bytes)
    require(receipt['status'] == 'COMPLETE_DEVELOPMENT_ANALYSIS_NOT_CONFIRMATION', 'Official scoring incomplete')
    require(receipt['source_archive_unchanged'] is True and receipt['S_released_after_all_actions_frozen'] is True
            and receipt['new_model_calls'] == 0, 'Official release guard not satisfied')
    result_bytes = (root / 'public' / 'R1_RESULTS.json').read_bytes()
    require(digest_bytes(result_bytes) == receipt['public_result_sha256'], 'Official result hash mismatch')
    result = json.loads(result_bytes)
    require(result['protocol_sha256'] == PROTOCOL, 'Wrong frozen scientific protocol')
    metadata = result['provenance']
    require(metadata['replay_inputs_sha256'] == receipt['replay_inputs_sha256'], 'Replay hash receipts disagree')
    freezes = []
    for filename, file_field, digest_field in [('PURCHASES_FROZEN.json', 'purchases_file_sha256', 'purchase_digest'),
                                              ('ACTIONS_FROZEN.json', 'actions_file_sha256', 'action_digest')]:
        raw = (private / filename).read_bytes()
        require(digest_bytes(raw) == metadata[file_field], f'{filename}: frozen file hash mismatch')
        value = json.loads(raw)
        require(payload_digest(value['payload']) == value['digest'] == result[digest_field],
                f'{filename}: payload binding mismatch')
        freezes.append(value)
    require(freezes[1]['payload']['purchase_digest'] == freezes[0]['digest'], 'Action-purchase link mismatch')
    replay_path = private / 'REPLAY_INPUTS.npz'
    replay_bytes = replay_path.read_bytes()
    require(digest_bytes(replay_bytes) == receipt['replay_inputs_sha256'], 'NPZ bytes not bound to completed scoring')
    # Load exactly the verified bytes, avoiding a second path read/time-of-check race.
    import io
    with np.load(io.BytesIO(replay_bytes), allow_pickle=False) as bundle:
        arrays = {key: bundle[key].copy() for key in bundle.files}
    replica_costs = json.loads((private / 'REPLICA_LOGICAL_COSTS.json').read_bytes())
    return arrays, result, freezes[0], freezes[1], replica_costs, {
        'complete_receipt_sha256': digest_bytes(receipt_bytes),
        'official_result_sha256': digest_bytes(result_bytes),
        'replay_inputs_sha256': digest_bytes(replay_bytes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Do not overwrite an existing replay audit')
    arrays, official, purchases, actions, replica_costs, pins = load_bound_inputs(args.analysis_dir)
    rebuilt = replay(arrays)
    report = compare_result(rebuilt, official, purchases, actions, replica_costs)
    report.update(pins)
    report['replay_source_sha256'] = digest_bytes(Path(__file__).read_bytes())
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'checks': report['checks'],
                      'numeric_checks': report['numeric_checks'],
                      'max_absolute_error': report['max_absolute_error'],
                      'mismatch_count': len(report['mismatches'])}, ensure_ascii=False))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
