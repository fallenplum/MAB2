"""Fabricated schema fixture, never official results; no data file reader."""
import numpy as np
import reporting as r


def make_fixture():
    i = np.arange(r.N)
    graphs = np.zeros((r.N, 2, 4), dtype=int)
    for seed in range(2):
        graphs[:, seed, 0] = ((i + seed) % 4 < 2)
        graphs[:, seed, 1] = ((i + 2 * seed) % 6 < 3)
        graphs[:, seed, 2] = ((i + seed) % 5 < 2)
        graphs[:, seed, 3] = ((i + seed) % 7 < 2)
    graphs[:48, 0, :] = [0, 1, 0, 0]
    graphs[:48, 1, :] = [1, 0, 0, 0]  # Explicit negative cross-seed products.
    em = {'graphs': graphs.tolist(), 'no_edge': (i % 3 == 0).astype(int).tolist(),
        'independent12': (i % 4 == 0).astype(int).tolist(), 'single': (i % 2 == 0).astype(int).tolist()}
    f1 = {}
    for key, values in em.items():
        a = np.asarray(values, dtype=float)
        f1[key] = np.where(a == 1, 1.0, 0.25).tolist()
    statuses = []
    for index in range(r.N):
        for condition in r.CONDITIONS:
            rejected = (index, condition) in ((0, 'S1/SINGLE_M'), (1, 'S2/B'))
            statuses.append({'selection_index': index, 'condition_key': condition,
                'status': 'INPUT_REJECTED_ZERO' if rejected else 'SCORED_STRICT_AGGREGATE'})
    em['single'][0] = f1['single'][0] = 0
    em['graphs'][1][1][1] = f1['graphs'][1][1][1] = 0
    cost = {}
    for j, name in enumerate(('B', 'NO_EDGE', 'INDEPENDENT12', 'SINGLE_M')):
        cost[name] = {}
        for model in r.MODELS:
            calls = 0 if name == 'SINGLE_M' and model == 'qwen' else (384 if name == 'SINGLE_M' else 2304 if model == 'qwen' else 2688)
            row = {'calls': calls, 'input_tokens': calls * (200 + j * 10),
                'output_tokens': calls * 32, 'CUDA_event_seconds': calls * 0.05}
            if name == 'SINGLE_M' and model == 'mistral': row = dict.fromkeys(r.COST_FIELDS, None)
            cost[name][model] = row
    # These are fabricated schema values. No bootstrap or scientific analysis is
    # run; the fixture tests that reporting copies them instead of recomputing.
    points = ((.025, -.015, .065), (.012, -.025, .050), (-.010, -.045, .025), (-.008, -.025, .010))
    primary = {}
    for (_, _, name), (estimate, lower, upper) in zip(r.PRIMARY, points):
        primary[name] = {'estimate': estimate, 'interval': [lower, upper], 'nonzero_questions': 160,
            'positive_questions': 72, 'negative_questions': 88, 'sample_variance': .2,
            'bootstrap_estimator_variance': .0005, 'bootstrap_nondegenerate': True,
            'support_guard_pass': True, 'interval_width': upper - lower}
    return {'synthetic_fixture': True, 'synthetic_marker': 'SYNTHETIC_SCHEMA_DEMONSTRATION_NOT_SCIENTIFIC_RESULTS',
        'status': 'COMPLETE_WITH_CAPACITY_FAILURES', 'array_axes': {'rows': 'exact selection manifest order',
            'graphs': list(r.GRAPH_NAMES), 'seeds': [26111301, 26111302]},
        'em': em, 'f1': f1, 'edge_indices': [e for e in (1, 5, 7, 9) for _ in range(96)],
        'condition_statuses': statuses, 'format_failure_conditions': 5, 'full_logical_costs_by_model': cost,
        'frontier': {'status': 'NOT_ASSESSED', 'reason': 'incomplete_or_capacity_or_integrity_evidence'},
        'audit_inputs': {'integrity_passed': True, 'unique_questions': 384, 'complete_conditions': 4224,
            'runtime_failure_count': 0, 'input_overflow_conditions': 2, 'cost_frontier_review': 'NOT_ASSESSED'},
        'statistics': {'primary_EM': primary, 'decision': {'L1': 'INCONCLUSIVE', 'L2': 'INCONCLUSIVE',
            'signed_direction': 'INCONCLUSIVE', 'theta_nonzero_supported': False, 'theta_negative_point_retained': True,
            'theta_small_effect_exclusion': 'NOT_PERMITTED_BY_THIS_BOOTSTRAP_CONTRACT',
            'theta_width_divided_by_0_0025': 14., 'action': 'NO_AUTOMATIC_UPGRADE_CAPACITY_FAILURES',
            'GPU_admission_granted': False, 'method_success_claim': False},
            'bootstrap': {'repetitions': 10000, 'family_size': 4, 'family_alpha': .05,
                'draw_index_sha256': 'a' * 64}},
        'private_per_question_arrays': True, 'scientific_or_GPU_admission_granted': False,
        'bindings': dict(r.FROZEN_PINS), 'archive_review_sha256': 'b' * 64,
        'physical_research_ledger': {'archive_metadata_sha256': {name: str(j) * 64 for j, name in enumerate(
            ('RAW_OUTPUT_MANIFEST.json', 'PARENT_RESIDENCE.json', 'ATTEMPT_RESULT.json'), 1)},
            'recomputed_as_logical_cost': False},
        'private_extra': {'qid': 'hotpotqa_PRIVATE_SYNTHETIC_SENTINEL', 'prompt': 'PRIVATE_PROMPT_SENTINEL',
            'answer': 'PRIVATE_ANSWER_SENTINEL', 'hash': 'c' * 64}}
