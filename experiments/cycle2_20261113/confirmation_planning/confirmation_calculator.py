"""Assumption-only confirmation planning. No dataset/result access or GPU calls.

Normal approximations are feasibility calculations, not validated small-cluster
confidence intervals. Inputs are assumed variance components, not observations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

NORMAL = NormalDist()
GSM8K_TRAIN_OFFICIAL_UPPER = 7473
GSM8K_TEST_OFFICIAL_UPPER = 1319

def finite(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(name + ' must be a finite number')
    return float(x)

def count(x, name, allow_zero=False):
    if isinstance(x, bool) or not isinstance(x, int) or x < (0 if allow_zero else 1):
        raise ValueError(name + ' must be an integer in range')
    return x

def unit(x, name):
    x = finite(x, name)
    if not 0 <= x <= 1:
        raise ValueError(name + ' must lie in [0,1]')
    return x

def nonnegative(x, name):
    x = finite(x, name)
    if x < 0:
        raise ValueError(name + ' must be nonnegative')
    return x

def alpha_value(alpha):
    alpha = finite(alpha, 'alpha')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between 0 and 1')
    return alpha

def paired_variance(effect, discordance):
    effect = finite(effect, 'effect')
    discordance = unit(discordance, 'discordance')
    if abs(effect) > discordance:
        raise ValueError('For binary pairs, abs(effect) cannot exceed discordance')
    return max(0.0, discordance-effect*effect)

def normal_summary(effect, mean_variance, alpha=0.05):
    effect = finite(effect, 'effect')
    variance = nonnegative(mean_variance, 'mean_variance')
    alpha = alpha_value(alpha)
    if variance == 0:
        return {'se': 0.0, 'normal_95_or_alpha_halfwidth': 0.0,
                'two_sided_normal_power': None, 'status': 'DEGENERATE_VARIANCE_NOT_A_POWER_GUARANTEE', 'alpha': alpha}
    se = math.sqrt(variance)
    z = NORMAL.inv_cdf(1-alpha/2)
    shift = abs(effect)/se
    power = NORMAL.cdf(-z-shift) + 1-NORMAL.cdf(z-shift)
    return {'se': se, 'normal_95_or_alpha_halfwidth': z*se,
            'two_sided_normal_power': power, 'status': 'ASSUMPTION_ONLY_NORMAL_APPROXIMATION', 'alpha': alpha}

def conditional_pair(n_questions, effect, discordance, alpha=0.05):
    n = count(n_questions, 'n_questions')
    v = paired_variance(effect, discordance)
    return {'unique_questions': n, 'assumed_effect': effect, 'assumed_discordance': discordance,
            'paired_variance': v, 'expected_discordant_pairs': n*discordance,
            'small_discordance_warning': n*discordance <= 20,
            **normal_summary(effect, v/n, alpha)}

def required_questions_normal(effect, discordance, target_power=0.8, alpha=0.05, max_n=10_000_000):
    """Numerically invert this approximation; not an exact McNemar sample size."""
    paired_variance(effect, discordance)
    alpha = alpha_value(alpha)
    p = unit(target_power, 'target_power')
    count(max_n, 'max_n')
    if not alpha < p < 1:
        raise ValueError('target_power must be strictly between alpha and 1')
    if effect == 0 or paired_variance(effect, discordance) == 0:
        return None
    def enough(n):
        return conditional_pair(n, effect, discordance, alpha)['two_sided_normal_power'] >= p
    if not enough(max_n):
        return None
    lo, hi = 1, max_n
    while lo < hi:
        mid = (lo+hi)//2
        if enough(mid): hi = mid
        else: lo = mid+1
    return lo

def episode_design(episodes, questions_per_episode, effect, discordance, between_episode_sd, alpha=0.05):
    e = count(episodes, 'episodes')
    m = count(questions_per_episode, 'questions_per_episode')
    v = paired_variance(effect, discordance)
    tau2 = nonnegative(between_episode_sd, 'between_episode_sd')**2
    if tau2 > v+1e-15:
        raise ValueError('Between-episode variance exceeds marginal paired variance')
    within = max(0.0, v-tau2)
    variance = tau2/e + within/(e*m)
    return {'episodes': e, 'questions_per_episode': m, 'unique_questions': e*m,
            'assumed_effect': effect, 'assumed_discordance': discordance,
            'assumed_between_episode_sd': between_episode_sd,
            'mean_variance': variance, 'infinite_questions_mean_variance_floor': tau2/e,
            'icc': tau2/v if v else None,
            'effective_iid_question_count': v/variance if variance else None,
            'warning': '8 or 5 clusters do not validate normal/bootstrap interval coverage; bank independence is an assumption.',
            **normal_summary(effect, variance, alpha)}

def crossed_seed_design(n_questions, seeds, effect, discordance, question_share, seed_share, alpha=0.05):
    n = count(n_questions, 'n_questions')
    s = count(seeds, 'seeds')
    q = unit(question_share, 'question_share')
    r = unit(seed_share, 'seed_share')
    if q+r > 1:
        raise ValueError('Variance shares must sum to at most one')
    v = paired_variance(effect, discordance)
    residual = 1-q-r
    variance = v*(q/n+r/s+residual/(n*s))
    return {'unique_questions': n, 'seeds': s, 'total_paired_evaluations': n*s,
            'assumed_question_variance_share': q, 'assumed_seed_variance_share': r,
            'assumed_residual_variance_share': residual, 'mean_variance': variance,
            'infinite_questions_seed_variance_floor': v*r/s,
            'warning': 'These crossed random components require independent sampled seed effects; same bank does not create fresh learning-bank replicates.',
            **normal_summary(effect, variance, alpha)}

def condition_interaction(n_questions, effect_difference, variance_a, variance_b, condition_correlation, alpha=0.05):
    n = count(n_questions, 'n_questions')
    va = nonnegative(variance_a, 'variance_a')
    vb = nonnegative(variance_b, 'variance_b')
    r = finite(condition_correlation, 'condition_correlation')
    if not -1 <= r <= 1:
        raise ValueError('Correlation must be in [-1,1]')
    v = max(0.0, va+vb-2*r*math.sqrt(va*vb))
    return {'unique_questions': n, 'assumed_condition_correlation': r,
            'interaction_pair_variance': v,
            'warning': 'DID is in [-2,2], not a binary McNemar variable. Moment grid does not assert every binary joint distribution is feasible.',
            **normal_summary(effect_difference, v/n, alpha)}

def role_capacity(source_upper, exposure_sets, reserve_count=0):
    """Sets must contain source-internal canonical content hashes only."""
    upper = count(source_upper, 'source_upper')
    reserve = count(reserve_count, 'reserve_count', True)
    union = set()
    for values in exposure_sets:
        values = set(values)
        if not all(isinstance(x, str) for x in values):
            raise ValueError('Exposure IDs must be strings')
        union.update(values)
    if len(union)+reserve > upper:
        raise ValueError('Exclusions/reserves exceed source capacity')
    return {'source_upper': upper, 'exposure_union_count': len(union), 'reserved_unexposed': reserve,
            'available_upper_after_known_exclusions': upper-len(union)-reserve,
            'warning': 'Unknown exposure, unresolved duplicates and source membership require split-owner acceptance before this is a valid available count.'}

def validate_role_hashes(role_to_hashes, prior_exposed_hashes=()):
    """Metadata-only check. Does not assign, accept, or open any questions."""
    if not isinstance(role_to_hashes, dict) or not role_to_hashes:
        raise ValueError('Require nonempty role map')
    seen = set(prior_exposed_hashes)
    counts = {}
    for role, values in role_to_hashes.items():
        if not isinstance(role, str): raise ValueError('Role must be a string')
        seq = list(values)
        current = set(seq)
        if len(current) != len(seq): raise ValueError('Duplicate within role')
        if current & seen: raise ValueError('Overlap between roles or prior exposures')
        if not all(isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x) for x in current):
            raise ValueError('Require canonical lowercase SHA256 strings')
        seen.update(current)
        counts[role] = len(current)
    return {'counts': counts, 'hashes_disjoint_from_supplied_exposures': True,
            'scope': 'Only checks supplied metadata, not completeness of exposure history or near-duplicates.'}

def gsm8k_test_capacity(source_internal_exposure_sets=(), reserve_count=0, requested=None):
    result = role_capacity(GSM8K_TEST_OFFICIAL_UPPER, source_internal_exposure_sets, reserve_count)
    if requested is not None:
        count(requested, 'requested', True)
        result['requested'] = requested
        result['fits_known_upper_bound'] = requested <= result['available_upper_after_known_exclusions']
    return result

def budget(used_gpu_hours, unsettled_reserved_gpu_hours, other_reserved_gpu_hours, proposed_gpu_hours, total_gpu_hours=160.0):
    used = nonnegative(used_gpu_hours, 'used_gpu_hours')
    unsettled = nonnegative(unsettled_reserved_gpu_hours, 'unsettled_reserved_gpu_hours')
    other = nonnegative(other_reserved_gpu_hours, 'other_reserved_gpu_hours')
    proposed = nonnegative(proposed_gpu_hours, 'proposed_gpu_hours')
    total = nonnegative(total_gpu_hours, 'total_gpu_hours')
    remaining = total-used-unsettled-other
    return {'total_gpu_hours': total, 'settled_gpu_hours': used, 'unsettled_reserved_gpu_hours': unsettled,
            'other_reserved_gpu_hours': other, 'available_before_proposal_gpu_hours': max(0.0, remaining),
            'overcommitted_before_proposal': remaining < 0, 'proposed_gpu_hours': proposed,
            'fits_assumed_ledger': remaining >= proposed,
            'warning': 'Only valid when all terms are disjoint current full-ledger charges; concurrency cannot divide GPU hours.'}

def call_budget(tasks, n_unique_service_per_task, service_seeds, service_union_calls_per_question_seed,
                learning_replicates_per_task, calls_per_learning_replicate,
                gpu_seconds_per_call, fixed_overhead_gpu_hours=0.0, failure_reserve_fraction=0.0):
    tasks = count(tasks, 'tasks')
    nq = count(n_unique_service_per_task, 'n_unique_service_per_task')
    seeds = count(service_seeds, 'service_seeds')
    menu = count(service_union_calls_per_question_seed, 'service_union_calls_per_question_seed')
    lr = count(learning_replicates_per_task, 'learning_replicates_per_task', True)
    lc = count(calls_per_learning_replicate, 'calls_per_learning_replicate', True)
    sec = nonnegative(gpu_seconds_per_call, 'gpu_seconds_per_call')
    if sec == 0: raise ValueError('GPU seconds per call must be positive')
    fixed = nonnegative(fixed_overhead_gpu_hours, 'fixed_overhead_gpu_hours')
    fail = nonnegative(failure_reserve_fraction, 'failure_reserve_fraction')
    service_calls = tasks*nq*seeds*menu
    learning_calls = tasks*lr*lc
    inference = (service_calls+learning_calls)*sec/3600
    return {'tasks': tasks, 'unique_service_questions_per_task': nq, 'service_seeds': seeds,
            'learning_replicates_per_task': lr, 'physical_service_calls': service_calls,
            'physical_learning_calls': learning_calls, 'assumed_gpu_seconds_per_call': sec,
            'inference_gpu_hours': inference, 'fixed_overhead_gpu_hours': fixed,
            'failure_reserve_fraction': fail,
            'estimated_total_gpu_hours': (inference+fixed)*(1+fail),
            'status': 'HYPOTHETICAL_COST_NOT_MEASURED_THROUGHPUT',
            'warning': 'No concurrent-card walltime division; shared-menu reuse needs exact identity and measured backend validation. Logical deployment costs separate.'}

def scenario_grid():
    conditional = []
    for effect in (0.01, 0.02, 0.03, 0.05):
        for d in (0.05, 0.1, 0.2, 0.4):
            for n in (192, 400, 800, 1319):
                row = conditional_pair(n, effect, d)
                row['normal_n_for_80pct_power'] = required_questions_normal(effect, d)
                conditional.append(row)
    episode = [episode_design(e, m, .02, .1, sd) for e in (5, 8, 16, 24)
               for m in (24, 48, 96) for sd in (0, .02, .05, .1)]
    seeds = [crossed_seed_design(n, s, .02, .1, q, r)
             for n in (192, 400, 800, 1319) for s in (1, 5, 8)
             for q, r in ((0, 0), (.5, 0), (1, 0), (.5, .01), (.5, .05))]
    costs = [call_budget(2, n, 5, menu, 5, 696, sec, 1, .15)
             for n in (192, 400, 800, 1319) for menu in (9, 21, 37)
             for sec in (.25, 1, 4, 10)]
    return {'schema': 'mab2_confirmation_assumption_grid_v1', 'observed_experiment_data': False,
            'final_confirmation_numbers_frozen': False, 'gpu_started': False,
            'official_upper_limits': {'gsm8k_train': 7473, 'gsm8k_test': 1319,
                'socratic_is_not_additional_independent_questions': True},
            'conditional_binary_pairs': conditional, 'independent_learning_episode_moment_grid': episode,
            'crossed_question_seed_moment_grid': seeds, 'hypothetical_cost_grid': costs,
            'cost_grid_assumptions': {'tasks': 2, 'service_seeds': 5, 'learning_replicates_per_task': 5,
                'calls_per_learning_replicate': 696, 'fixed_gpu_hours': 1, 'failure_reserve_fraction': .15,
                'warning': 'All throughput/overhead assumptions are invented planning inputs, not R1 measurements. Menu 9 only fits one graph, 21 one background full menu, 37 paired A/B full menu for current qualified geometry.'},
            'current_budget_snapshot': None, 'current_accepted_unexposed_counts': None}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit('Refuse overwrite')
    report = scenario_grid()
    report['calculator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(args.output), 'observed_experiment_data': False,
                      'scenario_rows': sum(len(report[k]) for k in ('conditional_binary_pairs','independent_learning_episode_moment_grid','crossed_question_seed_moment_grid','hypothetical_cost_grid'))}))

if __name__ == '__main__':
    main()
