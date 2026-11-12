"""Frozen R1 policy and exploratory statistics. No filesystem or data discovery.

The adapter must persist each returned freeze before opening the next stage.
Capabilities prevent accidental unpurchased reads through this API; Python
callbacks are not an OS security boundary. Keep hidden arrays in the adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations
import json
from math import comb
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

PROTOCOL_SHA256 = '5336eacc6713d4e444ddb631032409cd2718ad3b4aaee1387aff5baac40f5ecf'
CONDITIONS = ('change', 'no_change')
METHODS = ('candidate', 'uniform_pair_expectation', 'cyclic_pair',
           'max_uncertainty', 'no_graph_ablation', 'dirty_only',
           'source_only', 'current_stay', 'full_revalidation_high_cost')
EDGES = tuple((i, j) for i in range(4) for j in range(4) if i != j)
K_VALUES = (1, 2)


def _digest(value) -> str:
    if hasattr(value, '__dataclass_fields__'):
        value = asdict(value)
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode('utf-8')).hexdigest()


def _observations(value, shape, allowed, label):
    out = np.array(value, dtype=np.float64, copy=True)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f'{label}: expected finite shape {shape}, got {out.shape}')
    if not np.isin(out, allowed).all():
        raise ValueError(f'{label}: values must be in {allowed}')
    return out


@dataclass(frozen=True)
class EpisodeSpec:
    episode: int
    seed: int
    shift: int
    complement: int
    editable: tuple[int, ...]
    changed: tuple[int, ...]
    A: tuple[int, ...]
    B: tuple[int, ...]
    dirty: tuple[int, ...]


def canonical_episode_specs() -> tuple[EpisodeSpec, ...]:
    """Independent graph construction from the frozen template and rotations."""
    base_edit = (0, 3, 4, 6, 8, 9)
    base_changed = (1, 5, 10)
    base_on = {0, 3, 6}
    result = []
    for shift in range(4):
        permutation = {e: EDGES.index(((src + shift) % 4, (dst + shift) % 4))
                       for e, (src, dst) in enumerate(EDGES)}
        editable = tuple(sorted(permutation[e] for e in base_edit))
        changed = tuple(sorted(permutation[e] for e in base_changed))
        receivers = {EDGES[e][1] for e in changed}
        for complement in (0, 1):
            A = [1] * 12
            for e in base_edit:
                A[permutation[e]] = int((e in base_on) != bool(complement))
            B = A.copy()
            for e in changed:
                B[e] = 0
            episode = 2 * shift + complement
            result.append(EpisodeSpec(episode, 3701 + episode, shift, complement,
                                     editable, changed, tuple(A), tuple(B),
                                     tuple(int(EDGES[e][1] in receivers) for e in editable)))
    return tuple(result)


def specs_from_protocol_bytes(raw: bytes) -> tuple[EpisodeSpec, ...]:
    if sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('Frozen protocol SHA mismatch; no silent amendment')
    p = json.loads(raw)
    actual = tuple(EpisodeSpec(x['episode'], x['seed'], x['node_cyclic_shift'],
                              x['editable_complement'], tuple(x['editable_edges']),
                              tuple(x['changed_background_edges']), tuple(x['A']),
                              tuple(x['B']), tuple(x['dirty_by_editable']))
                   for x in p['data']['episodes'])
    if actual != canonical_episode_specs():
        raise ValueError('Frozen graph specification differs from independent construction')
    for x in p['data']['episodes']:
        if (x['I_questions'], x['C_questions'], x['S_questions']) != (12, 12, 24):
            raise ValueError('Frozen question count mismatch')
    return actual


def align_initial_columns(by_physical_edge: Mapping[int, Sequence[float]],
                          spec: EpisodeSpec) -> np.ndarray:
    """I adapter helper: map request.edge, NEVER request order or template name."""
    if set(by_physical_edge) != set(spec.editable):
        raise ValueError('I edge keys must exactly equal physical editable edges')
    return np.column_stack([_observations(by_physical_edge[e], (12,), (-1, 0, 1),
                                          'I physical-edge column') for e in spec.editable])


@dataclass(frozen=True)
class InitialState:
    episode: int
    editable: tuple[int, ...]
    gain: tuple[float, ...]
    width: tuple[float, ...]
    incumbent: float
    I_deltas_sha256: str


@dataclass(frozen=True)
class ReplicaPlan:
    episode: int
    condition: str
    k: int
    method: str
    replica: int
    purchased: tuple[int, ...]
    weight_numerator: int
    weight_denominator: int

    @property
    def key(self):
        return (self.episode, self.condition, self.k, self.method, self.replica)


@dataclass(frozen=True)
class PurchaseFreeze:
    protocol_sha256: str
    specs: tuple[EpisodeSpec, ...]
    initial: tuple[InitialState, ...]
    plans: tuple[ReplicaPlan, ...]

    @property
    def digest(self):
        return _digest(self)


def _rank(values, edges):
    return tuple(sorted(edges, key=lambda e: (-float(values[edges.index(e)]), e)))


def freeze_purchases(initial_deltas: Mapping[int, np.ndarray],
                     specs: Sequence[EpisodeSpec]) -> PurchaseFreeze:
    """I-only stage. Accepts no C callback, S values, or future cost arrays."""
    specs = tuple(specs)
    if specs != canonical_episode_specs() or set(initial_deltas) != set(range(8)):
        raise ValueError('Require all eight fixed graph episodes and exact I keys')
    states, plans = [], []
    for spec in specs:
        d = _observations(initial_deltas[spec.episode], (12, 6), (-1, 0, 1), 'I action differences')
        g = d.mean(axis=0)
        w = np.sqrt(d.var(axis=0, ddof=1) / 12) + 1 / 12
        b = float(max(0., float(g.max())))
        gap = w / (b - g + w)
        if not (np.isfinite(gap).all() and (gap > 0).all() and (gap <= 1).all()):
            raise ValueError('Incumbent gap invariant failed')
        states.append(InitialState(spec.episode, spec.editable,
                                   tuple(map(float, g)), tuple(map(float, w)), b,
                                   _digest(d.tolist())))
        for condition in CONDITIONS:
            dirty = np.array(spec.dirty if condition == 'change' else (0,) * 6)
            rankings = {'candidate': _rank((1 + dirty) * gap, spec.editable),
                        'max_uncertainty': _rank(w, spec.editable),
                        'no_graph_ablation': _rank(gap, spec.editable),
                        'dirty_only': _rank(dirty, spec.editable)}
            if condition == 'no_change' and rankings['candidate'] != rankings['no_graph_ablation']:
                raise AssertionError('no-change acquisition identity failed')
            for k in K_VALUES:
                for method in METHODS:
                    if method == 'uniform_pair_expectation':
                        subsets = tuple(combinations(spec.editable, k))
                    elif method in rankings:
                        subsets = (rankings[method][:k],)
                    elif method == 'cyclic_pair':
                        start = spec.episode % 6
                        subsets = (tuple(spec.editable[(start + j) % 6] for j in range(k)),)
                    elif method in ('source_only', 'current_stay'):
                        subsets = ((),)
                    elif method == 'full_revalidation_high_cost':
                        subsets = (spec.editable,)
                    else:
                        raise AssertionError(method)
                    for idx, subset in enumerate(subsets):
                        plans.append(ReplicaPlan(spec.episode, condition, k, method, idx,
                                                 tuple(subset), 1, len(subsets)))
    result = PurchaseFreeze(PROTOCOL_SHA256, specs, tuple(states), tuple(plans))
    _validate_purchase_freeze(result)
    return result


def _validate_purchase_freeze(freeze):
    if freeze.protocol_sha256 != PROTOCOL_SHA256 or freeze.specs != canonical_episode_specs():
        raise ValueError('Purchase protocol/spec mismatch')
    if len(freeze.initial) != 8 or tuple(s.episode for s in freeze.initial) != tuple(range(8)):
        raise ValueError('Missing initial states')
    for state, spec in zip(freeze.initial, freeze.specs):
        if state.editable != spec.editable or len(state.gain) != 6 or len(state.width) != 6:
            raise ValueError('Initial table edge mapping invalid')
        if not np.isfinite(state.gain).all() or not np.isfinite(state.width).all():
            raise ValueError('Initial table nonfinite')
        if any(abs(g) > 1 for g in state.gain) or any(w < 1 / 12 for w in state.width):
            raise ValueError('Initial table gain or width outside frozen bounds')
        if state.incumbent != max(0., max(state.gain)):
            raise ValueError('Wrong incumbent')
    seen = set()
    for plan in freeze.plans:
        if plan.key in seen:
            raise ValueError('Duplicate replica plan')
        seen.add(plan.key)
        if plan.condition not in CONDITIONS or plan.method not in METHODS or plan.k not in K_VALUES:
            raise ValueError('Unknown condition/method/quota')
        if plan.episode not in range(8):
            raise ValueError('Unknown episode')
        editable = freeze.specs[plan.episode].editable
        if len(set(plan.purchased)) != len(plan.purchased) or not set(plan.purchased) <= set(editable):
            raise ValueError('Illegal purchases')
        expected_n = (0 if plan.method in ('source_only', 'current_stay') else
                      6 if plan.method == 'full_revalidation_high_cost' else plan.k)
        denominator = comb(6, plan.k) if plan.method == 'uniform_pair_expectation' else 1
        if len(plan.purchased) != expected_n or (plan.weight_numerator, plan.weight_denominator) != (1, denominator):
            raise ValueError('Purchase cardinality or replica weight mismatch')
        state = freeze.initial[plan.episode]
        g, w = np.array(state.gain), np.array(state.width)
        gap = w / (state.incumbent - g + w)
        dirty = np.array(freeze.specs[plan.episode].dirty if plan.condition == 'change' else (0,) * 6)
        rankings = {'candidate': _rank((1 + dirty) * gap, editable),
                    'no_graph_ablation': _rank(gap, editable),
                    'max_uncertainty': _rank(w, editable),
                    'dirty_only': _rank(dirty, editable)}
        if plan.method in rankings and plan.purchased != rankings[plan.method][:plan.k]:
            raise ValueError('Purchase ranking differs from frozen I-only rule')
        if plan.method == 'cyclic_pair' and plan.purchased != tuple(editable[(plan.episode + j) % 6] for j in range(plan.k)):
            raise ValueError('Cyclic purchase differs from frozen rule')
    expected = {(e, c, k, m, j) for e in range(8) for c in CONDITIONS for k in K_VALUES
                for m in METHODS for j in range(comb(6, k) if m == 'uniform_pair_expectation' else 1)}
    if seen != expected:
        raise ValueError('Incomplete replica coverage')
    for e in range(8):
        for c in CONDITIONS:
            for k in K_VALUES:
                subsets = [tuple(sorted(p.purchased)) for p in freeze.plans
                           if p.key[:4] == (e, c, k, 'uniform_pair_expectation')]
                if sorted(subsets) != list(combinations(freeze.specs[e].editable, k)):
                    raise ValueError('Uniform expectation must enumerate each subset exactly once')


class PurchasedCView:
    """One replica's capability. The fetch callback must remain evaluator-owned."""
    def __init__(self, plan: ReplicaPlan, fetch: Callable[[ReplicaPlan, int], Sequence[float]]):
        self.plan = plan
        self.__fetch = fetch
        self.__seen = set()

    def read(self, edge: int) -> np.ndarray:
        if edge not in self.plan.purchased:
            raise PermissionError('Unpurchased C edge')
        if edge in self.__seen:
            raise PermissionError('Duplicate C read in replica')
        result = _observations(self.__fetch(self.plan, edge), (12,), (-1, 0, 1), 'Purchased C differences')
        self.__seen.add(edge)
        return result


@dataclass(frozen=True)
class ActionRecord:
    plan: ReplicaPlan
    updated_gain: tuple[float, ...]
    cell_provenance: tuple[str, ...]
    feedback_sha256: tuple[tuple[int, str], ...]
    action: Optional[int]


@dataclass(frozen=True)
class ActionFreeze:
    purchase_digest: str
    purchases: PurchaseFreeze
    records: tuple[ActionRecord, ...]

    @property
    def digest(self):
        return _digest(self)


def choose_action(gains: Sequence[float], edges: Sequence[int]) -> Optional[int]:
    if len(gains) != len(edges) or not len(edges) or len(set(edges)) != len(edges):
        raise ValueError('Action vectors invalid')
    if not np.isfinite(gains).all():
        raise ValueError('Action gains nonfinite')
    winner = min(range(len(edges)), key=lambda j: (-float(gains[j]), int(edges[j])))
    return int(edges[winner]) if float(gains[winner]) > 0 else None


def update_and_freeze_actions(purchases: PurchaseFreeze,
                             read_C: Callable[[ReplicaPlan, int], Sequence[float]],
                             *, persisted_purchase_digest: str) -> ActionFreeze:
    """C stage. Adapter first persists purchases; no service scorer is accepted."""
    _validate_purchase_freeze(purchases)
    purchase_digest = purchases.digest
    if persisted_purchase_digest != purchase_digest:
        raise ValueError('Purchase freeze receipt mismatch')
    records = []
    for plan in purchases.plans:
        state = purchases.initial[plan.episode]
        updated = list(state.gain)
        provenance = ['source_A'] * 6
        evidence = []
        view = PurchasedCView(plan, read_C)
        for edge in plan.purchased:
            d = view.read(edge)
            j = state.editable.index(edge)
            updated[j] = float(d.mean())
            provenance[j] = 'current_B' if plan.condition == 'change' else 'current_A'
            evidence.append((edge, _digest(d.tolist())))
        action = None if plan.method == 'current_stay' else choose_action(updated, state.editable)
        records.append(ActionRecord(plan, tuple(updated), tuple(provenance), tuple(evidence), action))
    if purchases.digest != purchase_digest:
        raise ValueError('C callback mutated frozen purchase state')
    result = ActionFreeze(purchase_digest, purchases, tuple(records))
    _validate_action_freeze(result)
    return result


def _validate_action_freeze(freeze):
    _validate_purchase_freeze(freeze.purchases)
    if freeze.purchase_digest != freeze.purchases.digest or tuple(r.plan for r in freeze.records) != freeze.purchases.plans:
        raise ValueError('Actions not bound to complete purchase freeze')
    keyed = {}
    common_feedback_hashes = {}
    for rec in freeze.records:
        p = rec.plan
        state = freeze.purchases.initial[p.episode]
        if len(rec.updated_gain) != 6 or len(rec.cell_provenance) != 6 or not np.isfinite(rec.updated_gain).all() or any(abs(g) > 1 for g in rec.updated_gain):
            raise ValueError('Invalid updated table')
        if tuple(e for e, _ in rec.feedback_sha256) != p.purchased:
            raise ValueError('Purchased feedback evidence mismatch')
        for edge, pin in rec.feedback_sha256:
            if not isinstance(pin, str) or len(pin) != 64 or any(c not in '0123456789abcdef' for c in pin):
                raise ValueError('Invalid purchased observation digest')
            observation_key = (p.episode, p.condition, edge)
            if observation_key in common_feedback_hashes and common_feedback_hashes[observation_key] != pin:
                raise ValueError('Same purchased C observation differs between replicas')
            common_feedback_hashes[observation_key] = pin
        for j, edge in enumerate(state.editable):
            expected = ('current_B' if p.condition == 'change' else 'current_A') if edge in p.purchased else 'source_A'
            if rec.cell_provenance[j] != expected:
                raise ValueError('Lost source graph provenance')
            if edge not in p.purchased and rec.updated_gain[j] != state.gain[j]:
                raise ValueError('Unpurchased value changed')
        expected = None if p.method == 'current_stay' else choose_action(rec.updated_gain, state.editable)
        if rec.action != expected:
            raise ValueError('Frozen action violates strict positive/tie editor')
        keyed[p.key] = rec
    for e in range(8):
        for k in K_VALUES:
            a = keyed[(e, 'no_change', k, 'candidate', 0)]
            b = keyed[(e, 'no_change', k, 'no_graph_ablation', 0)]
            if (a.plan.purchased, a.updated_gain, a.action) != (b.plan.purchased, b.updated_gain, b.action):
                raise ValueError('No-change full/no-graph identity failed; callback data inconsistent')


@dataclass(frozen=True)
class GradedReplica:
    action_record: ActionRecord
    service_scores: tuple[float, ...]


@dataclass(frozen=True)
class ServiceEvaluation:
    action_digest: str
    actions: ActionFreeze
    grades: tuple[GradedReplica, ...]

    @property
    def digest(self):
        return _digest(self)


def grade_service(actions: ActionFreeze,
                  read_S: Callable[[int, str, Optional[int]], Sequence[float]],
                  *, persisted_action_digest: str) -> ServiceEvaluation:
    """Evaluator-only stage. S values can never change immutable selected actions."""
    _validate_action_freeze(actions)
    action_digest = actions.digest
    if persisted_action_digest != action_digest:
        raise ValueError('Action freeze receipt mismatch')
    evaluated = {}
    rows = []
    for rec in actions.records:
        key = (rec.plan.episode, rec.plan.condition, rec.action)
        if key not in evaluated:
            evaluated[key] = tuple(map(float, _observations(read_S(*key), (24,), (0, 1), 'S binary quality')))
        rows.append(GradedReplica(rec, evaluated[key]))
    if actions.digest != action_digest:
        raise ValueError('S callback mutated frozen action state')
    return ServiceEvaluation(action_digest, actions, tuple(rows))


def _policy_arrays(evaluation: ServiceEvaluation):
    _validate_action_freeze(evaluation.actions)
    if evaluation.action_digest != evaluation.actions.digest:
        raise ValueError('Service evaluation action binding mismatch')
    if tuple(r.action_record for r in evaluation.grades) != evaluation.actions.records:
        raise ValueError('Incomplete service evaluation')
    arrays = {(c, k, m): np.zeros((8, 24), dtype=np.float64)
              for c in CONDITIONS for k in K_VALUES for m in METHODS}
    for row in evaluation.grades:
        p = row.action_record.plan
        y = _observations(row.service_scores, (24,), (0, 1), 'Frozen S quality')
        arrays[(p.condition, p.k, p.method)][p.episode] += y / p.weight_denominator
    return arrays


def episode_bootstrap(values, *, resamples=10000, seed=69173):
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (8,) or not np.isfinite(x).all():
        raise ValueError('Need eight finite episode contrasts')
    if resamples != 10000 or seed != 69173:
        raise ValueError('Frozen exploratory bootstrap parameters changed')
    indices = np.random.default_rng(seed).integers(0, 8, size=(resamples, 8))
    sampled = x[indices].mean(axis=1)
    lo, hi = np.quantile(sampled, [.025, .975], method='linear')
    return {'mean': float(x.mean()), 'ci95': [float(lo), float(hi)],
            'episode_values': x.tolist(), 'resamples': resamples, 'seed': seed,
            'unit': 'episode', 'independent_episode_count': 8,
            'scope': 'exploratory_eight_cluster_percentile_not_confirmation'}


def summarize(evaluation: ServiceEvaluation):
    arrays = _policy_arrays(evaluation)
    methods = []
    for (condition, k, method), values in arrays.items():
        chosen = {'stay': 0.0, 'add': 0.0, 'delete': 0.0}
        for row in evaluation.grades:
            rec, p = row.action_record, row.action_record.plan
            if (p.condition, p.k, p.method) != (condition, k, method):
                continue
            spec = evaluation.actions.purchases.specs[p.episode]
            mask = spec.B if condition == 'change' else spec.A
            category = 'stay' if rec.action is None else ('delete' if mask[rec.action] else 'add')
            chosen[category] += 1 / p.weight_denominator
        methods.append({'condition': condition, 'k': k, 'method': method,
                        'accuracy': float(values.mean()),
                        'episode_accuracy': values.mean(axis=1).tolist(),
                        'expected_episode_action_counts': chosen,
                        'service_questions': 192, 'learning_episodes': 8})
    contrasts = {}
    for k in K_VALUES:
        for condition in CONDITIONS:
            candidate = arrays[(condition, k, 'candidate')].mean(axis=1)
            for base in METHODS:
                if base == 'candidate':
                    continue
                diff = candidate - arrays[(condition, k, base)].mean(axis=1)
                contrasts[f'{condition}/k{k}/candidate-minus-{base}'] = episode_bootstrap(diff)
        change = arrays[('change', k, 'candidate')].mean(axis=1) - arrays[('change', k, 'uniform_pair_expectation')].mean(axis=1)
        stable = arrays[('no_change', k, 'candidate')].mean(axis=1) - arrays[('no_change', k, 'uniform_pair_expectation')].mean(axis=1)
        contrasts[f'change-minus-no_change/k{k}/candidate-minus-uniform'] = episode_bootstrap(change - stable)
    primary_key = 'change/k2/candidate-minus-uniform_pair_expectation'
    return {'schema': 'mab2_r1_array_summary_v1', 'protocol_sha256': PROTOCOL_SHA256,
            'purchase_digest': evaluation.actions.purchase_digest,
            'action_digest': evaluation.action_digest, 'evaluation_digest': evaluation.digest,
            'primary_key': primary_key, 'primary': contrasts[primary_key],
            'methods': methods, 'contrasts': contrasts,
            'uniform_reference': 'exact expectation over all subsets; each replica owns only its purchases',
            'point_rule_satisfied': bool(contrasts[primary_key]['mean'] >= .01 and
                                       contrasts['change/k2/candidate-minus-source_only']['mean'] >= 0),
            'budget_and_confirmation_feasibility_not_assessed_here': True,
            'evidence_tier': 'development_pilot_not_independent_confirmation',
            'secondary_multiplicity': 'descriptive_no_substitution_as_winner',
            'replicas_or_seeds_count_as_extra_independent_questions': False}


def conservative_budget(k: int, method: str):
    """Logical cap for one episode-condition policy; not physical cached spend."""
    if k not in K_VALUES or method not in METHODS:
        raise ValueError('Unknown quota or method')
    base, flip = 75008, 16768
    init = 0 if method == 'current_stay' else 12 * (base + 6 * flip)
    init_scores = 0 if method == 'current_stay' else 12 * 7
    bought = 0 if method in ('source_only', 'current_stay') else (6 if method == 'full_revalidation_high_cost' else k)
    audit = 12 * (base + bought * flip) if bought else 0
    audit_scores = 12 * (1 + bought) if bought else 0
    result = {'initial_token_cap': init, 'audit_token_cap': audit,
              'service_token_cap': 24 * base, 'total_token_cap': init + audit + 24 * base,
              'score_count_cap': init_scores + audit_scores + 24,
              'counterfactual_increment_calls': 12 * bought * 2,
              'uniform_cap_scope': 'per_replica_not_sum_over_evaluator_enumeration',
              'actual_tokens_or_GPU_occupancy': None,
              'current_stay_scope': 'practical_graph_reference_without_I_not_matched_initialization'}
    if k == 2 and method not in ('source_only', 'current_stay', 'full_revalidation_high_cost'):
        if (result['total_token_cap'], result['score_count_cap']) != (5210112, 144):
            raise AssertionError('Frozen main cap mismatch')
    return result


def jsonable_freeze(freeze):
    """Private adapter persistence payload; source/feedback arrays remain private."""
    return {'digest': freeze.digest, 'payload': asdict(freeze)}
