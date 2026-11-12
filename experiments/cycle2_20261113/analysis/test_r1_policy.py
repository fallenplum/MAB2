"""Synthetic independent-oracle and information-boundary tests; no real data."""
from dataclasses import FrozenInstanceError, replace
from itertools import combinations
import inspect
import unittest

import numpy as np

import r1_policy as r


def zero_I():
    return {e: np.zeros((12, 6)) for e in range(8)}


def fixed_C(plan, edge):
    # One immutable menu per episode-condition; method never affects observation.
    values = np.zeros(12)
    values[:(edge % 4)] = 1
    if plan.condition == 'no_change':
        values = -values
    return values


def S_scores(episode, condition, action):
    n = (episode * 3 + (0 if action is None else action + 1) +
         (2 if condition == 'change' else 0)) % 25
    return np.array([1] * n + [0] * (24 - n))


class R1PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = r.canonical_episode_specs()
        cls.purchases = r.freeze_purchases(zero_I(), cls.specs)
        cls.actions = r.update_and_freeze_actions(cls.purchases, fixed_C,
                                                  persisted_purchase_digest=cls.purchases.digest)
        cls.evaluation = r.grade_service(cls.actions, S_scores,
                                         persisted_action_digest=cls.actions.digest)

    def test_role_balance_and_complements(self):
        receivers = []
        for spec in self.specs:
            self.assertEqual(sum(spec.A), 9)
            self.assertEqual(sum(spec.B), 6)
            self.assertEqual(sum(spec.dirty), 3)
            self.assertEqual([i for i in range(12) if spec.A[i] != spec.B[i]], list(spec.changed))
            self.assertFalse(set(spec.changed) & set(spec.editable))
            stable = {r.EDGES[e][1] for e, d in zip(spec.editable, spec.dirty) if not d}
            self.assertEqual(len(stable), 1)
            receivers.extend(stable)
            self.assertEqual(sum(spec.A[e] for e in spec.editable), 3)
        self.assertEqual(sorted(receivers), [0, 0, 1, 1, 2, 2, 3, 3])
        for i in range(0, 8, 2):
            a, b = self.specs[i:i + 2]
            self.assertEqual(a.editable, b.editable)
            for e in range(12):
                self.assertEqual(a.A[e] != b.A[e], e in a.editable)

    def test_frozen_protocol_hash_rejects_any_other_bytes(self):
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            r.specs_from_protocol_bytes(b'{}')

    def test_rotated_menu_bridge_uses_physical_edge_not_template_order(self):
        template_edges = (0, 3, 4, 6, 8, 9)
        for spec in self.specs:
            permuted_in_request_order = []
            for edge in template_edges:
                src, dst = r.EDGES[edge]
                physical = r.EDGES.index(((src + spec.shift) % 4, (dst + spec.shift) % 4))
                permuted_in_request_order.append(physical)
            # Unique binary column barcode for every physical edge index.
            by_edge = {edge: np.array([(edge >> (j % 4)) & 1 for j in range(12)])
                       for edge in permuted_in_request_order}
            aligned = r.align_initial_columns(by_edge, spec)
            for column, edge in enumerate(spec.editable):
                self.assertEqual(int(sum(aligned[j, column] * (2 ** j) for j in range(4))), edge)
            if spec.shift:
                naive = np.column_stack([by_edge[e] for e in permuted_in_request_order])
                self.assertFalse(np.array_equal(naive, aligned))
        with self.assertRaises(ValueError):
            r.align_initial_columns({0: np.zeros(12)}, self.specs[0])

    def test_no_service_or_feedback_parameter_in_I_stage(self):
        self.assertEqual(list(inspect.signature(r.freeze_purchases).parameters), ['initial_deltas', 'specs'])
        self.assertNotIn('read_S', inspect.signature(r.update_and_freeze_actions).parameters)

    def test_source_copy_not_alias_and_freeze_immutable(self):
        source = zero_I()
        plan = r.freeze_purchases(source, self.specs)
        digest = plan.digest
        source[0][:] = 1
        self.assertEqual(plan.digest, digest)
        self.assertEqual(plan.initial[0].gain, (0.,) * 6)
        with self.assertRaises(FrozenInstanceError):
            plan.initial = ()

    def test_bad_I_shape_nonfinite_fractional_rejected(self):
        for bad in [np.zeros((11, 6)), np.full((12, 6), np.nan), np.full((12, 6), .5)]:
            source = zero_I(); source[3] = bad
            with self.assertRaises(ValueError):
                r.freeze_purchases(source, self.specs)

    def test_uncertainty_floor_and_source_variance_scalar_oracle(self):
        source = zero_I()
        source[0][:, 0] = [1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]
        plan = r.freeze_purchases(source, self.specs)
        mean = sum(source[0][:, 0]) / 12
        var = sum((v - mean) ** 2 for v in source[0][:, 0]) / 11
        self.assertAlmostEqual(plan.initial[0].gain[0], mean)
        self.assertAlmostEqual(plan.initial[0].width[0], (var / 12) ** .5 + 1 / 12)
        self.assertEqual(plan.initial[0].width[1], 1 / 12)
        self.assertEqual(plan.initial[0].incumbent, mean)

    def test_dirty_rank_and_no_change_identity(self):
        keyed = {p.key: p for p in self.purchases.plans}
        for spec in self.specs:
            dirty_sorted = tuple(e for e, d in zip(spec.editable, spec.dirty) if d)
            for k in (1, 2):
                self.assertEqual(keyed[(spec.episode, 'change', k, 'candidate', 0)].purchased,
                                 dirty_sorted[:k])
                self.assertEqual(keyed[(spec.episode, 'no_change', k, 'candidate', 0)].purchased,
                                 keyed[(spec.episode, 'no_change', k, 'no_graph_ablation', 0)].purchased)

    def test_uniform_all_subsets_once_equal_weight(self):
        for e in range(8):
            for condition in r.CONDITIONS:
                for k in (1, 2):
                    plans = [p for p in self.purchases.plans if p.key[:4] == (e, condition, k, 'uniform_pair_expectation')]
                    self.assertEqual([p.purchased for p in plans], list(combinations(self.specs[e].editable, k)))
                    self.assertAlmostEqual(sum(1 / p.weight_denominator for p in plans), 1)

    def test_cyclic_physical_edges_not_local_indices(self):
        for p in self.purchases.plans:
            if p.method == 'cyclic_pair':
                edges = self.specs[p.episode].editable
                self.assertEqual(p.purchased, tuple(edges[(p.episode + j) % 6] for j in range(p.k)))

    def test_unpurchased_C_is_rejected_without_fetch(self):
        p = next(p for p in self.purchases.plans if p.method == 'candidate')
        calls = []
        view = r.PurchasedCView(p, lambda plan, edge: calls.append(edge) or np.zeros(12))
        forbidden = next(e for e in self.specs[p.episode].editable if e not in p.purchased)
        with self.assertRaises(PermissionError):
            view.read(forbidden)
        self.assertEqual(calls, [])
        view.read(p.purchased[0])
        with self.assertRaises(PermissionError):
            view.read(p.purchased[0])
        self.assertEqual(calls, [p.purchased[0]])

    def test_only_purchased_C_cells_requested_no_source_or_stay_reads(self):
        calls = []
        def fetch(plan, edge):
            self.assertIn(edge, plan.purchased)
            self.assertNotIn(plan.method, ('source_only', 'current_stay'))
            calls.append((plan.key, edge))
            return fixed_C(plan, edge)
        actions = r.update_and_freeze_actions(self.purchases, fetch,
                                              persisted_purchase_digest=self.purchases.digest)
        expected = [(p.key, e) for p in self.purchases.plans for e in p.purchased]
        self.assertEqual(calls, expected)
        for rec in actions.records:
            for j, edge in enumerate(self.specs[rec.plan.episode].editable):
                if edge not in rec.plan.purchased:
                    self.assertEqual(rec.updated_gain[j], 0)
                    self.assertEqual(rec.cell_provenance[j], 'source_A')

    def test_uniform_replicas_cannot_pool_feedback(self):
        # Exactly one beneficial edge. A k-subset can discover it only with k/6 probability.
        special = {spec.episode: spec.editable[-1] for spec in self.specs}
        def fetch(plan, edge):
            return np.ones(12) if edge == special[plan.episode] else np.zeros(12)
        actions = r.update_and_freeze_actions(self.purchases, fetch,
                                              persisted_purchase_digest=self.purchases.digest)
        for e in range(8):
            for c in r.CONDITIONS:
                for k in (1, 2):
                    records = [a for a in actions.records if a.plan.key[:4] == (e, c, k, 'uniform_pair_expectation')]
                    mass = sum(1 / a.plan.weight_denominator for a in records if a.action == special[e])
                    self.assertAlmostEqual(mass, k / 6)
                    self.assertTrue(all(a.action is None for a in records if special[e] not in a.plan.purchased))

    def test_editor_strict_zero_tiny_positive_and_physical_ties(self):
        edges = (9, 4, 0)
        self.assertIsNone(r.choose_action([0., -1., 0.], edges))
        self.assertEqual(r.choose_action([1., 1., 1.], edges), 0)
        self.assertEqual(r.choose_action([1e-300, 0., 0.], edges), 9)
        self.assertEqual(r.choose_action([-1., .25, .25], edges), 0)

    def test_wrong_freeze_receipts_fail_before_callbacks(self):
        def bomb(*args):
            self.fail('Callback called before failed receipt check')
        with self.assertRaises(ValueError):
            r.update_and_freeze_actions(self.purchases, bomb, persisted_purchase_digest='bad')
        with self.assertRaises(ValueError):
            r.grade_service(self.actions, bomb, persisted_action_digest='bad')

    def test_no_change_observation_mismatch_detected(self):
        def inconsistent(plan, edge):
            if plan.condition == 'no_change' and plan.method == 'candidate':
                return np.ones(12)
            return np.zeros(12)
        with self.assertRaisesRegex(ValueError, 'differs between replicas|identity failed'):
            r.update_and_freeze_actions(self.purchases, inconsistent,
                                       persisted_purchase_digest=self.purchases.digest)

    def test_same_changed_observation_cannot_differ_across_replicas(self):
        def inconsistent(plan, edge):
            return np.ones(12) if plan.condition == 'change' and plan.method == 'candidate' else np.zeros(12)
        with self.assertRaisesRegex(ValueError, 'differs between replicas'):
            r.update_and_freeze_actions(self.purchases, inconsistent,
                                       persisted_purchase_digest=self.purchases.digest)

    def test_purchase_or_action_tamper_rejected(self):
        bad = replace(self.purchases, plans=self.purchases.plans[:-1])
        with self.assertRaises(ValueError):
            r.update_and_freeze_actions(bad, fixed_C, persisted_purchase_digest=bad.digest)
        first = self.actions.records[0]
        changed = replace(first, updated_gain=(9.,) * 6)
        bad_actions = replace(self.actions, records=(changed,) + self.actions.records[1:])
        with self.assertRaises(ValueError):
            r.grade_service(bad_actions, S_scores, persisted_action_digest=bad_actions.digest)

    def test_valid_but_wrong_rank_purchase_rejected(self):
        first = self.purchases.plans[0]
        self.assertEqual(first.method, 'candidate')
        alternatives = [e for e in self.specs[first.episode].editable if e not in first.purchased]
        changed = replace(first, purchased=(alternatives[-1],))
        bad = replace(self.purchases, plans=(changed,) + self.purchases.plans[1:])
        with self.assertRaisesRegex(ValueError, 'ranking differs'):
            r.update_and_freeze_actions(bad, fixed_C, persisted_purchase_digest=bad.digest)

    def test_S_evaluation_never_changes_actions_and_reads_selected_only(self):
        before = self.actions.digest
        calls = []
        allowed = {(a.plan.episode, a.plan.condition, a.action) for a in self.actions.records}
        def fetch(*key):
            self.assertIn(key, allowed)
            calls.append(key)
            return np.ones(24)
        ev = r.grade_service(self.actions, fetch, persisted_action_digest=before)
        self.assertEqual(set(calls), allowed)
        self.assertEqual(len(calls), len(allowed))
        self.assertEqual(self.actions.digest, before)
        self.assertEqual(ev.actions.digest, before)

    def test_bad_C_or_S_rejects_incomplete_or_fractional_observations(self):
        with self.assertRaises(ValueError):
            r.update_and_freeze_actions(self.purchases, lambda *x: np.zeros(11),
                                       persisted_purchase_digest=self.purchases.digest)
        with self.assertRaises(ValueError):
            r.grade_service(self.actions, lambda *x: np.full(24, .5),
                            persisted_action_digest=self.actions.digest)

    def test_exact_uniform_expected_quality_scalar_oracle(self):
        summary = r.summarize(self.evaluation)
        target = next(x for x in summary['methods'] if x['condition'] == 'change' and x['k'] == 2 and x['method'] == 'uniform_pair_expectation')
        episode_values = []
        for spec in self.specs:
            subtotal = 0
            for subset in combinations(spec.editable, 2):
                # Rebuild each table from zero, use only that subset, choose scalar max.
                g = {e: 0. for e in spec.editable}
                for e in subset:
                    g[e] = (e % 4) / 12
                winner = min(spec.editable, key=lambda e: (-g[e], e))
                action = winner if g[winner] > 0 else None
                subtotal += sum(S_scores(spec.episode, 'change', action)) / 24 / 15
            episode_values.append(subtotal)
        np.testing.assert_allclose(target['episode_accuracy'], episode_values, rtol=0, atol=2e-16)
        self.assertAlmostEqual(target['accuracy'], sum(episode_values) / 8)

    def test_bootstrap_episode_unit_independent_count_oracle(self):
        x = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
        actual = r.episode_bootstrap(x)
        draw = np.random.default_rng(69173).integers(0, 8, size=(10000, 8))
        manual = [sum(x[int(i)] for i in row) / 8 for row in draw]
        np.testing.assert_allclose(actual['ci95'], np.quantile(manual, [.025, .975]), atol=0, rtol=0)
        self.assertEqual(actual['mean'], .125)
        self.assertEqual(actual['independent_episode_count'], 8)
        with self.assertRaises(ValueError):
            r.episode_bootstrap(np.repeat(x, 24))

    def test_summary_primary_and_difference_of_differences_scalar(self):
        summary = r.summarize(self.evaluation)
        means = {(x['condition'], x['k'], x['method']): x['episode_accuracy'] for x in summary['methods']}
        key = ('change', 2)
        manual = [(a - b) for a, b in zip(means[(*key, 'candidate')], means[(*key, 'uniform_pair_expectation')])]
        np.testing.assert_allclose(summary['primary']['episode_values'], manual)
        stable = [a - b for a, b in zip(means[('no_change', 2, 'candidate')], means[('no_change', 2, 'uniform_pair_expectation')])]
        did = summary['contrasts']['change-minus-no_change/k2/candidate-minus-uniform']['episode_values']
        np.testing.assert_allclose(did, [x - y for x, y in zip(manual, stable)])
        self.assertFalse(summary['replicas_or_seeds_count_as_extra_independent_questions'])

    def test_complete_cap_arithmetic_and_uniform_not_15x_charged(self):
        expected = 12 * (75008 + 6 * 16768) + 12 * (75008 + 2 * 16768) + 24 * 75008
        a = r.conservative_budget(2, 'candidate')
        b = r.conservative_budget(2, 'uniform_pair_expectation')
        self.assertEqual(expected, 5210112)
        self.assertEqual(a['total_token_cap'], expected)
        self.assertEqual(a, b)
        self.assertEqual(a['score_count_cap'], 144)
        self.assertEqual(a['counterfactual_increment_calls'], 48)
        source = r.conservative_budget(2, 'source_only')
        self.assertEqual(source['audit_token_cap'], 0)
        self.assertEqual(source['score_count_cap'], 108)
        full = r.conservative_budget(2, 'full_revalidation_high_cost')
        self.assertGreater(full['total_token_cap'], expected)
        self.assertEqual(full['score_count_cap'], 192)


if __name__ == '__main__':
    unittest.main(verbosity=2)
