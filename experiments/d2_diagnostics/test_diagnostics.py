"""Synthetic-only mathematical tests. No project data, models or reports read."""

import unittest
from unittest import mock

import numpy as np

from diagnostics import (
    DEFAULT_ALPHAS, FitResult, PreparedProblem, collapse_graph_aliases,
    constant_residual_decomposition, contrast_graph_diagnostics,
    fit_path, fit_ridge, full_pairwise_problem, pairwise_loss_identity,
    prepare_problem, question_mean_mse, select_alpha,
)


def synthetic_phi_design(seed=17, d=13, n_questions=6, n_graphs=8):
    rng = np.random.default_rng(seed)
    phi = np.column_stack((rng.normal(size=(n_questions, 8)), np.ones(n_questions)))
    X, q = [], []
    for qi in range(n_questions):
        g = np.column_stack((np.ones(n_graphs), rng.integers(0, 2, size=(n_graphs, d-1))))
        X.append(np.einsum('a,nb->nab', phi[qi], g).reshape(n_graphs, 9*d))
        q.extend([qi]*n_graphs)
    return np.vstack(X), np.asarray(q), phi


def augmented_least_squares_reference(problem, alpha):
    """Independent reference via ridge pseudo-observations, not FWL."""
    p = problem.X.shape[1]
    penalty = np.sqrt(alpha)*np.eye(p)
    penalty[:, list(problem.unpenalized_columns)] = 0
    A = np.vstack((np.sqrt(problem.weights)[:, None]*problem.X, penalty))
    b = np.concatenate((np.sqrt(problem.weights)*problem.y, np.zeros(p)))
    return np.linalg.lstsq(A, b, rcond=None)[0]


class NumericalTests(unittest.TestCase):
    def test_r0_matches_direct_normal_equations(self):
        rng = np.random.default_rng(11)
        X, y = rng.normal(size=(63, 17)), rng.normal(size=63)
        w = np.exp(rng.normal(size=63))
        p = PreparedProblem(X, y, w)
        for f in fit_path(p):
            ref = np.linalg.solve(X.T@(w[:, None]*X)+f.alpha*np.eye(17), X.T@(w*y))
            np.testing.assert_allclose(f.beta, ref, atol=1e-11, rtol=1e-10)
            np.testing.assert_allclose(X@f.beta, X@ref, atol=1e-11, rtol=1e-10)

    def test_rank_deficient_fwl_matches_augmented_solution(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(45, 11))
        X[:, 0] = 1
        X[:, 1] = 2*X[:, 0]
        X[:, 2] = X[:, 3]+3*X[:, 0]
        p = PreparedProblem(X, rng.normal(size=45), np.linspace(.1, 1, 45), (0, 1, 3))
        for f in fit_path(p, (1e-4, .3, 1000)):
            ref = augmented_least_squares_reference(p, f.alpha)
            np.testing.assert_allclose(f.beta, ref, atol=2e-10, rtol=2e-9)
            self.assertEqual(f.diagnostics['nuisance_rank'], 2)

    def test_path_reuses_main_svd(self):
        rng = np.random.default_rng(2)
        X, y = rng.normal(size=(20, 6)), rng.normal(size=20)
        with mock.patch('numpy.linalg.svd', wraps=np.linalg.svd) as svd:
            path = fit_path(PreparedProblem(X, y, np.ones(20)))
        self.assertEqual(len(path), 8)
        self.assertEqual(svd.call_count, 1)
        X[:, 0] = 1
        with mock.patch('numpy.linalg.svd', wraps=np.linalg.svd) as svd:
            fit_path(PreparedProblem(X, y, np.ones(20), (0,)))
        self.assertEqual(svd.call_count, 2)  # nuisance + shared main SVD

    def test_all_nuisance_and_zero_columns(self):
        X = np.column_stack((np.ones(5), np.arange(5), np.zeros(5)))
        y = 2+3*np.arange(5)
        for f in fit_path(PreparedProblem(X, y, np.ones(5), (0, 1, 2))):
            np.testing.assert_allclose(X@f.beta, y, atol=1e-12)
            self.assertEqual(f.diagnostics['effective_df'], 2)

    def test_constant_target_exposes_pseudocredit(self):
        # Every question has z=0 and z=1. True graph effect is exactly zero.
        d = 13
        X = np.zeros((12, 9*d))
        X[:, 8*d] = 1
        X[:, 8*d+1] = np.tile([0, 1], 6)
        q, y = np.repeat(np.arange(6), 2), np.full(12, 3.0)
        credit = np.zeros((1, 9*d)); credit[:, 8*d+1] = 1
        fits = {v: fit_ridge(prepare_problem(X, y, q, v, d=d), 1) for v in ('R0', 'R1', 'R1b')}
        self.assertGreater(abs(float((credit@fits['R0'].beta)[0])), .1)
        for variant in ('R1', 'R1b'):
            self.assertLess(abs(float((credit@fits[variant].beta)[0])), 1e-12)
            np.testing.assert_allclose(X@fits[variant].beta, y, atol=1e-12)

    def test_r1b_removes_context_baseline_graph_effect(self):
        X, q, phi = synthetic_phi_design()
        baseline = 1.3 + 2*phi[:, 0] - .4*phi[:, 3]
        y = baseline[q]
        p = prepare_problem(X, y, q, 'R1b', d=13)
        fit = fit_ridge(p, .1)
        keep = np.ones(X.shape[1], dtype=bool); keep[list(p.unpenalized_columns)] = False
        np.testing.assert_allclose(fit.beta[keep], 0, atol=1e-11)
        np.testing.assert_allclose(X@fit.beta, y, atol=1e-11)

    def test_r1_global_shift_credit_invariance(self):
        X, q, _ = synthetic_phi_design()
        y = np.random.default_rng(3).normal(size=len(q))
        p1 = prepare_problem(X, y, q, 'R1', d=13)
        p2 = prepare_problem(X, y+17, q, 'R1', d=13)
        b1, b2 = fit_ridge(p1, .03).beta, fit_ridge(p2, .03).beta
        delta = b2-b1; delta[8*13] -= 17
        np.testing.assert_allclose(delta, 0, atol=1e-11)

    def test_r2_centers_both_operands_and_is_shift_invariant(self):
        X, q, _ = synthetic_phi_design()
        rng = np.random.default_rng(15)
        y = rng.normal(size=len(q))
        p = prepare_problem(X, y, q, 'R2', d=13)
        shift = rng.normal(size=6)*5
        p2 = prepare_problem(X, y+shift[q], q, 'R2', d=13)
        for qi in range(6):
            rows = q == qi
            np.testing.assert_allclose(p.weights[rows].sum(), 1)
            np.testing.assert_allclose(p.X[rows].mean(axis=0), 0, atol=1e-14)
            np.testing.assert_allclose(p.y[rows].mean(), 0, atol=1e-14)
        np.testing.assert_allclose(p.X[:, np.arange(9)*13], 0, atol=1e-14)
        for f1, f2 in zip(fit_path(p), fit_path(p2)):
            np.testing.assert_allclose(f1.beta, f2.beta, atol=1e-11)

    def test_complete_pairwise_loss_and_coefficients_equal_r2(self):
        rng = np.random.default_rng(16)
        q = np.repeat(np.arange(4), [2, 3, 5, 1])
        X, y = rng.normal(size=(len(q), 7)), rng.normal(size=len(q))
        centered = prepare_problem(X, y, q, 'R2')
        paired = full_pairwise_problem(X, y, q)
        for fc, fp in zip(fit_path(centered), fit_path(paired)):
            np.testing.assert_allclose(fc.beta, fp.beta, atol=5e-11, rtol=5e-10)
            self.assertAlmostEqual(fc.diagnostics['objective'], fp.diagnostics['objective'], places=11)
        for row in pairwise_loss_identity(y-X@np.arange(7), q):
            self.assertLess(row['absolute_error'], 1e-12)

    def test_graph_alias_deduplication_preserves_weight_and_fit(self):
        rng = np.random.default_rng(21)
        X, y = rng.normal(size=(6, 5)), rng.normal(size=6)
        q, gid = np.repeat([0, 1], 3), np.tile(np.arange(3), 2)
        original = prepare_problem(X, y, q, 'R0', graph_ids=gid)
        copies = np.array([0, 0, 1, 2, 3, 4, 4, 4, 5])
        repeated = prepare_problem(X[copies], y[copies], q[copies], 'R0', graph_ids=gid[copies])
        self.assertEqual(len(repeated.y), 6)
        np.testing.assert_allclose(repeated.weights.sum(), 2)
        np.testing.assert_allclose(fit_ridge(original, .2).beta, fit_ridge(repeated, .2).beta, atol=1e-13)
        conflicting = y[copies].copy(); conflicting[1] += 1
        with self.assertRaises(ValueError):
            collapse_graph_aliases(X[copies], conflicting, q[copies], gid[copies])

    def test_r3_fixed_count_sign_and_order_invariance(self):
        rng = np.random.default_rng(23)
        X, y, q = rng.normal(size=(8, 5)), rng.normal(size=8), np.repeat([0, 1], 4)
        dq = np.repeat([0, 1], 24)
        ix = np.concatenate([rng.integers(4, size=(24, 2))+4*i for i in range(2)])
        DX, Dy = X[ix[:, 0]]-X[ix[:, 1]], y[ix[:, 0]]-y[ix[:, 1]]
        kwargs = dict(difference_X=DX, difference_y=Dy, difference_question_ids=dq)
        p = prepare_problem(X, y, q, 'R3', **kwargs)
        np.testing.assert_allclose(p.weights, 1/24)
        shifted_y = y+np.array([123., -7.])[q]
        np.testing.assert_allclose(shifted_y[ix[:, 0]]-shifted_y[ix[:, 1]], Dy, atol=2e-14)
        order, sign = rng.permutation(48), rng.choice([-1, 1], size=48)
        p2 = prepare_problem(X, y, q, 'R3', difference_X=DX[order]*sign[:, None],
                             difference_y=Dy[order]*sign, difference_question_ids=dq[order])
        for f1, f2 in zip(fit_path(p), fit_path(p2)):
            np.testing.assert_allclose(f1.beta, f2.beta, atol=1e-11)
        with self.assertRaises(ValueError):
            prepare_problem(X, y, q, 'R3', difference_X=DX[:-1], difference_y=Dy[:-1], difference_question_ids=dq[:-1])

    def test_row_order_does_not_change_r0_fwl_or_r2(self):
        X, q, _ = synthetic_phi_design()
        rng = np.random.default_rng(27); y = rng.normal(size=len(q)); order = rng.permutation(len(q))
        for v in ('R0', 'R1', 'R1b', 'R2'):
            a = prepare_problem(X, y, q, v, d=13)
            b = prepare_problem(X[order], y[order], q[order], v, d=13)
            np.testing.assert_allclose(fit_ridge(a, .01).beta, fit_ridge(b, .01).beta, atol=3e-11)

    def test_constant_residual_decomposition_at_fixed_alpha(self):
        rng = np.random.default_rng(29)
        X, y, q = rng.normal(size=(29, 9)), rng.normal(size=29)+2, np.repeat([0, 1, 2], [4, 8, 17])
        CX = rng.normal(size=(12, 9))
        p = prepare_problem(X, y, q, 'R0')
        result = constant_residual_decomposition(p, .07, CX)
        np.testing.assert_allclose(result['beta_direct'], fit_ridge(p, .07).beta, atol=1e-13)
        self.assertLess(result['beta_linearity_max_abs'], 1e-12)
        self.assertLess(result['credit_linearity_max_abs'], 1e-12)
        self.assertAlmostEqual(result['weighted_target_mean'], np.mean([y[q == qi].mean() for qi in range(3)]))

    def test_tuning_is_question_balanced_ties_max_and_no_refit(self):
        self.assertAlmostEqual(question_mean_mse([1, 3, 3, 3], [0, 0, 0, 0], [0, 1, 1, 1]), 5)
        p = prepare_problem(np.eye(3), np.arange(3.), np.arange(3), 'R0')
        path = fit_path(p)
        # Every tune credit prediction is zero: exact tie, so max alpha wins.
        s = select_alpha(p, np.zeros((4, 3)), np.ones(4), [0, 1, 1, 1], fits=path)
        self.assertEqual(s.best_alpha, 1000)
        self.assertIs(s.best_fit, path[-1])
        self.assertTrue(all(set(t) == {'alpha', 'tune_credit_mse'} for t in s.all_trials))
        import json
        json.dumps(s.all_trials, allow_nan=False)

    def test_connectivity_rank_spectrum_and_isolates(self):
        # Two connected triples plus one isolated vertex: rank = 7 - 3 = 4.
        D = np.zeros((5, 7))
        for i, (a, b) in enumerate([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5)]):
            D[i, a], D[i, b] = -1, 1
        info = contrast_graph_diagnostics(D, alpha=.5)
        self.assertEqual((info['rank'], info['n_components'], info['nullity']), (4, 3, 3))
        self.assertEqual(info['component_sizes'], [1, 3, 3])
        self.assertTrue(info['rank_matches_connectivity'])
        self.assertAlmostEqual(info['weighted_spectral_trace'], 2)
        self.assertEqual(info['r2_centering_rank'], 6)
        eig = np.linalg.eigvalsh(D.T@D/5)
        np.testing.assert_allclose(info['weighted_laplacian_eigenvalues'], eig, atol=1e-15)
        self.assertAlmostEqual(info['graph_operator_effective_df'], np.sum(eig/(eig+.5)))
        order = [4, 2, 1, 3, 0]
        info2 = contrast_graph_diagnostics(-D[order], alpha=.5)
        self.assertAlmostEqual(info['graph_operator_effective_df'], info2['graph_operator_effective_df'])

    def test_near_tie_uses_fixed_absolute_tolerance(self):
        p = prepare_problem([[1.]], [0.], [0], 'R0')
        path = [FitResult(np.array([0.]), 1., {}),
                FitResult(np.array([np.sqrt(5e-13)]), 10., {}),
                FitResult(np.array([np.sqrt(2e-12)]), 100., {})]
        selected = select_alpha(p, [[1.]], [0.], [0], alphas=(1, 10, 100), fits=path)
        self.assertEqual(selected.best_alpha, 10)

    def test_second_order_phi_major_nuisance_layout(self):
        X, q, phi = synthetic_phi_design(d=79, n_questions=3, n_graphs=4)
        y = (2*phi[:, 1]+.3)[q]
        p = prepare_problem(X, y, q, 'R1b', d=79)
        self.assertEqual(p.X.shape[1], 711)
        self.assertEqual(p.unpenalized_columns, tuple(k*79 for k in range(9)))
        f = fit_ridge(p, .1)
        np.testing.assert_allclose(X@f.beta, y, atol=1e-11)
        penalized = np.ones(711, dtype=bool); penalized[list(p.unpenalized_columns)] = False
        np.testing.assert_allclose(f.beta[penalized], 0, atol=1e-11)

    def test_zero_incidence_rows_are_recorded(self):
        info = contrast_graph_diagnostics(np.zeros((24, 3)), alpha=1)
        self.assertEqual(info['n_components'], 3)
        self.assertEqual(info['rank'], 0)
        self.assertEqual(info['n_zero_rows'], 24)
        self.assertEqual(info['graph_operator_effective_df'], 0)

    def test_invalid_arrays_fail_closed(self):
        with self.assertRaises(ValueError):
            fit_ridge(PreparedProblem(np.eye(2), np.ones(2), np.array([1, 0])), 1)
        with self.assertRaises(ValueError):
            fit_ridge(PreparedProblem(np.eye(2), np.ones(2), np.ones(2)), -1)
        with self.assertRaises(ValueError):
            prepare_problem(np.eye(2), np.ones(2), [0, 1], 'R1', d=13)
        with self.assertRaises(ValueError):
            contrast_graph_diagnostics([[1, 1]])
        with self.assertRaises(ValueError):
            question_mean_mse([np.nan], [0], [0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
