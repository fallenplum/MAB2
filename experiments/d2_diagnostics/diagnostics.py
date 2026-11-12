"""Pure-array D2 diagnostics; this module never opens data or launches experiments.

The objective is sum_i w_i (y_i-X_i beta)^2 + alpha ||beta_pen||^2.
Each training question has weight one. X is phi-major: kron(phi, graph),
with nine phi columns, graph dimension d, graph intercept at column zero,
and the constant phi coordinate last. Thus the global intercept is 8*d.

R0: penalize all; R1: exempt 8*d; R1b: exempt all {k*d: k=0..8};
R2: center X and y within question, retaining graph weights 1/n_q;
R3: fit the supplied 24 prescribed contrasts per question at weight 1/24.
No final-test access, alpha refit, or model selection on training loss occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
TIE_ATOL = 1e-12


@dataclass(frozen=True)
class PreparedProblem:
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    unpenalized_columns: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitResult:
    beta: np.ndarray
    alpha: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SelectionResult:
    best_fit: FitResult
    best_alpha: float
    all_trials: list[dict[str, float]]


def _xy(X, y):
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != len(y):
        raise ValueError("X must be n-by-p and y must be length n")
    if not X.shape[0] or not X.shape[1]:
        raise ValueError("empty design is not permitted")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite")
    return X, y


def _group_indices(question_ids, n):
    ids = np.asarray(question_ids)
    if ids.ndim != 1 or len(ids) != n:
        raise ValueError("question_ids must have one scalar ID per row")
    groups = {}
    for i, value in enumerate(ids):
        key = value.item() if isinstance(value, np.generic) else value
        if key is None or (isinstance(key, float) and not np.isfinite(key)):
            raise ValueError("missing or nonfinite question ID")
        groups.setdefault(key, []).append(i)
    return [(key, np.asarray(rows, dtype=int)) for key, rows in groups.items()]


def question_weights(question_ids) -> np.ndarray:
    """Uniform row weights within each question; total weight is #questions."""
    groups = _group_indices(question_ids, len(question_ids))
    w = np.empty(len(question_ids), dtype=np.float64)
    for _, rows in groups:
        w[rows] = 1.0 / len(rows)
    return w


def collapse_graph_aliases(X, y, question_ids, graph_ids, *, atol=1e-12):
    """Keep one row per (question, graph); reject conflicting aliases.

    This is deduplication, not averaging distinct stochastic observations. The
    adapter must first construct the prescribed seed-mean target per graph.
    Returns X, y, question_ids, graph_ids and retained original row indices.
    """
    X, y = _xy(X, y)
    q, g = np.asarray(question_ids), np.asarray(graph_ids)
    _group_indices(q, len(y))
    if g.ndim != 1 or len(g) != len(y):
        raise ValueError("graph_ids must have one scalar ID per row")
    first, keep = {}, []
    for i in range(len(y)):
        key = (q[i].item() if isinstance(q[i], np.generic) else q[i],
               g[i].item() if isinstance(g[i], np.generic) else g[i])
        if key in first:
            j = first[key]
            if not np.allclose(X[i], X[j], atol=atol, rtol=0) or abs(y[i]-y[j]) > atol:
                raise ValueError("duplicate graph alias has conflicting X or target")
        else:
            first[key] = i
            keep.append(i)
    keep = np.asarray(keep, dtype=int)
    return X[keep], y[keep], q[keep], g[keep], keep


def prepare_problem(X, y, question_ids, variant, *, d=None, graph_ids=None,
                    difference_X=None, difference_y=None,
                    difference_question_ids=None, difference_count=24):
    """Prepare one fixed training objective without looking at tune data.

    d is the graph feature dimension (13 for C1, 79 for C2), not X.shape[1].
    Input graph rows must be unique, or supply graph_ids to deduplicate aliases.
    For R3, difference_* are already the prescribed oriented contrast rows;
    this function never invents or selects contrasts. It verifies exactly 24
    rows per question by default. All problem arrays are float64.
    """
    X, y = _xy(X, y)
    q = np.asarray(question_ids)
    _group_indices(q, len(y))
    original_n = len(y)
    if graph_ids is not None:
        X, y, q, _, _ = collapse_graph_aliases(X, y, q, graph_ids)
    groups = _group_indices(q, len(y))
    if d is not None and (not isinstance(d, (int, np.integer)) or d < 1 or X.shape[1] != 9*d):
        raise ValueError("phi-major design must have exactly 9*d columns")
    if variant not in {"R0", "R1", "R1b", "R2", "R3"}:
        raise ValueError("unknown D2 variant")
    metadata = {"variant": variant, "n_questions": len(groups),
                "n_original_rows": original_n, "n_unique_rows": len(y),
                "weight_convention": "sum_one_per_question", "d": d,
                "profiled_question_offsets_count": len(groups) if variant == "R2" else 0}
    exempt = ()
    if variant in {"R1", "R1b"}:
        if d is None:
            raise ValueError("R1/R1b require d")
        if not np.allclose(X[:, 8*d], 1.0, atol=1e-12, rtol=0):
            raise ValueError("column 8*d must be the global constant one")
        exempt = (8*d,) if variant == "R1" else tuple(k*d for k in range(9))
    if variant == "R3":
        if any(z is None for z in (difference_X, difference_y, difference_question_ids)):
            raise ValueError("R3 requires all three difference arrays")
        DX, Dy = _xy(difference_X, difference_y)
        if DX.shape[1] != X.shape[1]:
            raise ValueError("raw and contrast feature dimensions differ")
        dq = np.asarray(difference_question_ids)
        dg = _group_indices(dq, len(Dy))
        if {k for k, _ in dg} != {k for k, _ in groups}:
            raise ValueError("raw and contrast question sets differ")
        if any(len(rows) != difference_count for _, rows in dg):
            raise ValueError("R3 must have the prescribed number of contrasts per question")
        metadata["contrasts_per_question"] = int(difference_count)
        return PreparedProblem(DX, Dy, question_weights(dq), (), metadata)
    w = question_weights(q)
    if variant == "R2":
        X, y = X.copy(), y.copy()
        for _, rows in groups:
            # Both operands are centered with the same original graph weights.
            local_w = w[rows] / w[rows].sum()
            X[rows] -= local_w @ X[rows]
            y[rows] -= local_w @ y[rows]
        metadata["centered_within_question"] = True
    return PreparedProblem(X, y, w, exempt, metadata)


class _WeightedFWLSolver:
    """A path shares one main SVD; nuisance projection is at most nine columns."""

    def __init__(self, problem):
        X, y = _xy(problem.X, problem.y)
        w = np.asarray(problem.weights, dtype=np.float64)
        if w.shape != y.shape or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("weights must be finite, strictly positive and length n")
        self.sqrtw, self.p = np.sqrt(w), X.shape[1]
        cols = tuple(problem.unpenalized_columns)
        if len(set(cols)) != len(cols) or any(not isinstance(c, (int, np.integer)) or c < 0 or c >= self.p for c in cols):
            raise ValueError("invalid or repeated unpenalized column")
        self.nuisance = np.asarray(cols, dtype=int)
        self.penalized = np.asarray([j for j in range(self.p) if j not in cols], dtype=int)
        Xw = X * self.sqrtw[:, None]
        self.raw_trace = float(np.square(Xw).sum())
        B = Xw[:, self.penalized]
        self.nuisance_rank = 0
        self.Q = np.empty((len(y), 0))
        self.Zpinv = np.empty((0, len(y)))
        self.ZpinvB = np.empty((0, len(self.penalized)))
        if len(cols):
            Z = Xw[:, self.nuisance]
            Uz, sz, Vtz = np.linalg.svd(Z, full_matrices=False)
            cutoff = np.finfo(float).eps * max(Z.shape) * (sz[0] if len(sz) else 0.0)
            keep = sz > cutoff
            self.nuisance_rank = int(keep.sum())
            self.Q = Uz[:, keep]
            self.Zpinv = (Vtz[keep].T / sz[keep]) @ self.Q.T
            self.ZpinvB = self.Zpinv @ B
            B = B - self.Q @ (self.Q.T @ B)
        if len(self.penalized):
            self.U, self.s, self.Vt = np.linalg.svd(B, full_matrices=False)
        else:
            self.U, self.s, self.Vt = np.empty((len(y), 0)), np.empty(0), np.empty((0, 0))
        self.rank_cutoff = np.finfo(float).eps * max(B.shape) * (self.s[0] if len(self.s) else 0.0)
        self.rank = int((self.s > self.rank_cutoff).sum())

    def solve(self, y, alpha):
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        y = np.asarray(y, dtype=np.float64)
        vector = y.ndim == 1
        Y = y[:, None] if vector else y
        if Y.ndim != 2 or Y.shape[0] != len(self.sqrtw) or not np.isfinite(Y).all():
            raise ValueError("invalid target array")
        Yw = Y * self.sqrtw[:, None]
        Yr = Yw - self.Q @ (self.Q.T @ Yw)
        if alpha == 0:
            shrink = np.divide(1.0, self.s, out=np.zeros_like(self.s), where=self.s > self.rank_cutoff)
        else:
            shrink = self.s / (self.s*self.s + alpha)
        bp = self.Vt.T @ (shrink[:, None] * (self.U.T @ Yr))
        beta = np.zeros((self.p, Y.shape[1]), dtype=np.float64)
        beta[self.penalized] = bp
        if len(self.nuisance):
            beta[self.nuisance] = self.Zpinv @ Yw - self.ZpinvB @ bp
        return beta[:, 0] if vector else beta

    def diagnostics(self, problem, beta, alpha):
        resid = problem.y - problem.X @ beta
        rss = float(np.dot(problem.weights, resid*resid))
        penalty = float(np.dot(beta[self.penalized], beta[self.penalized]))
        dfp = self.rank if alpha == 0 else float(np.sum(self.s*self.s/(self.s*self.s+alpha)))
        return {"weighted_rss": rss, "penalized_beta_squared_norm": penalty,
                "objective": rss + float(alpha)*penalty,
                "nuisance_rank": self.nuisance_rank, "penalized_design_rank": self.rank,
                "effective_df": float(self.nuisance_rank + dfp),
                "effective_df_scope": "prepared_observation_operator",
                "profiled_question_offsets_count": int(problem.metadata.get("profiled_question_offsets_count", 0)),
                "projected_spectral_trace": float(np.dot(self.s, self.s)),
                "raw_weighted_feature_trace": self.raw_trace,
                "n_rows": len(problem.y), "n_columns": self.p,
                "weight_sum": float(np.sum(problem.weights)),
                "main_svd_count_per_path": 1 if len(self.penalized) else 0}


def fit_path(problem: PreparedProblem, alphas: Sequence[float] = DEFAULT_ALPHAS) -> list[FitResult]:
    """Return fits in supplied alpha order, with one shared main SVD.

    No alpha is selected here. The returned objects retain only coefficients
    and scalar diagnostics, so the SVD matrices are freed on return.
    """
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(not np.isfinite(a) or a < 0 for a in alphas):
        raise ValueError("alphas must be nonempty, finite and nonnegative")
    solver = _WeightedFWLSolver(problem)
    fits = []
    for alpha in alphas:
        beta = solver.solve(problem.y, alpha)
        fits.append(FitResult(beta, alpha, solver.diagnostics(problem, beta, alpha)))
    return fits


def fit_ridge(problem: PreparedProblem, alpha: float) -> FitResult:
    return fit_path(problem, (alpha,))[0]


def question_mean_mse(prediction, target, question_ids) -> float:
    pred, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if pred.ndim != 1 or target.shape != pred.shape or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target must be finite equal-length vectors")
    groups = _group_indices(question_ids, len(pred))
    if not groups:
        raise ValueError("empty scoring data")
    sq = (pred-target)**2
    return float(np.mean([float(np.mean(sq[rows])) for _, rows in groups]))


def select_alpha(problem, tune_credit_X, tune_credit_y, tune_question_ids,
                 alphas=DEFAULT_ALPHAS, *, fits=None) -> SelectionResult:
    """Tune only: minimize question-mean credit MSE; ties <=1e-12 use max alpha.

    all_trials is a JSON-serializable list of {alpha, tune_credit_mse} dicts.
    best_fit is exactly a training fit, never refit on train+tune. If passing
    fits from fit_path, caller owns provenance; their alpha list must match.
    """
    TX, Ty = _xy(tune_credit_X, tune_credit_y)
    if TX.shape[1] != problem.X.shape[1]:
        raise ValueError("tune and train feature dimensions differ")
    expected = tuple(float(a) for a in alphas)
    path = fit_path(problem, expected) if fits is None else list(fits)
    if tuple(f.alpha for f in path) != expected or not path:
        raise ValueError("precomputed fits must match the ordered alpha grid")
    trials = [{"alpha": f.alpha, "tune_credit_mse": question_mean_mse(TX@f.beta, Ty, tune_question_ids)} for f in path]
    minimum = min(t["tune_credit_mse"] for t in trials)
    eligible = [i for i, t in enumerate(trials) if t["tune_credit_mse"] <= minimum + TIE_ATOL]
    best = max(eligible, key=lambda i: path[i].alpha)
    return SelectionResult(path[best], path[best].alpha, trials)


def constant_residual_decomposition(problem, alpha, credit_X=None):
    """Exact linear R0 decomposition at the SAME alpha, not a retuned method.

    y = weighted training mean + residual. Return direct/constant/residual
    beta and (optionally) credit predictions plus numerical linearity errors.
    """
    if problem.unpenalized_columns or problem.metadata.get("variant", "R0") != "R0":
        raise ValueError("this attribution diagnostic is defined for R0")
    mu = float(np.average(problem.y, weights=problem.weights))
    targets = np.column_stack((problem.y, np.full_like(problem.y, mu), problem.y-mu))
    solver = _WeightedFWLSolver(problem)
    betas = solver.solve(targets, alpha)
    result = {"alpha": float(alpha), "weighted_target_mean": mu,
              "beta_direct": betas[:, 0], "beta_constant": betas[:, 1],
              "beta_residual": betas[:, 2],
              "beta_linearity_max_abs": float(np.max(np.abs(betas[:, 0]-betas[:, 1]-betas[:, 2])))}
    if credit_X is not None:
        CX = np.asarray(credit_X, dtype=float)
        if CX.ndim != 2 or CX.shape[1] != len(betas) or not np.isfinite(CX).all():
            raise ValueError("invalid credit design")
        pred = CX @ betas
        result.update(credit_direct=pred[:, 0], credit_constant=pred[:, 1], credit_residual=pred[:, 2],
                      credit_linearity_max_abs=float(np.max(np.abs(pred[:, 0]-pred[:, 1]-pred[:, 2]))) if len(pred) else 0.0)
    return result


def full_pairwise_problem(X, y, question_ids):
    """All i<j contrasts weighted 1/n_q**2: exactly the R2 loss, including alpha.

    This is a mathematical diagnostic from existing graph targets, not R3's
    prescribed 24 contrasts. It materializes O(sum n_q**2) rows.
    """
    X, y = _xy(X, y)
    dx, dy, weights = [], [], []
    for _, rows in _group_indices(question_ids, len(y)):
        n = len(rows)
        for i in range(n):
            for j in range(i+1, n):
                a, b = rows[i], rows[j]
                dx.append(X[a]-X[b]); dy.append(y[a]-y[b]); weights.append(1.0/(n*n))
    if not dx:
        raise ValueError("no within-question pairs exist")
    return PreparedProblem(np.asarray(dx), np.asarray(dy), np.asarray(weights), (),
                           {"variant": "R2_complete_pairwise_identity", "weight_convention": "1/n_q^2_per_pair"})


def pairwise_loss_identity(residual, question_ids):
    r = np.asarray(residual, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError("residual must be a finite vector")
    rows_out = []
    for key, rows in _group_indices(question_ids, len(r)):
        v, n = r[rows], len(rows)
        centered = float(np.mean((v-v.mean())**2))
        pairs = float(sum((v[i]-v[j])**2 for i in range(n) for j in range(i+1, n))/(n*n))
        rows_out.append({"question_id": key, "n": n, "centered_loss": centered,
                         "complete_pairwise_loss": pairs, "absolute_error": abs(centered-pairs)})
    return rows_out


def contrast_graph_diagnostics(D, *, weights=None, alpha=None):
    """Incidence rank/connectivity and weighted Laplacian spectrum for one question.

    D columns are ALL unique graph vertices, including isolated ones. Each
    nonzero row must have one -1 and one +1; zero (self) rows are permitted.
    Default weights are 1/m, matching R3 m=24. effective_df is the graph
    operator's df, distinct from fit_path's feature-model smoother df.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or min(D.shape) < 1 or not np.isfinite(D).all():
        raise ValueError("D must be finite and nonempty")
    m, n = D.shape
    w = np.full(m, 1.0/m) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (m,) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("invalid contrast weights")
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    zero = 0
    for row in D:
        nz = np.flatnonzero(np.abs(row) > 1e-12)
        if not len(nz):
            zero += 1; continue
        if len(nz) != 2 or not np.allclose(np.sort(row[nz]), [-1, 1], atol=1e-12, rtol=0):
            raise ValueError("D rows must be oriented incidence vectors")
        parent[find(int(nz[0]))] = find(int(nz[1]))
    sizes = {}
    for i in range(n):
        root = find(i); sizes[root] = sizes.get(root, 0)+1
    s = np.linalg.svd(D, compute_uv=False)
    cutoff = np.finfo(float).eps * max(D.shape) * (s[0] if len(s) else 0)
    rank = int((s > cutoff).sum())
    sw = np.linalg.svd(D*np.sqrt(w)[:, None], compute_uv=False)
    spectrum = np.sort(np.concatenate((sw*sw, np.zeros(n-len(sw)))))
    out = {"n_vertices": n, "n_contrasts": m, "n_zero_rows": zero,
           "rank": rank, "nullity": n-rank, "n_components": len(sizes),
           "component_sizes": sorted(sizes.values()), "incidence_rank_expected": n-len(sizes),
           "rank_matches_connectivity": rank == n-len(sizes),
           "unweighted_spectral_trace": float(np.sum(D*D)),
           "weighted_spectral_trace": float(spectrum.sum()),
           "weighted_laplacian_eigenvalues": spectrum.tolist(),
           "r2_centering_rank": n-1, "r2_weighted_trace": (n-1)/n}
    if alpha is not None:
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("invalid alpha")
        out["alpha"] = alpha
        out["graph_operator_effective_df"] = float(rank if alpha == 0 else np.sum(spectrum/(spectrum+alpha)))
        out["r2_graph_operator_effective_df"] = float(n-1 if alpha == 0 else (n-1)*(1/n)/(1/n+alpha))
    return out
