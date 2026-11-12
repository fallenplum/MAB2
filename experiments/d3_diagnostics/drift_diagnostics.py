"""Pure-array D3 drift enrichment; no files, networking, or experiment launcher.

Run/import with repo/experiments or the stage parent on PYTHONPATH.
Only fit_selector uses train/tune labels. Current-question scores/top2 depend
solely on frozen beta and graph bits. evaluate_selector uses labels solely as
an offline scorer. All real-array access and protocol locking belong to root.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any

import numpy as np

try:
    from d2_diagnostics import diagnostics as ridge
except ImportError:
    from d2_code import diagnostics as ridge

N_EDGES = 12
N_SEEDS = 5
TOP_K = 2
PAIRS = tuple(itertools.combinations(range(N_EDGES), 2))
SEED_PAIRS = tuple(itertools.combinations(range(N_SEEDS), 2))
BOOTSTRAP_SEED = 59173
BOOTSTRAP_RESAMPLES = 10000


@dataclass(frozen=True)
class SelectorModel:
    order: int
    beta: np.ndarray
    alpha: float
    alpha_trials: list[dict[str, float]]
    diagnostics: dict[str, Any]


def _bits(bits):
    b = np.asarray(bits, dtype=np.float64)
    if b.ndim != 3 or b.shape[1:] != (2, N_EDGES) or b.shape[0] < 1:
        raise ValueError("anchor bits must have shape [q,2,12]")
    if not np.isfinite(b).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("graph bits must be exactly binary")
    return b


def _targets(targets):
    t = np.asarray(targets, dtype=np.float64)
    if t.ndim != 4 or t.shape[1:] != (2, N_EDGES, N_SEEDS) or t.shape[0] < 1:
        raise ValueError("credit targets must have shape [q,2,12,5]")
    if not np.isfinite(t).all():
        raise ValueError("credit targets must be finite")
    return t


def graph_features(masks, order=2):
    """Graph potential features, ordered exactly [1,bits,pairs(i<j)]."""
    b = np.asarray(masks, dtype=np.float64)
    if b.ndim < 1 or b.shape[-1] != N_EDGES or not np.isfinite(b).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("masks must end with 12 binary coordinates")
    if order not in (1, 2):
        raise ValueError("only prespecified graph orders 1 and 2 are supported")
    blocks = [np.ones((*b.shape[:-1], 1)), b]
    if order == 2:
        blocks.append(np.stack([b[..., i]*b[..., j] for i, j in PAIRS], axis=-1))
    return np.concatenate(blocks, axis=-1)


def credit_design(bits, order=2):
    """Presence contrast feature rows, shape [q,2,12,13 or 79].

    Credit means potential(edge=1)-potential(edge=0), irrespective of whether
    the anchor currently contains the edge. No removal-sign flip is applied.
    """
    b = _bits(bits)
    if order not in (1, 2):
        raise ValueError("only orders 1 and 2 are supported")
    d = 13 if order == 1 else 79
    X = np.zeros((*b.shape, d), dtype=np.float64)
    for edge in range(N_EDGES):
        X[..., edge, edge+1] = 1
    if order == 2:
        for col, (i, j) in enumerate(PAIRS, start=13):
            X[..., i, col] = b[..., j]
            X[..., j, col] = b[..., i]
    return X


def fit_selector(train_bits, train_targets, tune_bits, tune_targets, *, order=2):
    """Fit the predeclared selector (order 2) or degenerate control (order 1).

    One training question contributes all 24 fixed contrasts, each with weight
    1/24. Alpha selection is the unchanged D2 tune question-mean credit MSE
    rule. No exposed/test arrays are accepted. No train+tune refit occurs.
    The adapter must enforce disjoint question IDs and exact 200/80 counts.
    """
    train_bits, tune_bits = _bits(train_bits), _bits(tune_bits)
    train_targets, tune_targets = _targets(train_targets), _targets(tune_targets)
    if len(train_bits) != len(train_targets) or len(tune_bits) != len(tune_targets):
        raise ValueError("bit and target question counts differ")
    n, nt = len(train_bits), len(tune_bits)
    X = credit_design(train_bits, order).reshape(n*24, -1)
    y = train_targets.mean(axis=-1).reshape(-1)
    p = ridge.PreparedProblem(X, y, np.full(n*24, 1/24), (),
                              {"variant": "D3_graph_only_R3", "order": order, "n_questions": n})
    fits = ridge.fit_path(p, ridge.DEFAULT_ALPHAS)
    selected = ridge.select_alpha(p, credit_design(tune_bits, order).reshape(nt*24, -1),
                                  tune_targets.mean(axis=-1).reshape(-1),
                                  np.repeat(np.arange(nt), 24), fits=fits)
    info = dict(selected.best_fit.diagnostics)
    info.update(train_question_count=n, tune_question_count=nt,
                graph_feature_dimension=X.shape[1], effective_credit_parameter_dimension=X.shape[1]-1,
                candidate_eligible=(order == 2), refit_after_tune=False)
    return SelectorModel(order, selected.best_fit.beta, selected.best_alpha, selected.all_trials, info)


def predict_credits(beta, bits, *, order=2):
    b = np.asarray(beta, dtype=np.float64)
    d = 13 if order == 1 else 79
    if b.shape != (d,) or not np.isfinite(b).all():
        raise ValueError("invalid graph-only coefficient vector")
    return credit_design(bits, order) @ b


def drift_scores(credit_predictions):
    p = np.asarray(credit_predictions, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (2, N_EDGES) or p.shape[0] < 1 or not np.isfinite(p).all():
        raise ValueError("credit predictions must have shape [q,2,12]")
    return np.abs(p[:, 1]-p[:, 0])


def select_top2(scores):
    r = np.asarray(scores, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != N_EDGES or r.shape[0] < 1 or not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("drift scores must be finite nonnegative [q,12]")
    # Stable order retains smaller edge indices for exact score ties.
    return np.argsort(-r, axis=1, kind="stable")[:, :TOP_K]


def squared_drift_by_edge(targets):
    """Unclipped off-diagonal seed-product estimate; shape [q,12].

    Delta uses the same seed at both anchors. Averaging all 10 s<t pairs is
    algebraically equal to averaging the 20 ordered unequal-seed products.
    """
    t = _targets(targets)
    delta = t[:, 1]-t[:, 0]
    return np.mean(np.stack([delta[..., s]*delta[..., u] for s, u in SEED_PAIRS], axis=-1), axis=-1)


def bootstrap_question_mean(values):
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all():
        raise ValueError("bootstrap observations must be finite question-level values")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_RESAMPLES, 200):
        count = min(200, BOOTSTRAP_RESAMPLES-start)
        indices = rng.integers(0, len(v), size=(count, len(v)))
        means[start:start+count] = v[indices].mean(axis=1)
    return {"estimate": float(v.mean()), "interval95": np.quantile(means, [.025, .975]).tolist(),
            "question_count": len(v), "bootstrap_unit": "question", "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED, "data_role": "exposed_development", "independent_confirmation": False}


def evaluate_selector(scores, targets):
    """Offline scorer: select with scores only, then score against drift labels.

    Returns a dict with JSON-safe summary plus private ndarray per_question,
    selected_indices, and squared_drift_by_edge. Negative cross-products and
    negative selected-minus-uniform effects remain unchanged.
    """
    chosen = select_top2(scores)
    z = squared_drift_by_edge(targets)
    if len(chosen) != len(z):
        raise ValueError("score and target question counts differ")
    selected = np.take_along_axis(z, chosen, axis=1).mean(axis=1)
    uniform = z.mean(axis=1)
    difference = selected-uniform
    summary = {"question_count": len(z), "k": TOP_K,
               "selected_squared_drift_mean": float(selected.mean()),
               "uniform_top2_expected_squared_drift_mean": float(uniform.mean()),
               "enrichment": bootstrap_question_mean(difference),
               "all_scores_exactly_tied_questions": int(np.sum(np.ptp(np.asarray(scores), axis=1) == 0)),
               "negative_squared_drift_estimate_coordinates": int(np.sum(z < 0)),
               "negative_estimates_clipped": False, "anchor_pairs_are_within_question": True,
               "same_question_targets_entered_selection": False}
    return {"summary": summary, "selected_indices": chosen,
            "per_question": {"selected_squared_drift": selected, "uniform_expected_squared_drift": uniform,
                             "enrichment": difference}, "squared_drift_by_edge": z}


def first_order_degeneracy(model, train_targets):
    """Exact balanced-design identity: beta_e=N/(N+12*alpha)*B0_e."""
    if model.order != 1:
        raise ValueError("degeneracy control requires order one")
    t = _targets(train_targets)
    n = len(t)
    b0 = t.mean(axis=(0, 1, 3))
    expected = n/(n+12*model.alpha)*b0
    return {"B0": b0, "expected_order1_beta_edges": expected,
            "coefficient_identity_max_abs": float(np.max(np.abs(model.beta[1:]-expected))),
            "constant_coefficient_abs": float(abs(model.beta[0])),
            "context_drift_identically_zero": True}


def spending_gate(summary, *, integrity_ok, nomination_frozen, candidate_name, expected_questions=120):
    """A development-data spending decision, never a scientific confirmation.

    This gate deliberately does not inspect global MSE or action gain.
    Controls cannot be substituted for the protocol's unique candidate.
    """
    en = summary["enrichment"]
    estimate = float(en["estimate"])
    lower = float(en["interval95"][0])
    upper = float(en["interval95"][1])
    selected = float(summary["selected_squared_drift_mean"])
    finite = bool(np.isfinite([estimate, lower, upper, selected]).all() and lower <= upper)
    checks = {"integrity_ok": bool(integrity_ok), "nomination_frozen": bool(nomination_frozen),
              "unique_protocol_candidate": candidate_name == "D3_graph_only_order2_R3",
              "expected_question_count": int(summary["question_count"]) == expected_questions,
              "finite_statistics": finite, "positive_enrichment_point": finite and estimate > 0,
              "positive_enrichment_lower95": finite and lower > 0,
              "positive_selected_squared_drift_point": finite and selected > 0}
    passed = all(checks.values())
    if not finite or not checks["integrity_ok"] or not checks["nomination_frozen"] or not checks["unique_protocol_candidate"] or not checks["expected_question_count"]:
        evidence = "INVALID_OR_UNVERIFIED"
    elif passed:
        evidence = "POSITIVE_DEVELOPMENT_SIGNAL"
    elif upper < 0:
        evidence = "REVERSE_DEVELOPMENT_SIGNAL"
    elif lower == upper == estimate == 0:
        evidence = "ZERO_ENRICHMENT_IN_OBSERVED_SAMPLE"
    elif lower > 0 and selected <= 0:
        evidence = "ENRICHMENT_WITHOUT_POSITIVE_SELECTED_DRIFT"
    else:
        evidence = "INCONCLUSIVE"
    return {"pass": passed, "status": "GO_BOUNDED_INDEPENDENT_PILOT" if passed else "STOP_CURRENT_CANDIDATE_ROUTE",
            "evidence_classification": evidence,
            "point_direction": "positive" if estimate > 0 else "negative" if estimate < 0 else "zero" if estimate == 0 else "nonfinite",
            "checks": checks, "independent_pilot_gpu_hour_cap": 16 if passed else 0,
            "formal_round_cap": 1 if passed else 0, "independent_confirmation": False,
            "fallback_allowed": False, "action_gain_gate_used": False, "global_MSE_gate_used": False,
            "launch_requires": ["frozen_independent_pilot_protocol", "minimum_novelty_mechanism_audit"],
            "interpretation": "development resource gate; failure is not universal impossibility"}
