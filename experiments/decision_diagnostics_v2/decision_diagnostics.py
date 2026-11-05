"""Pure CPU, aggregate-only exploratory diagnostics of one graph toggle.

Inputs contain frozen credit predictions and five observed signed credit
differences. No fitting, file access, label opening or sequential rollout occurs.
"""
from __future__ import annotations

import math

import numpy as np


MODELS = ("B0", "C1", "C2")
POLICIES = (*MODELS, "ZERO_STAY", "UNIFORM_13_ACTIONS")
SIGNS = ("negative", "zero", "positive")
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 49173


def _numeric_array(value, name, *, allow_bool=False):
    array = np.asarray(value)
    kinds = "biuf" if allow_bool else "iuf"
    if array.dtype.kind not in kinds:
        raise ValueError(name + " must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(name + " must contain finite values")
    return array


def _validated_inputs(predictions, observed_differences, anchor_bits):
    predictions = _numeric_array(predictions, "predictions")
    observed = _numeric_array(observed_differences, "observed_differences")
    bits = _numeric_array(anchor_bits, "anchor_bits", allow_bool=True)
    if (predictions.ndim != 4 or predictions.shape[0] < 1
            or predictions.shape[1:] != (2, 12, 3)):
        raise ValueError("predictions must have shape [Q,2,12,3] with Q >= 1")
    if observed.shape != (*predictions.shape[:3], 5):
        raise ValueError("observed_differences must have shape [Q,2,12,5]")
    if bits.shape != predictions.shape[:3] or not np.isin(bits, (0, 1)).all():
        raise ValueError("anchor_bits must have shape [Q,2,12] and contain only 0 or 1")
    return predictions, observed, bits.astype(np.int8)


def _sign_confusion(predictions, observed_mean):
    observed_sign = np.sign(observed_mean).astype(np.int8).ravel() + 1
    result = {}
    for column, name in enumerate(MODELS):
        predicted_sign = np.sign(predictions[..., column]).astype(np.int8).ravel() + 1
        counts = np.bincount(observed_sign * 3 + predicted_sign, minlength=9).reshape(3, 3)
        observed_support = counts.sum(axis=1)
        predicted_support = counts.sum(axis=0)
        agreement = int(np.trace(counts))
        result[name] = {
            "confusion_counts": counts.tolist(),
            "observed_support": dict(zip(SIGNS, map(int, observed_support))),
            "predicted_support": dict(zip(SIGNS, map(int, predicted_support))),
            "agreement_count": agreement,
            "agreement_rate": agreement / len(observed_sign),
            "opposite_nonzero_sign_count": int(counts[0, 2] + counts[2, 0]),
            "by_observed_sign": {sign: {"support": int(observed_support[index]),
                "agreement_count": int(counts[index, index]),
                "agreement_rate": float(counts[index, index] / observed_support[index])
                    if observed_support[index] else None} for index, sign in enumerate(SIGNS)},
        }
    return {"class_order": list(SIGNS), "rows": "observed_five_seed_mean_sign",
            "columns": "predicted_existence_credit_sign", "coordinates": len(observed_sign),
            "threshold": 0.0, "zero_rule": "exact floating-point equality",
            "observed_sign_is_noisy_proxy": True, "true_credit_sign_claimed": False,
            "models": result}


def _average_ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    ranks = starts + (counts - 1) / 2 + 1
    return ranks[inverse]


def _spearman_record(left, right):
    left_rank, right_rank = _average_ranks(left), _average_ranks(right)
    left_rank = left_rank - left_rank.mean()
    right_rank = right_rank - right_rank.mean()
    left_constant, right_constant = not left_rank.any(), not right_rank.any()
    correlation = None
    if not (left_constant or right_constant):
        correlation = float((left_rank @ right_rank) / (np.linalg.norm(left_rank) * np.linalg.norm(right_rank)))
    return correlation, left_constant, right_constant


def _summarize_correlations(records):
    defined = [value for value, _, _ in records if value is not None]
    return {"question_anchor_pairs": len(records), "defined_pairs": len(defined),
            "undefined_pairs": len(records) - len(defined),
            "constant_prediction_pairs": sum(left for _, left, _ in records),
            "constant_observed_proxy_pairs": sum(right for _, _, right in records),
            "both_constant_pairs": sum(left and right for _, left, right in records),
            "mean_defined_spearman": math.fsum(defined) / len(defined) if defined else None,
            "positive_defined_pairs": sum(value > 0 for value in defined),
            "negative_defined_pairs": sum(value < 0 for value in defined),
            "zero_defined_pairs": sum(value == 0 for value in defined)}


def _spearman_summary(predictions, observed_mean):
    models = {}
    for column, name in enumerate(MODELS):
        anchors = [[_spearman_record(predictions[q, anchor, :, column], observed_mean[q, anchor])
                    for q in range(len(predictions))] for anchor in range(2)]
        models[name] = {"overall": _summarize_correlations(anchors[0] + anchors[1]),
                        "by_anchor_index": [_summarize_correlations(records) for records in anchors]}
    return {"ranking_target": "twelve signed existence credits before multiplying by toggle direction",
            "tie_rank": "average", "aggregation": "equal weight over defined question-anchor correlations",
            "constant_vectors": "undefined, excluded from the mean and counted explicitly", "models": models}


def _greedy_action(gains):
    edge = np.argmax(gains, axis=-1)
    best = np.take_along_axis(gains, edge[..., None], axis=-1)[..., 0]
    return np.where(best > 0, edge, 12)


def _selection_support(actions, bits):
    total = actions.size
    stay = actions == 12
    selected_bits = np.take_along_axis(bits, np.minimum(actions, 11)[..., None], axis=-1)[..., 0]
    additions = int(((selected_bits == 0) & ~stay).sum())
    deletions = int(((selected_bits == 1) & ~stay).sum())
    stay_count = int(stay.sum())
    counts = np.bincount(actions.ravel(), minlength=13)
    return {"count_kind": "deterministic_selected_action", "question_anchor_decisions": total,
            "toggle_count": additions + deletions, "addition_count": additions,
            "deletion_count": deletions, "stay_count": stay_count,
            "toggle_rate": (additions + deletions) / total, "addition_rate": additions / total,
            "deletion_rate": deletions / total, "stay_rate": stay_count / total,
            "action_counts_edge0_through_edge11_then_stay": counts.tolist(),
            "action_rates_edge0_through_edge11_then_stay": (counts / total).tolist()}


def _uniform_support(bits):
    total = int(np.prod(bits.shape[:2]))
    additions, deletions = float((bits == 0).sum() / 13), float((bits == 1).sum() / 13)
    return {"count_kind": "expected_under_uniform_draw_over_all_13_actions",
            "question_anchor_decisions": total, "toggle_count": total * 12 / 13,
            "addition_count": additions, "deletion_count": deletions, "stay_count": total / 13,
            "toggle_rate": 12 / 13, "addition_rate": additions / total,
            "deletion_rate": deletions / total, "stay_rate": 1 / 13,
            "action_counts_edge0_through_edge11_then_stay": [total / 13] * 13,
            "action_rates_edge0_through_edge11_then_stay": [1 / 13] * 13}


def _paired_bootstrap(question_gains):
    comparisons = [(name + "_minus_ZERO_STAY", column, 3) for column, name in enumerate(MODELS)]
    comparisons += [(name + "_minus_UNIFORM_13_ACTIONS", column, 4) for column, name in enumerate(MODELS)]
    comparisons += [("C2_minus_B0", 2, 0), ("C2_minus_C1", 2, 1)]
    differences = np.stack([question_gains[:, left] - question_gains[:, right]
                            for _, left, right in comparisons], axis=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty((BOOTSTRAP_SAMPLES, len(comparisons)), dtype=np.float64)
    question_count = len(question_gains)
    for start in range(0, BOOTSTRAP_SAMPLES, 256):
        stop = min(start + 256, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, question_count, size=(stop - start, question_count))
        samples[start:stop] = differences[indices].mean(axis=1)
    return {name: {"estimate": math.fsum(differences[:, column]) / question_count,
                   "interval": np.quantile(samples[:, column], [.025, .975], method="linear").tolist(),
                   "question_count": question_count, "confidence": .95,
                   "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                   "resampling_unit": "whole_question_with_both_anchors",
                   "exploratory": True, "multiplicity_adjustment": "none"}
            for column, (name, _, _) in enumerate(comparisons)}


def decision_diagnostics(predictions, observed_differences, anchor_bits):
    """Return JSON-serializable aggregates for a fixed one-toggle-or-stay menu.

    ``predictions[q,a,e,:]`` follows B0,C1,C2 order. Observed differences are
    always Q_s(mask1)-Q_s(mask0), with five fixed seed observations. Inputs remain
    unchanged. No per-question values, identifiers or fitted parameters return.
    """
    predictions, observed, bits = _validated_inputs(predictions, observed_differences, anchor_bits)
    observed_mean = observed.mean(axis=-1)
    direction = 1 - 2 * bits
    observed_gain = direction * observed_mean
    gain_menu = np.concatenate((observed_gain, np.zeros((*bits.shape[:2], 1))), axis=-1)
    selections, gains = {}, []
    for column, name in enumerate(MODELS):
        action = _greedy_action(direction * predictions[..., column])
        selections[name] = _selection_support(action, bits)
        gains.append(np.take_along_axis(gain_menu, action[..., None], axis=-1)[..., 0])
    selections["ZERO_STAY"] = _selection_support(np.full(bits.shape[:2], 12), bits)
    gains.append(np.zeros(bits.shape[:2]))
    selections["UNIFORM_13_ACTIONS"] = _uniform_support(bits)
    gains.append(gain_menu.mean(axis=-1))
    question_gains = np.stack(gains, axis=-1).mean(axis=1)
    if not np.isfinite(question_gains).all():
        raise ValueError("Gain arithmetic is not finite")
    policies = {name: {"mean_observed_gain": math.fsum(question_gains[:, column]) / len(bits),
                       "selection_support": selections[name],
                       "gain_evaluation": "exact uniform-action expected gain" if name == "UNIFORM_13_ACTIONS"
                           else "observed gain of the deterministic selected action",
                       "negative_gain_question_anchor_pairs": int((gains[column] < 0).sum()),
                       "zero_gain_question_anchor_pairs": int((gains[column] == 0).sum()),
                       "positive_gain_question_anchor_pairs": int((gains[column] > 0).sum())}
                for column, name in enumerate(POLICIES)}
    oracle_action = _greedy_action(observed_gain)
    oracle_gain = gain_menu.max(axis=-1).mean(axis=1)
    return {"schema": "graph_credit_exploratory_decision_diagnostics_v1",
            "scope": "one observed single-toggle-or-stay quality diagnostic",
            "question_count": len(bits), "anchors_per_question": 2, "edges_per_anchor": 12,
            "quality_seeds": 5, "model_order": list(MODELS), "contains_individual_identifiers": False,
            "observed_target": "Q_s(mask1)-Q_s(mask0), averaged over five fixed observed seeds",
            "action_rule": {"gain_formula": "(1-2*anchor_bit)*predicted_existence_credit",
                            "threshold": 0.0, "selection": "largest strictly positive gain, otherwise stay",
                            "tie_break": "smallest edge index", "stay_action_index": 12,
                            "maximum_toggles": 1, "quality_cost_lambda": 0.0},
            "sign_diagnostics": _sign_confusion(predictions, observed_mean),
            "spearman": _spearman_summary(predictions, observed_mean),
            "policies": policies, "paired_comparisons": _paired_bootstrap(question_gains),
            "observed_oracle": {"mean_observed_selected_max_gain": math.fsum(oracle_gain) / len(bits),
                                "selection_support": _selection_support(oracle_action, bits),
                                "rule": "max(0,max of twelve observed five-seed-mean toggle gains)",
                                "maximum_selection_bias_present": True,
                                "scope": "descriptive observed-menu upper bound",
                                "true_oracle_or_true_regret_claimed": False},
            "limitations": {"exploratory": True, "multiplicity_adjustment": "none",
                            "inference_scope": "exploratory development diagnostics, no new heldout gate",
                            "observations_are_noisy_proxies": True, "negative_values_preserved": True,
                            "net_budget_benefit_claimed": False, "sequential_rollout_claimed": False,
                            "random_baseline": "expectation over all twelve toggles and stay, without filtering",
                            "bootstrap_degenerate_single_question": len(bits) == 1}}
