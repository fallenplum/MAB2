"""Post-hoc diagnostics of precomputed, uncalibrated ridge leverage proxies."""
from __future__ import annotations

import numpy as np


MARGIN_DENOMINATOR_FLOOR = 1e-12


def _average_ranks(values):
    """Return one-based ascending ranks with the average rank for every tie."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    average_ranks = np.cumsum(counts) - (counts - 1) / 2.0
    return average_ranks[inverse]


def _spearman(left, right):
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return None
    correlation = np.dot(left_centered, right_centered) / denominator
    return float(np.clip(correlation, -1.0, 1.0))


def _roc_auc(positive, scores):
    """Mann-Whitney ROC AUC, assigning one half to each tied score pair."""
    positive_count = int(positive.sum())
    negative_count = len(positive) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    positive_rank_sum = _average_ranks(scores)[positive].sum()
    wins = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(wins / (positive_count * negative_count))


def uncertainty_diagnostics(predictions, targets, leverage):
    """Summarize C1/C2 proxies without fitting, selecting, or calibrating a model.

    Inputs are real finite arrays with shapes ``[Q,2,12,3]`` for predictions
    ordered B0/C1/C2, ``[Q,2,12,5]`` for observed seed credit differences, and
    ``[Q,2,12,2]`` for nonnegative proxies ordered C1/C2. Q must be positive.
    The caller supplies ``sqrt(d.T @ inv(X.T @ W @ X + alpha*I) @ d)``.

    Each coordinate has equal weight. Spearman uses all coordinates. Both AUCs
    use only coordinates whose observed five-seed mean is strictly nonzero.
    Their positive class is ``sign(prediction) != sign(observed_mean)``. Thus a
    zero prediction disagrees with a nonzero observed mean. Larger scores
    identify that same noisy observed-sign disagreement. All score ties receive
    average ranks. Constant-input Spearman and missing-class AUC return None.

    The observed targets are noisy and correlated within question. This function
    produces descriptive post-hoc associations, with no confidence calibration,
    true-sign error-rate claim, certificate, model selection, or scientific gate.
    """
    raw_arrays = tuple(np.asarray(value) for value in (predictions, targets, leverage))
    if any(np.iscomplexobj(value) for value in raw_arrays):
        raise ValueError("Diagnostic inputs must be real arrays")
    predictions_array, targets_array, leverage_array = (
        np.asarray(value, dtype=float) for value in raw_arrays
    )
    question_count = predictions_array.shape[0] if predictions_array.ndim else 0
    expected_shapes = (
        (question_count, 2, 12, 3),
        (question_count, 2, 12, 5),
        (question_count, 2, 12, 2),
    )
    arrays = (predictions_array, targets_array, leverage_array)
    if question_count < 1 or any(value.shape != shape for value, shape in zip(arrays, expected_shapes)):
        raise ValueError("Expected predictions[Q,2,12,3], targets[Q,2,12,5], leverage[Q,2,12,2], Q>=1")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("Diagnostic inputs must be finite")
    if np.any(leverage_array < 0):
        raise ValueError("Ridge leverage square-root proxies must be nonnegative")

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            observed_mean = targets_array.mean(axis=-1)
            nonzero = observed_mean != 0
            selected_coordinates = int(nonzero.sum())
            coordinate_count = int(observed_mean.size)
            models = {}
            for model_index, name in enumerate(("C1", "C2")):
                prediction = predictions_array[..., model_index + 1]
                model_leverage = leverage_array[..., model_index]
                absolute_residual = np.abs(prediction - observed_mean)
                disagreements = np.sign(prediction[nonzero]) != np.sign(observed_mean[nonzero])
                positive_count = int(disagreements.sum())
                margin_score = -np.abs(prediction[nonzero]) / np.maximum(
                    model_leverage[nonzero], MARGIN_DENOMINATOR_FLOOR
                )
                models[name] = {
                    "leverage": {
                        "minimum": float(model_leverage.min()),
                        "maximum": float(model_leverage.max()),
                        "mean": float(model_leverage.mean()),
                        "coordinates": coordinate_count,
                    },
                    "leverage_absolute_observed_residual_spearman": _spearman(
                        model_leverage.ravel(), absolute_residual.ravel()
                    ),
                    "observed_nonzero_mean_sign_diagnostic": {
                        "coordinates": selected_coordinates,
                        "excluded_zero_observed_mean_coordinates": coordinate_count - selected_coordinates,
                        "positive_observed_sign_disagreements": positive_count,
                        "negative_observed_sign_agreements": selected_coordinates - positive_count,
                        "leverage_roc_auc": _roc_auc(disagreements, model_leverage[nonzero]),
                        "negative_normalized_margin_roc_auc": _roc_auc(disagreements, margin_score),
                    },
                }
    except FloatingPointError as error:
        raise ValueError("Diagnostic arithmetic exceeded the finite floating-point range") from error

    return {
        "schema": "graph_credit_decision_uncertainty_diagnostics_v1",
        "question_count": question_count,
        "coordinates": coordinate_count,
        "prediction_model_order": ["B0", "C1", "C2"],
        "leverage_model_order": ["C1", "C2"],
        "B0_leverage_defined": False,
        "proxy_formula": "sqrt(d^T (X^T W X + alpha I)^-1 d)",
        "margin_score_formula": "-abs(prediction)/max(leverage,1e-12)",
        "margin_denominator_floor": MARGIN_DENOMINATOR_FLOOR,
        "auc_positive_class": "sign(prediction) differs from sign(observed five-seed mean)",
        "auc_selection": "observed five-seed mean != 0, with no tolerance",
        "auc_score_direction": "larger score identifies an observed sign disagreement",
        "tie_rule": "average ranks, with one half credit for each tied positive-negative AUC pair",
        "scope": "post-hoc descriptive diagnostic on caller-supplied pre-existing noisy targets",
        "coordinates_correlated_within_question": True,
        "confidence_calibrated": False,
        "true_sign_error_rate_estimated": False,
        "certificate_provided": False,
        "model_selection_performed": False,
        "scientific_gate_defined": False,
        "models": models,
    }
