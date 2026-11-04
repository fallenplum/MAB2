"""CPU-only G2 ridge fitting, label-free prediction and explicitly gated evaluation.

The caller supplies train-fitted [PC1..PC8, 1] features. No encoder, PCA, file
reader, label reader or process control is used by this module.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
import math

import numpy as np

SEEDS = (2701, 2702, 2703, 2704, 2705)
ALPHA_GRID = tuple(10.0 ** exponent for exponent in range(-4, 4))
TIE_ATOL = 1e-12
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 39173
MODEL_NAMES = ("B0", "C1", "C2")
OBJECTIVE = "sum_q mean_unique_mask[(mean_5seed_quality - psi_theta)^2] + alpha * sum_all_theta_squared"


class PredictorError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PredictorError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _mask(value):
    require(isinstance(value, (list, tuple, np.ndarray)) and len(value) == 12,
            "Mask must have twelve binary integer coordinates")
    require(all(isinstance(bit, (int, np.integer)) and not isinstance(bit, (bool, np.bool_))
                and bit in (0, 1) for bit in value), "Mask must have twelve binary integer coordinates")
    return tuple(int(bit) for bit in value)


def graph_features(mask, order=2):
    mask = _mask(mask)
    require(order in (1, 2), "Only first/second-order graph features are frozen")
    values = [1.0, *mask]
    if order == 2:
        values.extend(mask[left] * mask[right] for left, right in itertools.combinations(range(12), 2))
    return np.asarray(values, dtype=np.float64)


def feature_vector(phi, mask, order=2):
    value = np.asarray(phi, dtype=np.float64)
    require(value.shape == (9,) and np.isfinite(value).all() and value[-1] == 1.0,
            "External phi must be finite [PC1..PC8,1] with nine entries")
    return np.kron(value, graph_features(mask, order))


def _resolve_mask(row, masks):
    key = row.get("mask_id", row.get("mask_sha256"))
    direct = row.get("graph_mask")
    require(key is not None or direct is not None, "Quality row lacks mask identity")
    value = _mask(direct) if direct is not None else _mask(masks[key])
    if key is not None and key in masks:
        require(value == _mask(masks[key]), "Mask alias disagrees with graph_mask")
    return value


def aggregate_quality(rows, masks, *, partition=None):
    """Each unique (question, actual mask) is one mean-quality training row.

    Repeated aliases for the same paid (question, mask, seed) are ignored only
    when their quality is identical. Conflicting aliases fail closed.
    """
    grouped = defaultdict(dict)
    aliases_ignored = 0
    for row in rows:
        qid, seed, value = row.get("question_id"), row.get("seed"), row.get("quality")
        require(isinstance(qid, str) and bool(qid), "Quality row lacks question_id")
        require(type(seed) is int and seed in SEEDS, "Quality row has an unplanned seed")
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                "Quality must be a finite value between zero and one")
        if partition is not None and row.get("partition") is not None:
            require(row["partition"] == partition, "Quality row belongs to a different partition")
        key = (qid, _resolve_mask(row, masks))
        if seed in grouped[key]:
            require(grouped[key][seed] == value, "Conflicting same-mask/seed alias qualities")
            aliases_ignored += 1
        else:
            grouped[key][seed] = float(value)
    require(bool(grouped), "No quality rows supplied")
    result = {}
    for key, by_seed in grouped.items():
        require(set(by_seed) == set(SEEDS), "Every unique graph requires all five frozen seeds")
        result[key] = {"quality": math.fsum(by_seed.values()) / len(SEEDS), "by_seed": by_seed}
    return result, aliases_ignored


def normalize_contrasts(contrasts, masks, *, question_ids=None):
    selected = None if question_ids is None else set(question_ids)
    normalized, seen, anchor_candidates = [], set(), {}
    anchors_by_question = defaultdict(set)
    for row in contrasts:
        qid = row.get("question_id")
        if selected is not None and qid not in selected:
            continue
        anchor, edge = row.get("anchor_id"), row.get("edge")
        require(isinstance(qid, str) and qid and isinstance(anchor, str) and anchor,
                "Contrast requires question_id and anchor_id")
        require(type(edge) is int and edge in range(12), "Contrast edge must be one of twelve frozen edges")
        key = (qid, anchor, edge)
        require(key not in seen, "Duplicate question/anchor/edge contrast")
        seen.add(key)
        left = _mask(row["mask0"]) if "mask0" in row else _mask(masks[row["mask0_id"]])
        right = _mask(row["mask1"]) if "mask1" in row else _mask(masks[row["mask1_id"]])
        require(left[edge] == 0 and right[edge] == 1
                and all(left[index] == right[index] for index in range(12) if index != edge),
                "Contrast must be signed e=1 minus e=0 with all other edges fixed")
        normalized.append({"question_id": qid, "anchor_id": anchor, "edge": edge,
                           "mask0": left, "mask1": right})
        anchors_by_question[qid].add(anchor)
        anchor_key = (qid, anchor)
        choices = {left, right}
        anchor_candidates[anchor_key] = anchor_candidates.get(anchor_key, choices) & choices
    require(bool(normalized), "No planned contrasts supplied")
    if selected is not None:
        require(set(anchors_by_question) == selected, "Contrast questions do not cover this partition")
    require(all(len(anchors) == 2 for anchors in anchors_by_question.values()),
            "Each question requires exactly two anchor contexts")
    for qid, anchors in anchors_by_question.items():
        for anchor in anchors:
            require({edge for question, context, edge in seen if question == qid and context == anchor} == set(range(12)),
                    "Every anchor must cover all twelve edges")
            require(len(anchor_candidates[(qid, anchor)]) == 1,
                    "The twelve edge contrasts must share one original anchor graph")
    return sorted(normalized, key=lambda row: (row["question_id"], row["anchor_id"], row["edge"]))


def _check_quality_coverage(quality, contrasts):
    expected = {(row["question_id"], row[name]) for row in contrasts for name in ("mask0", "mask1")}
    require(set(quality) == expected, "Quality graphs differ from the unique frozen contrast masks")


def _contrast_targets(quality, contrasts):
    return np.asarray([
        [quality[(row["question_id"], row["mask1"])]["by_seed"][seed]
         - quality[(row["question_id"], row["mask0"])]["by_seed"][seed] for seed in SEEDS]
        for row in contrasts], dtype=np.float64)


def _training_arrays(quality, phi, order):
    counts = defaultdict(int)
    for qid, mask in quality:
        counts[qid] += 1
    keys = sorted(quality)
    matrix = np.stack([feature_vector(phi[qid], mask, order) for qid, mask in keys])
    targets = np.asarray([quality[key]["quality"] for key in keys], dtype=np.float64)
    weights = np.asarray([1.0 / counts[qid] for qid, mask in keys], dtype=np.float64)
    return matrix, targets, weights, counts


def ridge_path(matrix, targets, weights, alphas=ALPHA_GRID):
    """Minimize sum_i w_i(y_i-x_i theta)^2 + alpha ||theta||^2.

    Every coefficient, including both graph/question constant features, receives
    the same L2 penalty. SVD keeps rank-deficient alias designs well-defined.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    require(matrix.ndim == 2 and len(matrix) > 0 and targets.shape == weights.shape == (len(matrix),)
            and np.isfinite(matrix).all() and np.isfinite(targets).all()
            and np.isfinite(weights).all() and np.all(weights > 0), "Invalid finite weighted regression arrays")
    require(all(type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0 for alpha in alphas),
            "Ridge alphas must be finite and positive")
    scale = np.sqrt(weights)
    left, singular, right = np.linalg.svd(matrix * scale[:, None], full_matrices=False)
    projected = left.T @ (targets * scale)
    return {float(alpha): right.T @ (singular * projected / (singular * singular + alpha)) for alpha in alphas}


def _contrast_features(contrasts, phi, order):
    return np.stack([feature_vector(phi[row["question_id"]], row["mask1"], order)
                     - feature_vector(phi[row["question_id"]], row["mask0"], order) for row in contrasts])


def _question_mse(predicted, targets, contrasts):
    losses = defaultdict(list)
    for row, prediction, target in zip(contrasts, predicted, targets):
        losses[row["question_id"]].append((float(prediction) - float(target)) ** 2)
    require(all(len(values) == 24 for values in losses.values()), "Tune risk requires 24 contrasts per question")
    return math.fsum(math.fsum(values) / 24 for values in losses.values()) / len(losses)


def select_alpha(scores):
    require(bool(scores) and all(math.isfinite(value) for value in scores.values()), "Invalid tune alpha scores")
    best = min(scores.values())
    return max(alpha for alpha, value in scores.items() if value <= best + TIE_ATOL)


def fit_and_select(train_rows, tune_rows, phi, masks, contrasts):
    train, train_aliases = aggregate_quality(train_rows, masks, partition="train")
    tune, tune_aliases = aggregate_quality(tune_rows, masks, partition="tune")
    train_ids, tune_ids = {key[0] for key in train}, {key[0] for key in tune}
    require(not train_ids & tune_ids, "Training and tuning questions must be disjoint")
    train_contrasts = normalize_contrasts(contrasts, masks, question_ids=train_ids)
    tune_contrasts = normalize_contrasts(contrasts, masks, question_ids=tune_ids)
    _check_quality_coverage(train, train_contrasts)
    _check_quality_coverage(tune, tune_contrasts)
    train_credits = _contrast_targets(train, train_contrasts)
    tune_mean_credits = _contrast_targets(tune, tune_contrasts).mean(axis=1)
    baseline = []
    for edge in range(12):
        rows = [index for index, row in enumerate(train_contrasts) if row["edge"] == edge]
        baseline.append(float(train_credits[rows].mean()))
    models = {}
    for name, order in (("C1", 1), ("C2", 2)):
        matrix, targets, weights, counts = _training_arrays(train, phi, order)
        coefficients = ridge_path(matrix, targets, weights)
        tune_features = _contrast_features(tune_contrasts, phi, order)
        scores = {alpha: _question_mse(tune_features @ theta, tune_mean_credits, tune_contrasts)
                  for alpha, theta in coefficients.items()}
        selected = select_alpha(scores)
        models[name] = {"order": order, "dimensions": matrix.shape[1], "alpha": selected,
                        "coefficients": coefficients[selected].tolist(),
                        "tune_credit_mse": scores[selected],
                        "alpha_scores": [{"alpha": alpha, "tune_credit_mse": scores[alpha]} for alpha in ALPHA_GRID]}
    result = {"schema": "graph_credit_g2_predictor_v1", "objective": OBJECTIVE,
              "per_question_weight": "1 / unique_training_masks_for_question",
              "phi_order": "PC1..PC8,1", "graph_feature_order": "1,m0..m11,mi*mj for i<j lexicographic",
              "tensor_order": "phi_major_kron", "all_coefficients_penalized": True,
              "seeds": list(SEEDS), "alpha_grid": list(ALPHA_GRID), "alpha_tie_atol": TIE_ATOL,
              "B0": baseline, "models": models, "train_question_ids": sorted(train_ids),
              "tune_question_ids": sorted(tune_ids), "train_unique_graph_rows": len(train),
              "tune_unique_graph_rows": len(tune), "train_alias_rows_ignored": train_aliases,
              "tune_alias_rows_ignored": tune_aliases,
              "unique_training_masks_by_question": dict(sorted(counts.items()))}
    result["parameter_sha256"] = digest(result)
    return result


def predict(parameters, phi, masks, contrasts):
    """Predict signed credits with quality-free question/mask metadata only."""
    expected_hash = parameters.get("parameter_sha256")
    require(expected_hash == digest({key: value for key, value in parameters.items() if key != "parameter_sha256"}),
            "Frozen predictor parameters changed")
    require(parameters.get("schema") == "graph_credit_g2_predictor_v1", "Unsupported predictor schema")
    rows = normalize_contrasts(contrasts, masks)
    fitted_ids = set(parameters["train_question_ids"]) | set(parameters["tune_question_ids"])
    require(not {row["question_id"] for row in rows} & fitted_ids,
            "Heldout prediction questions overlap training or tuning")
    values = {}
    for name, order in (("C1", 1), ("C2", 2)):
        theta = np.asarray(parameters["models"][name]["coefficients"], dtype=np.float64)
        features = _contrast_features(rows, phi, order)
        require(theta.shape == (features.shape[1],) and np.isfinite(theta).all(), "Invalid frozen coefficient dimensions")
        values[name] = features @ theta
    baseline = np.asarray(parameters["B0"], dtype=np.float64)
    require(baseline.shape == (12,) and np.isfinite(baseline).all(), "Invalid unconditional baseline")
    return [{"question_id": row["question_id"], "anchor_id": row["anchor_id"], "edge": row["edge"],
             "mask0": list(row["mask0"]), "mask1": list(row["mask1"]),
             "predictions": {"B0": float(baseline[row["edge"]]), "C1": float(values["C1"][index]),
                             "C2": float(values["C2"][index])},
             "parameter_sha256": expected_hash} for index, row in enumerate(rows)]


def question_bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), "At least two finite question blocks required")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 256):
        stop = min(start + 256, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[indices].mean(axis=1)
    return {"estimate": float(values.mean()), "interval": np.quantile(samples, [.025, .975]).tolist(),
            "question_count": len(values), "resampling_unit": "whole_question", "confidence": .95,
            "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED}


def evaluate(test_qualities, predictions, masks, *, prediction_freeze):
    """Open supplied heldout qualities only after verifying the frozen predictions.

    The production caller owns the durable marker and independent root check.
    This pure function verifies their state/hash before iterating test_qualities.
    """
    require(isinstance(prediction_freeze, dict) and prediction_freeze.get("state") == "PREDICTIONS_FROZEN",
            "Heldout evaluation requires PREDICTIONS_FROZEN")
    require(prediction_freeze.get("predictions_sha256") == digest(predictions), "Frozen heldout predictions hash mismatch")
    contrasts = normalize_contrasts(predictions, masks)
    require(len(predictions) == len(contrasts), "Prediction support mismatch")
    by_key = {(row["question_id"], row["anchor_id"], row["edge"]): row for row in predictions}
    quality, aliases = aggregate_quality(test_qualities, masks, partition="test")
    _check_quality_coverage(quality, contrasts)
    credits = _contrast_targets(quality, contrasts)
    risks, naive_mse = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for row, targets in zip(contrasts, credits):
        key = (row["question_id"], row["anchor_id"], row["edge"])
        stored = by_key[key]["predictions"]
        require(set(stored) == set(MODEL_NAMES), "Every heldout contrast requires B0,C1,C2 predictions")
        for model in MODEL_NAMES:
            prediction = stored[model]
            require(type(prediction) in (int, float) and math.isfinite(prediction), "Nonfinite heldout prediction")
            errors = prediction - targets
            risk = math.fsum(float(left * right) for left, right in itertools.combinations(errors, 2)) / 10
            risks[row["question_id"]][model].append(risk)
            naive_mse[row["question_id"]][model].append(float((prediction - targets.mean()) ** 2))
    question_risks, deltas, diagnostics = {}, [], []
    for qid in sorted(risks):
        question_risks[qid] = {model: math.fsum(risks[qid][model]) / 24 for model in MODEL_NAMES}
        deltas.append(question_risks[qid]["B0"] - question_risks[qid]["C2"])
        diagnostics.append(question_risks[qid]["C1"] - question_risks[qid]["C2"])
        naive_delta = math.fsum(naive_mse[qid]["B0"]) / 24 - math.fsum(naive_mse[qid]["C2"]) / 24
        require(math.isclose(deltas[-1], naive_delta, rel_tol=1e-10, abs_tol=1e-12), "Paired noise cancellation identity failed")
    primary, diagnostic = question_bootstrap(deltas), question_bootstrap(diagnostics)
    return {"schema": "graph_credit_g2_heldout_evaluation_v1", "predictions_sha256": digest(predictions),
            "scientific_gate": "G2_PASS" if primary["interval"][0] > 0 else "INSUFFICIENT_PREDICTIVE_GAIN",
            "primary": {"comparison": "B0_risk_minus_C2_risk", **primary},
            "first_order_diagnostic": {"comparison": "C1_risk_minus_C2_risk", **diagnostic},
            "mean_cross_seed_risk": {model: math.fsum(v[model] for v in question_risks.values()) / len(question_risks) for model in MODEL_NAMES},
            "question_risks": question_risks, "negative_risks_preserved": True,
            "contrasts_per_question": 24, "quality_seed_count": 5, "test_alias_rows_ignored": aliases,
            "scope": "G2 heldout graph-conditional credit prediction gate", "scientific_completion": False}
