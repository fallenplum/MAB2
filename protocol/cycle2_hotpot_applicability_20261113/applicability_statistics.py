"""Frozen-shape, pure-array candidate analysis. No file, label or model access.

Input metric dictionaries each contain graphs [384,2,4] in A,B,A-e,B-e
order, and no_edge/independent12/single [384] at seed1. EM must be binary.
Rows follow the frozen selection manifest. Guards received here must come from
an independently bound archive audit; this module does not certify them.
"""
from __future__ import annotations

import hashlib
import numpy as np

N = 384
EDGES = (1, 5, 7, 9)
BOOTSTRAP_SEED = 261113103
BOOTSTRAP_REPETITIONS = 10000
QUANTILES = (0.00625, 0.99375)
NAMES = (
    "B_MINUS_NO_EDGE_EM", "B_MINUS_INDEPENDENT12_EM",
    "MEAN_SIGNED_DELTA_EM", "MEAN_CROSSSEED_DELTA_PRODUCT_EM",
)


def _validated_metric(metric: dict, binary: bool) -> dict:
    if set(metric) != {"graphs", "no_edge", "independent12", "single"}:
        raise ValueError("metric requires exactly graphs/no_edge/independent12/single")
    out = {}
    for name, shape in (("graphs", (N, 2, 4)), ("no_edge", (N,)),
                        ("independent12", (N,)), ("single", (N,))):
        value = np.asarray(metric[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name}: invalid shape or nonfinite values")
        if ((value < 0) | (value > 1)).any():
            raise ValueError(f"{name}: scores outside [0,1]")
        if binary and not ((value == 0) | (value == 1)).all():
            raise ValueError(f"{name}: EM must be binary")
        out[name] = value
    return out


def _question_statistics(metric: dict) -> tuple[np.ndarray, np.ndarray]:
    graph = metric["graphs"]
    gamma_a = graph[:, :, 0] - graph[:, :, 2]
    gamma_b = graph[:, :, 1] - graph[:, :, 3]
    delta = gamma_b - gamma_a
    b = graph[:, 0, 1]
    primary = np.column_stack((b - metric["no_edge"],
                               b - metric["independent12"],
                               delta.mean(axis=1), delta[:, 0] * delta[:, 1]))
    secondary = np.column_stack((b - metric["single"],
                                 gamma_a[:, 0], gamma_a[:, 1],
                                 gamma_b[:, 0], gamma_b[:, 1],
                                 delta[:, 0], delta[:, 1]))
    return primary, secondary


def _indices(edge_indices: np.ndarray) -> tuple[np.ndarray, str]:
    edge = np.asarray(edge_indices)
    if edge.shape != (N,) or not np.issubdtype(edge.dtype, np.integer):
        raise ValueError("edge indices must be 384 integers")
    if set(edge.tolist()) != set(EDGES):
        raise ValueError("unexpected edge index")
    strata = [np.flatnonzero(edge == e) for e in EDGES]
    if any(len(s) != 96 for s in strata):
        raise ValueError("exactly 96 questions per edge required")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draw = rng.integers(0, 96, size=(BOOTSTRAP_REPETITIONS, 4, 96))
    index = np.concatenate([s[draw[:, j, :]] for j, s in enumerate(strata)], axis=1)
    digest = hashlib.sha256(np.asarray(index, dtype="<i8").tobytes(order="C")).hexdigest()
    return index, digest


def _summary(values: np.ndarray, index: np.ndarray, quantiles: tuple) -> dict:
    means = values[index].mean(axis=1)
    interval = np.quantile(means, quantiles, method="linear")
    nonzero = int(np.count_nonzero(values))
    variance = float(np.var(values, ddof=1))
    bootstrap_variance = float(np.var(means, ddof=1))
    # Fixed strata can each be constant although the pooled sample varies.
    # ptp avoids a tiny positive variance from centering identical floats.
    resampling_nondegenerate = bool(np.ptp(means) > 0 and interval[1] > interval[0])
    return {
        "estimate": float(values.mean()), "interval": interval.tolist(),
        "nonzero_questions": nonzero,
        "positive_questions": int(np.count_nonzero(values > 0)),
        "negative_questions": int(np.count_nonzero(values < 0)),
        "sample_variance": variance,
        "bootstrap_estimator_variance": bootstrap_variance,
        "bootstrap_nondegenerate": resampling_nondegenerate,
        "support_guard_pass": bool(nonzero >= 20 and variance > 0 and resampling_nondegenerate),
        "interval_width": float(interval[1] - interval[0]),
    }


def _classify(primary: dict, overflow: int, cost_frontier: str) -> dict:
    quality = [primary[NAMES[0]], primary[NAMES[1]]]
    both = all(x["support_guard_pass"] and x["interval"][0] > 0 for x in quality)
    disfavored = any(x["support_guard_pass"] and x["interval"][1] < 0 for x in quality)
    l1 = "SUPPORTED_BOTH" if both else ("B_DISFAVORED" if disfavored else "INCONCLUSIVE")
    d, theta = primary[NAMES[2]], primary[NAMES[3]]
    direction = "INCONCLUSIVE"
    if d["support_guard_pass"] and d["interval"][0] > 0:
        direction = "REDUNDANCY_DIRECTION_SUPPORTED"
    elif d["support_guard_pass"] and d["interval"][1] < 0:
        direction = "REDUNDANCY_DIRECTION_REVERSED"
    theta_supported = bool(theta["support_guard_pass"] and theta["interval"][0] > 0)
    l2_supported = direction != "INCONCLUSIVE" or theta_supported
    l2 = "GRAPH_CONDITIONAL_SIGNAL_SUPPORTED" if l2_supported else "INCONCLUSIVE"
    if overflow:
        action = "NO_AUTOMATIC_UPGRADE_CAPACITY_FAILURES"
    elif disfavored or cost_frontier == "DOMINATED":
        action = "STOP_CURRENT_B_DIRECTION"
    elif both and l2_supported:
        action = ("CONSIDER_INDEPENDENT_MECHANISM_PROTOCOL"
                  if cost_frontier == "NOT_DOMINATED"
                  else "PENDING_COST_FRONTIER_REVIEW")
    elif both:
        action = "STATIC_COMMUNICATION_SUPPORTED_REVALIDATION_UNRESOLVED"
    elif l2_supported:
        action = "CONDITIONAL_RESPONSE_ONLY_NO_SYSTEM_SUCCESS"
    else:
        action = "EVIDENCE_INSUFFICIENT_STOP_AUTOMATIC_UPGRADE"
    return {
        "L1": l1, "L2": l2, "signed_direction": direction,
        "theta_nonzero_supported": theta_supported,
        "theta_negative_point_retained": theta["estimate"] < 0,
        "theta_small_effect_exclusion": "NOT_PERMITTED_BY_THIS_BOOTSTRAP_CONTRACT",
        "theta_width_divided_by_0_0025": theta["interval_width"] / 0.0025,
        "quality_scale_tags": [
            {"point_at_least_5pp": x["estimate"] >= .05,
             "interval_lower_at_least_5pp": x["support_guard_pass"] and x["interval"][0] >= .05,
             "approx_within_plus_minus_2pp": x["support_guard_pass"] and
             x["interval"][0] >= -.02 and x["interval"][1] <= .02}
            for x in quality],
        "action": action, "GPU_admission_granted": False,
        "method_success_claim": False,
    }


def analyze(em: dict, f1: dict, edge_indices: np.ndarray, audit: dict) -> dict:
    """Analyze the fixed 384-question plan after an independent archive audit.

    audit keys: integrity_passed=True, unique_questions=384,
    complete_conditions=4224, runtime_failure_count=0,
    input_overflow_conditions>=0, cost_frontier_review in
    NOT_ASSESSED/NOT_DOMINATED/DOMINATED. No file IO is performed.
    """
    expected = {"integrity_passed", "unique_questions", "complete_conditions",
                "runtime_failure_count", "input_overflow_conditions", "cost_frontier_review"}
    if set(audit) != expected:
        raise ValueError("audit schema mismatch")
    if audit["integrity_passed"] is not True or audit["unique_questions"] != N:
        raise ValueError("integrity or unique-question coverage not complete")
    if audit["complete_conditions"] != 4224 or audit["runtime_failure_count"] != 0:
        raise ValueError("incomplete condition coverage or runtime failures")
    overflow = audit["input_overflow_conditions"]
    if type(overflow) is not int or not 0 <= overflow <= 4224:
        raise ValueError("invalid overflow count")
    frontier = audit["cost_frontier_review"]
    if frontier not in {"NOT_ASSESSED", "NOT_DOMINATED", "DOMINATED"}:
        raise ValueError("invalid cost-frontier status")
    em = _validated_metric(em, binary=True)
    f1 = _validated_metric(f1, binary=False)
    primary, secondary = _question_statistics(em)
    fprimary, fsecondary = _question_statistics(f1)
    index, digest = _indices(edge_indices)
    results = {name: _summary(primary[:, j], index, QUANTILES)
               for j, name in enumerate(NAMES)}
    secondary_names = ("B_MINUS_SINGLE", "GAMMA_A_SEED1", "GAMMA_A_SEED2",
                       "GAMMA_B_SEED1", "GAMMA_B_SEED2", "DELTA_SEED1", "DELTA_SEED2")
    descriptive = {
        "EM": {name: _summary(secondary[:, j], index, (.025, .975))
               for j, name in enumerate(secondary_names)},
        "F1_main_forms": {name.replace("_EM", "_F1"): _summary(fprimary[:, j], index, (.025, .975))
                          for j, name in enumerate(NAMES)},
        "F1_other": {name: _summary(fsecondary[:, j], index, (.025, .975))
                     for j, name in enumerate(secondary_names)},
    }
    edge = np.asarray(edge_indices)
    return {
        "protocol_status": "CANDIDATE_ANALYSIS_NO_ADMISSION",
        "primary_EM": results, "secondary_descriptive_only": descriptive,
        "per_edge_primary_means_descriptive": {
            str(e): dict(zip(NAMES, primary[edge == e].mean(axis=0).tolist())) for e in EDGES},
        "decision": _classify(results, overflow, frontier),
        "bootstrap": {"rng": "PCG64", "seed": BOOTSTRAP_SEED,
                      "repetitions": BOOTSTRAP_REPETITIONS,
                      "strata": list(EDGES), "counts": [96]*4,
                      "primary_quantiles": list(QUANTILES), "quantile_method": "linear",
                      "draw_index_sha256": digest, "family_size": 4, "family_alpha": .05,
                      "coverage_claim": "approximate bootstrap, not finite-sample guarantee"},
        "audit_inputs_not_independently_verified_by_this_module": dict(audit),
    }
