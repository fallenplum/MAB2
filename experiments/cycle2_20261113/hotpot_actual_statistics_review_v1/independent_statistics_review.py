"""Independent read-only replay of the released Hotpot applicability arrays.

No production statistics/frontier imports, label access, raw archive traversal,
model execution, or per-question output. Bootstrap means use frequency weights
instead of the production gather-and-mean implementation. Inputs are explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

N = 384
EDGES = (1, 5, 7, 9)
SEEDS = (26111301, 26111302)
GRAPH_NAMES = ("A", "B", "A_MINUS_E", "B_MINUS_E")
PRIMARY = ("B_MINUS_NO_EDGE_EM", "B_MINUS_INDEPENDENT12_EM",
           "MEAN_SIGNED_DELTA_EM", "MEAN_CROSSSEED_DELTA_PRODUCT_EM")
SECONDARY = ("B_MINUS_SINGLE", "GAMMA_A_SEED1", "GAMMA_A_SEED2",
             "GAMMA_B_SEED1", "GAMMA_B_SEED2", "DELTA_SEED1", "DELTA_SEED2")
EXPECTED_EVALUATION = "0f63c65f6d60d8ca924ff1c9e814425b4e906a1658f1ce81abf24530b9d32270"
EXPECTED_RECEIPT = "653a7df8c6eb0df7d6c7c23c1253c53219437578b6e8ea55e435e9c8254ca506"
EXPECTED_REVIEW = "f3482981cb9bbd11c08a746ae35620e7ccb5868ea31cfed46900ea1332887e36"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError("duplicate JSON key")
            d[k] = v
        return d
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def form_columns(metric):
    """Return question-level contrasts; positive gamma means an edge helps."""
    g = np.asarray(metric["graphs"], dtype=np.float64)
    a_credit = g[:, :, 0] - g[:, :, 2]
    b_credit = g[:, :, 1] - g[:, :, 3]
    # Expanding this also yields B-Bminus-A+Aminus at each seed.
    drift = b_credit - a_credit
    return np.column_stack([
        g[:, 0, 1] - np.asarray(metric["no_edge"]),
        g[:, 0, 1] - np.asarray(metric["independent12"]),
        (drift[:, 0] + drift[:, 1]) / 2,
        drift[:, 0] * drift[:, 1],
        g[:, 0, 1] - np.asarray(metric["single"]),
        a_credit[:, 0], a_credit[:, 1], b_credit[:, 0], b_credit[:, 1],
        drift[:, 0], drift[:, 1],
    ])


def resampling_weights(edge, repetitions=10000, seed=261113103):
    """Pin exact protocol draws; independently aggregate draw multiplicities."""
    positions = [np.nonzero(edge == e)[0] for e in EDGES]
    draws = np.random.Generator(np.random.PCG64(seed)).integers(
        0, 96, size=(repetitions, 4, 96))
    actual_rows = np.empty((repetitions, N), dtype=np.int64)
    for j in range(4):
        actual_rows[:, j * 96:(j + 1) * 96] = positions[j][draws[:, j]]
    weights = np.zeros((repetitions, N), dtype=np.float64)
    np.add.at(weights, (np.arange(repetitions)[:, None], actual_rows), 1)
    return weights, digest(np.asarray(actual_rows, dtype="<i8").tobytes(order="C"))


def summarize(column, bootstrap_means, quantiles):
    low, high = np.quantile(bootstrap_means, quantiles, method="linear")
    positive = int(np.sum(column > 0))
    negative = int(np.sum(column < 0))
    pooled_var = float(np.var(column, ddof=1))
    boot_var = float(np.var(bootstrap_means, ddof=1))
    varying = bool(np.max(bootstrap_means) > np.min(bootstrap_means) and high > low)
    return {
        "estimate": float(np.mean(column)), "interval": [float(low), float(high)],
        "nonzero_questions": positive + negative, "positive_questions": positive,
        "negative_questions": negative, "sample_variance": pooled_var,
        "bootstrap_estimator_variance": boot_var, "bootstrap_nondegenerate": varying,
        "support_guard_pass": bool(positive + negative >= 20 and pooled_var > 0 and varying),
        "interval_width": float(high - low),
    }


def classify(summaries, overflow, frontier):
    quality = [summaries[x] for x in PRIMARY[:2]]
    q_positive = [v["support_guard_pass"] and v["interval"][0] > 0 for v in quality]
    q_negative = [v["support_guard_pass"] and v["interval"][1] < 0 for v in quality]
    l1 = ("SUPPORTED_BOTH" if all(q_positive) else
          "B_DISFAVORED" if any(q_negative) else "INCONCLUSIVE")
    drift = summaries[PRIMARY[2]]
    theta = summaries[PRIMARY[3]]
    direction = "INCONCLUSIVE"
    if drift["support_guard_pass"]:
        if drift["interval"][0] > 0:
            direction = "REDUNDANCY_DIRECTION_SUPPORTED"
        if drift["interval"][1] < 0:
            direction = "REDUNDANCY_DIRECTION_REVERSED"
    theta_positive = bool(theta["support_guard_pass"] and theta["interval"][0] > 0)
    conditional = theta_positive or direction != "INCONCLUSIVE"
    if overflow:
        action = "NO_AUTOMATIC_UPGRADE_CAPACITY_FAILURES"
    elif any(q_negative) or frontier == "DOMINATED":
        action = "STOP_CURRENT_B_DIRECTION"
    elif all(q_positive) and conditional:
        action = ("CONSIDER_INDEPENDENT_MECHANISM_PROTOCOL" if frontier == "NOT_DOMINATED"
                  else "PENDING_COST_FRONTIER_REVIEW")
    elif all(q_positive):
        action = "STATIC_COMMUNICATION_SUPPORTED_REVALIDATION_UNRESOLVED"
    elif conditional:
        action = "CONDITIONAL_RESPONSE_ONLY_NO_SYSTEM_SUCCESS"
    else:
        action = "EVIDENCE_INSUFFICIENT_STOP_AUTOMATIC_UPGRADE"
    return {
        "L1": l1,
        "L2": "GRAPH_CONDITIONAL_SIGNAL_SUPPORTED" if conditional else "INCONCLUSIVE",
        "signed_direction": direction, "theta_nonzero_supported": theta_positive,
        "theta_negative_point_retained": theta["estimate"] < 0,
        "theta_small_effect_exclusion": "NOT_PERMITTED_BY_THIS_BOOTSTRAP_CONTRACT",
        "theta_width_divided_by_0_0025": theta["interval_width"] / .0025,
        "quality_scale_tags": [
            {"point_at_least_5pp": q["estimate"] >= .05,
             "interval_lower_at_least_5pp": q["support_guard_pass"] and q["interval"][0] >= .05,
             "approx_within_plus_minus_2pp": q["support_guard_pass"] and
             q["interval"][0] >= -.02 and q["interval"][1] <= .02} for q in quality],
        "action": action, "GPU_admission_granted": False, "method_success_claim": False,
    }


def compare_trees(actual, expected, checks, name):
    """Numeric tolerance is solely for independent summation, not gate changes."""
    if isinstance(expected, dict):
        check(checks, name + ".keys", set(actual) == set(expected))
        for k in expected:
            compare_trees(actual[k], expected[k], checks, name + "." + k)
    elif isinstance(expected, list):
        check(checks, name + ".length", len(actual) == len(expected))
        for j, v in enumerate(expected):
            compare_trees(actual[j], v, checks, name + f"[{j}]")
    elif type(expected) is float:
        difference = abs(float(actual) - expected)
        check(checks, name, bool(difference <= 3e-14), abs_difference=difference)
    else:
        check(checks, name, actual == expected)


def check(checks, name, ok, **details):
    checks.append({"name": name, "passed": bool(ok), **details})
    if not ok:
        raise AssertionError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--evaluation-receipt", type=Path, required=True)
    parser.add_argument("--score-review", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    checks, inputs, raw_inputs = [], {}, {}
    protocol_dir = args.source_root / "protocol/cycle2_hotpot_applicability_20261113"
    implementation = args.source_root / "experiments/cycle2_20261113/hotpot_postclosure_v1"
    source_files = {
        "scientific_design": protocol_dir / "SCIENTIFIC_DESIGN_FROZEN.json",
        "statistics_source": implementation / "frozen_dependencies/frozen_dependencies/applicability_statistics.py",
        "frontier_source": implementation / "frozen_dependencies/frozen_dependencies/point_frontier.py",
        "frontier_rule": protocol_dir / "cost_frontier_rule/PROSPECTIVE_RULE.md",
        "evaluation_adapter_source": implementation / "frozen_dependencies/evaluation_adapter.py",
        "postclosure_source": implementation / "postclosure.py",
    }
    files = {"evaluation": args.evaluation, "evaluation_receipt": args.evaluation_receipt,
             "score_review": args.score_review, **source_files}
    for name, path in files.items():
        raw_inputs[name] = path.read_bytes()
        inputs[name] = {"filename": path.name, "sha256": digest(raw_inputs[name]),
                        "size_bytes": len(raw_inputs[name])}
    for name, expected in (("evaluation", EXPECTED_EVALUATION),
                           ("evaluation_receipt", EXPECTED_RECEIPT), ("score_review", EXPECTED_REVIEW)):
        check(checks, "input_pin." + name, inputs[name]["sha256"] == expected)
    e = decode(raw_inputs["evaluation"])
    er = decode(raw_inputs["evaluation_receipt"])
    sr = decode(raw_inputs["score_review"])
    p = decode(raw_inputs["scientific_design"])
    pin_pairs = [
        ("statistics_source", e["bindings"]["statistics_sha256"]),
        ("statistics_source", p["analysis_pins"]["statistics_sha256"]),
        ("frontier_source", e["bindings"]["frontier_sha256"]),
        ("frontier_source", p["analysis_pins"]["point_frontier_sha256"]),
        ("frontier_rule", p["analysis_pins"]["point_frontier_rule_sha256"]),
        ("scientific_design", e["bindings"]["scientific_protocol_sha256"]),
        ("evaluation_adapter_source", e["bindings"]["evaluation_adapter_sha256"]),
        ("evaluation_adapter_source", er["evaluation_adapter_sha256"]),
        ("postclosure_source", er["postclosure_source_sha256"]),
        ("evaluation", er["evaluation_sha256"]),
        ("score_review", er["score_review_sha256"]),
        ("score_review", e["archive_review_sha256"]),
    ]
    for j, (name, expected) in enumerate(pin_pairs):
        check(checks, f"source_or_receipt_binding.{j}.{name}", inputs[name]["sha256"] == expected)
    check(checks, "review_binding_map", sr["bindings"] == e["bindings"])
    check(checks, "scoring_authorized", sr["decision"] == "APPROVED_FOR_SCORING")
    check(checks, "no_new_calls_or_admission", er["new_model_calls"] == 0 and
          er["scientific_or_GPU_admission_granted"] is False and
          er["publication_authorized"] is False and e["scientific_or_GPU_admission_granted"] is False)
    for key in ("all_attempts_closed", "all_native_cost_fields_complete", "physical_ledger_closed", "resident_ledger_closed"):
        check(checks, "review." + key, sr[key] is True)
    check(checks, "review_counts", sr["unique_questions"] == N and sr["terminal_conditions"] == 4224
          and sr["runtime_failure_count"] == 0 and sr["input_overflow_conditions"] == 0)
    check(checks, "complete_scoring", e["status"] == "COMPLETE_BOUND_SCORING")
    check(checks, "named_axes", e["array_axes"] == {
        "rows": "exact selection manifest order", "graphs": list(GRAPH_NAMES), "seeds": list(SEEDS)})
    check(checks, "protocol_axes", p["statistics"]["graph_array_order"] == list(GRAPH_NAMES)
          and p["statistics"]["primary_order"] == list(PRIMARY))
    check(checks, "protocol_rng", p["statistics"]["bootstrap_repetitions"] == 10000
          and p["seeds"]["bootstrap_pcg64"] == 261113103
          and p["statistics"]["bootstrap_draw_shape"] == [10000, 4, 96]
          and p["statistics"]["primary_percentile_quantiles"] == [.00625, .99375]
          and p["statistics"]["secondary_percentile_quantiles"] == [.025, .975])
    edge = np.asarray(e["edge_indices"])
    check(checks, "balanced_edge_strata", edge.shape == (N,) and
          np.issubdtype(edge.dtype, np.integer) and Counter(edge.tolist()) == {x: 96 for x in EDGES})
    expected_keys = {"S1/" + x for x in (*GRAPH_NAMES, "NO_EDGE", "INDEPENDENT12", "SINGLE_M")}
    expected_keys |= {"S2/" + x for x in GRAPH_NAMES}
    seen = set()
    for row in e["condition_statuses"]:
        pair = row["selection_index"], row["condition_key"]
        if pair in seen or not 0 <= pair[0] < N or pair[1] not in expected_keys:
            raise AssertionError("duplicate_or_invalid_condition_status")
        if row["status"] != "SCORED_STRICT_AGGREGATE":
            raise AssertionError("noncomplete_condition")
        seen.add(pair)
    check(checks, "384_by_11_terminal_coverage", len(seen) == 4224)
    columns, condition_means = {}, {}
    for metric in ("em", "f1"):
        m = e[metric]
        check(checks, metric + ".metric_keys", set(m) == {"graphs", "no_edge", "independent12", "single"})
        for key, shape in (("graphs", (N, 2, 4)), ("no_edge", (N,)),
                           ("independent12", (N,)), ("single", (N,))):
            a = np.asarray(m[key], dtype=float)
            check(checks, metric + "." + key + ".range_shape", a.shape == shape and
                  np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all())
            if metric == "em":
                check(checks, metric + "." + key + ".binary", np.isin(a, [0, 1]).all())
        columns[metric] = form_columns(m)
        g = np.asarray(m["graphs"], dtype=float)
        condition_means[metric] = {
            f"S{s + 1}/{name}": float(g[:, s, j].mean())
            for s in range(2) for j, name in enumerate(GRAPH_NAMES)}
        condition_means[metric].update({"S1/" + name: float(np.mean(m[key]))
            for name, key in (("NO_EDGE", "no_edge"), ("INDEPENDENT12", "independent12"), ("SINGLE_M", "single"))})
    weights, indices_hash = resampling_weights(edge)
    check(checks, "resampling_draw_sha256", indices_hash == e["statistics"]["bootstrap"]["draw_index_sha256"])
    check(checks, "all_draws_weight384", (weights.sum(axis=1) == N).all())
    for x in EDGES:
        check(checks, "each_draw_edge96." + str(x), (weights[:, edge == x].sum(axis=1) == 96).all())
    bootstrap_means = {k: weights @ a / N for k, a in columns.items()}
    primary = {name: summarize(columns["em"][:, j], bootstrap_means["em"][:, j], (.00625, .99375))
               for j, name in enumerate(PRIMARY)}
    secondary = {
        "EM": {name: summarize(columns["em"][:, 4 + j], bootstrap_means["em"][:, 4 + j], (.025, .975))
               for j, name in enumerate(SECONDARY)},
        "F1_main_forms": {name.replace("_EM", "_F1"): summarize(columns["f1"][:, j], bootstrap_means["f1"][:, j], (.025, .975))
                          for j, name in enumerate(PRIMARY)},
        "F1_other": {name: summarize(columns["f1"][:, 4 + j], bootstrap_means["f1"][:, 4 + j], (.025, .975))
                     for j, name in enumerate(SECONDARY)},
    }
    per_edge = {str(x): dict(zip(PRIMARY, columns["em"][edge == x, :4].mean(axis=0).tolist())) for x in EDGES}
    compare_trees(primary, e["statistics"]["primary_EM"], checks, "primary")
    compare_trees(secondary, e["statistics"]["secondary_descriptive_only"], checks, "secondary")
    compare_trees(per_edge, e["statistics"]["per_edge_primary_means_descriptive"], checks, "per_edge_descriptive")
    counts = {name: int(round(condition_means["em"]["S1/" + name] * N))
              for name in ("B", "NO_EDGE", "INDEPENDENT12", "SINGLE_M")}
    cost_fields = ("calls", "input_tokens", "output_tokens", "CUDA_event_seconds")
    model_names = ("qwen", "mistral")
    vectors = {}
    for name, costs in e["full_logical_costs_by_model"].items():
        check(checks, name + ".cost_models", set(costs) == set(model_names))
        for model in model_names:
            d = costs[model]
            check(checks, name + "." + model + ".cost_fields", set(d) == set(cost_fields))
            check(checks, name + "." + model + ".no_unknown_cost", all(type(d[f]) in (float, int)
                  and np.isfinite(d[f]) and d[f] >= 0 for f in cost_fields))
            check(checks, name + "." + model + ".integer_counts_tokens",
                  all(type(d[f]) is int for f in cost_fields[:3]))
            expected = (0 if model == "qwen" else N) if name == "SINGLE_M" else (6 * N if model == "qwen" else 7 * N)
            check(checks, name + "." + model + ".logical_calls", d["calls"] == expected)
            check(checks, name + "." + model + ".zero_and_positive_consistency",
                  all(d[f] == 0 for f in cost_fields[1:]) if expected == 0 else d["input_tokens"] >= expected)
        vectors[name] = [costs[model][f] for model in model_names for f in cost_fields]
    compare_trees(counts, e["frontier"]["success_counts"], checks, "frontier.counts")
    compare_trees(vectors, e["frontier"]["cost_vectors"], checks, "frontier.costs")
    compare_trees([m + "/" + f for m in model_names for f in cost_fields],
                  e["frontier"]["cost_vector_order"], checks, "frontier.axes")
    comparisons, witnesses = {}, []
    for name in ("NO_EDGE", "INDEPENDENT12", "SINGLE_M"):
        differences = [a - b for a, b in zip(vectors[name], vectors["B"])]
        success_difference = counts[name] - counts["B"]
        weak = success_difference >= 0 and all(v <= 0 for v in differences)
        strict = success_difference > 0 or any(v < 0 for v in differences)
        comparisons[name] = {"success_count_difference_from_B": success_difference,
            "cost_vector_difference_from_B": differences,
            "weak_quality_and_cost": weak, "strictly_better_component": strict,
            "observed_point_dominates_B": weak and strict}
        if weak and strict:
            witnesses.append(name)
    compare_trees(comparisons, e["frontier"]["comparisons"], checks, "frontier.comparisons")
    compare_trees(witnesses, e["frontier"]["witnesses"], checks, "frontier.witnesses")
    frontier_status = "DOMINATED" if witnesses else "NOT_DOMINATED"
    check(checks, "frontier.status", frontier_status == e["frontier"]["status"])
    for key, value in (("sampling_uncertainty_assessed", False), ("population_dominance_claim", False),
                       ("GPU_admission_granted", False), ("does_not_rescue_primary_gates", True),
                       ("resident_cost_is_separate_physical_research_ledger", True)):
        check(checks, "frontier.boundary." + key, e["frontier"][key] is value)
    decision = classify(primary, e["audit_inputs"]["input_overflow_conditions"], frontier_status)
    compare_trees(decision, e["statistics"]["decision"], checks, "decision")
    check(checks, "audit_frontier_same", e["audit_inputs"]["cost_frontier_review"] == frontier_status)
    check(checks, "bootstrap_metadata", e["statistics"]["bootstrap"]["seed"] == 261113103 and
          e["statistics"]["bootstrap"]["repetitions"] == 10000 and
          e["statistics"]["bootstrap"]["primary_quantiles"] == [.00625, .99375] and
          e["statistics"]["bootstrap"]["quantile_method"] == "linear" and
          e["statistics"]["bootstrap"]["family_size"] == 4 and
          e["statistics"]["bootstrap"]["family_alpha"] == .05)
    unchanged = True
    for name, path in files.items():
        unchanged = unchanged and digest(path.read_bytes()) == inputs[name]["sha256"]
    check(checks, "all_inputs_unchanged_at_end", unchanged)
    n_floats = [c["abs_difference"] for c in checks if "abs_difference" in c]
    results = {
        "status": "PASS_INDEPENDENT_STATISTICAL_REPLAY_NOT_NEW_EXPERIMENT",
        "inputs": inputs, "source_sha256": digest(Path(__file__).read_bytes()),
        "checks_count": len(checks), "checks": checks,
        "max_absolute_numeric_replay_difference": max(n_floats, default=0.),
        "primary_EM": primary, "secondary_descriptive_only": secondary,
        "condition_means": condition_means, "seed1_EM_success_counts": counts,
        "format_failure_conditions_reported_not_independently_rescored": e["format_failure_conditions"],
        "bootstrap_draw_index_sha256": indices_hash,
        "bootstrap_implementation": "Same fixed PCG64 draws; independent frequency-weight matrix multiplication",
        "frontier": {"status": frontier_status, "cost_vectors": vectors,
                     "cost_vector_order": [m + "/" + f for m in model_names for f in cost_fields],
                     "comparisons": comparisons, "witnesses": witnesses},
        "decision": decision,
        "scope_limits": {
            "labels_or_raw_questions_or_raw_generations_read": False,
            "raw_scoring_independently_repeated": False,
            "raw_node_costs_independently_recomputed_in_this_review": False,
            "full_cost_aggregate_geometry_and_Pareto_recomputed": True,
            "source_and_array_pins_verified": True,
            "new_GPU_or_model_calls": 0, "E_or_Git_modified": False,
            "new_admission_or_publication_authorized": False,
            "current_B_stop_is_project_stop": False,
        },
        "runtime": {"elapsed_seconds": time.perf_counter() - start,
                    "python": platform.python_version(), "numpy": np.__version__},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "RESULTS.json"
    with target.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": results["status"], "checks_count": len(checks),
        "max_absolute_numeric_replay_difference": results["max_absolute_numeric_replay_difference"],
        "primary_EM": primary, "condition_means": condition_means, "decision": decision,
        "frontier": results["frontier"], "results_sha256": digest(target.read_bytes())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
