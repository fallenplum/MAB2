"""Finite strict graph-VOI simulator used by the P3 CPU contract.

The simulator has real parent fingerprints. Querying a source can validate a
cached descendant cone when its output is unchanged, or dirty the cone when it
changes. Terminal decisions use a finite weighted label aggregator, so the
certificate can be exhaustively checked against every unknown label assignment.
"""
from __future__ import annotations

import argparse
import itertools
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


FAMILY_SPECS = {
    "chain": {"sources": 2, "terminals": 5, "parents": "chain"},
    "fanout": {"sources": 3, "terminals": 8, "parents": "fanout"},
    "diamond": {"sources": 3, "terminals": 8, "parents": "diamond"},
    "mixed": {"sources": 4, "terminals": 10, "parents": "mixed"},
}
METHODS = (
    "stale",
    "exact_dependency_cache",
    "descendant_mask_recompute",
    "masked_pairwise",
    "generic_adaptive",
    "strict_certificate_first",
    "topology_voi_exact_dp",
    "full_recompute",
    "offline_full_information",
)

METHODS_H1 = METHODS + ("topology_voi_h1",)


def build_dependencies(kind: str, n_sources: int, n_terminals: int) -> list[tuple[int, ...]]:
    if kind == "chain":
        return [tuple([0]) for _ in range(n_terminals)]
    if kind == "fanout":
        return [tuple([0] if i < n_terminals // 2 else [1, 2]) for i in range(n_terminals)]
    if kind == "diamond":
        return [tuple([0, 1]) if i % 2 == 0 else tuple([1, 2]) for i in range(n_terminals)]
    return [tuple(sorted({i % n_sources, (i * 3 + 1) % n_sources})) for i in range(n_terminals)]


def terminal_label(bits: tuple[int, ...], deps: tuple[int, ...], index: int, salt: int) -> int:
    value = sum((i + 1) * bits[i] for i in deps) + index + salt
    return int(value % 3)


def make_case(rng: np.random.Generator, family: str, drift: float, regime: str, edit_count: int) -> dict:
    spec = FAMILY_SPECS[family]
    n_sources = spec["sources"]
    n_terminals = spec["terminals"]
    deps = build_dependencies(spec["parents"], n_sources, n_terminals)
    old = tuple(int(v) for v in rng.integers(0, 2, n_sources))
    changed = tuple(sorted(int(v) for v in rng.choice(n_sources, size=min(edit_count, n_sources), replace=False)))
    new = list(old)
    for source in changed:
        if rng.random() < drift:
            new[source] = 1 - new[source]
    new = tuple(new)
    weights = rng.uniform(0.4, 1.6, n_terminals)
    if regime == "heterogeneous":
        weights *= np.asarray([1.0 + 2.5 * (i % 3 == 0) for i in range(n_terminals)])
    costs = rng.uniform(2.0, 8.0, n_sources + n_terminals)
    if regime == "heterogeneous":
        costs *= np.asarray([1.0 + 3.0 * (i % 4 == 0) for i in range(len(costs))])
    salt = int(rng.integers(0, 7))
    old_labels = tuple(terminal_label(old, d, i, salt) for i, d in enumerate(deps))
    new_labels = tuple(terminal_label(new, d, i, salt) for i, d in enumerate(deps))
    return {
        "family": family,
        "deps": deps,
        "old": old,
        "new": new,
        "changed": changed,
        "weights": tuple(float(v) for v in weights),
        "costs": tuple(float(v) for v in costs),
        "old_labels": old_labels,
        "new_labels": new_labels,
        "salt": salt,
        "source_count": n_sources,
        "terminal_count": n_terminals,
    }


def aggregate(labels: Iterable[int], weights: tuple[float, ...]) -> int:
    scores = np.zeros(3, dtype=float)
    for label, weight in zip(labels, weights):
        scores[int(label)] += weight
    return int(np.argmax(scores))


def certificate(known: dict[int, int], weights: tuple[float, ...], n_terminals: int) -> tuple[bool, int]:
    scores = np.zeros(3, dtype=float)
    unknown_weight = 0.0
    for index, weight in enumerate(weights):
        if index in known:
            scores[known[index]] += weight
        else:
            unknown_weight += weight
    candidate = int(np.argmax(scores))
    other = max(float(scores[i]) for i in range(3) if i != candidate)
    return bool(scores[candidate] > other + unknown_weight), candidate


def valid_cached_terminals(case: dict, observed_sources: dict[int, int]) -> dict[int, int]:
    known = {}
    for i, deps in enumerate(case["deps"]):
        if all(source in observed_sources for source in deps):
            # A cache hit is legal only when every effective parent output is
            # unchanged. The current output is then the old transcript value.
            if all(observed_sources[source] == case["old"][source] for source in deps):
                known[i] = case["old_labels"][i]
    return known


def dirty_terminals(case: dict, observed_sources: dict[int, int]) -> list[int]:
    known = valid_cached_terminals(case, observed_sources)
    return [i for i in range(case["terminal_count"]) if i not in known]


def resolve_terminals(case: dict, observed_sources: dict[int, int], known: dict[int, int], order: list[int]) -> tuple[int, float, int, bool]:
    queried = 0
    cost = 0.0
    for terminal in order:
        if terminal in known:
            continue
        known[terminal] = case["new_labels"][terminal]
        queried += 1
        cost += case["costs"][case["source_count"] + terminal]
        safe, choice = certificate(known, case["weights"], case["terminal_count"])
        if safe:
            return choice, cost, queried, True
    safe, choice = certificate(known, case["weights"], case["terminal_count"])
    return (aggregate(case["new_labels"], case["weights"]) if not safe else choice), cost, queried, safe


def source_order(case: dict, method: str) -> list[int]:
    sources = list(range(case["source_count"]))
    costs = case["costs"]
    if method == "generic_adaptive":
        return sorted(sources, key=lambda i: (1.0 / max(costs[i], 1e-6), i), reverse=True)
    if method == "masked_pairwise":
        impact = {i: sum(case["weights"][t] for t, d in enumerate(case["deps"]) if i in d) for i in sources}
        return sorted(sources, key=lambda i: (impact[i] / max(costs[i], 1e-6), i), reverse=True)
    if method == "strict_certificate_first":
        impact = {i: sum(case["weights"][t] for t, d in enumerate(case["deps"]) if i in d) for i in sources}
        return sorted(sources, key=lambda i: (impact[i] / max(costs[i], 1e-6), i), reverse=True)
    return sources


def run_static(case: dict, method: str) -> dict:
    target = aggregate(case["new_labels"], case["weights"])
    old_action = aggregate(case["old_labels"], case["weights"])
    if method == "stale":
        return {"action": old_action, "target": target, "cost": 0.0, "queries": 0, "certificate": False}
    if method == "offline_full_information":
        return {"action": target, "target": target, "cost": 0.0, "queries": 0, "certificate": True}
    if method == "full_recompute":
        return {"action": target, "target": target, "cost": float(sum(case["costs"])), "queries": len(case["costs"]), "certificate": True}

    observed = {}
    cost = 0.0
    queries = 0
    order = source_order(case, method)
    for source in order:
        observed[source] = case["new"][source]
        cost += case["costs"][source]
        queries += 1
        known = valid_cached_terminals(case, observed)
        safe, choice = certificate(known, case["weights"], case["terminal_count"])
        if safe and method == "strict_certificate_first":
            return {"action": choice, "target": target, "cost": cost, "queries": queries, "certificate": True}

    known = valid_cached_terminals(case, observed)
    dirty = dirty_terminals(case, observed)
    if method == "exact_dependency_cache":
        terminal_order = dirty
    elif method == "descendant_mask_recompute":
        terminal_order = list(range(case["terminal_count"]))
    elif method in {"masked_pairwise", "generic_adaptive"}:
        terminal_order = sorted(dirty, key=lambda i: (case["weights"][i] / max(case["costs"][case["source_count"] + i], 1e-6), i), reverse=True)
    else:
        terminal_order = dirty
    action, terminal_cost, terminal_queries, cert = resolve_terminals(case, observed, known, terminal_order)
    if method == "descendant_mask_recompute":
        action = target
        terminal_cost = sum(case["costs"][case["source_count"] + i] for i in dirty)
        terminal_queries = len(dirty)
    return {"action": action, "target": target, "cost": cost + terminal_cost, "queries": queries + terminal_queries, "certificate": cert}


def run_dp(case: dict) -> dict:
    """Exact finite-prior DP over source and terminal queries.

    The state contains observed source bits and observed terminal labels. A
    query changes the posterior over assignments and may change which terminal
    cache entries are valid. This is the strict nonmyopic reference policy.
    """
    target = aggregate(case["new_labels"], case["weights"])
    assignments = list(itertools.product((0, 1), repeat=case["source_count"]))

    def possible(source_tuple, terminal_tuple):
        observed = dict(source_tuple)
        explicit = dict(terminal_tuple)
        return [
            assignment for assignment in assignments
            if all(assignment[i] == value for i, value in observed.items())
            and all(terminal_label(assignment, case["deps"][i], i, case["salt"]) == value for i, value in explicit.items())
        ]

    def known_labels(source_tuple, terminal_tuple):
        known = valid_cached_terminals(case, dict(source_tuple))
        known.update(dict(terminal_tuple))
        return known

    @lru_cache(None)
    def future(source_tuple, terminal_tuple):
        known = known_labels(source_tuple, terminal_tuple)
        safe, _ = certificate(known, case["weights"], case["terminal_count"])
        if safe:
            return 0.0
        posterior = possible(source_tuple, terminal_tuple)
        if not posterior:
            return float("inf")
        candidates = []
        observed_sources = dict(source_tuple)
        for source in range(case["source_count"]):
            if source in observed_sources:
                continue
            expected = case["costs"][source]
            branch_cost = 0.0
            for bit in (0, 1):
                branch = [assignment for assignment in posterior if assignment[source] == bit]
                if branch:
                    next_source = tuple(sorted((*source_tuple, (source, bit))))
                    branch_cost += len(branch) / len(posterior) * future(next_source, terminal_tuple)
            candidates.append(expected + branch_cost)
        for terminal in range(case["terminal_count"]):
            if terminal in known or not all(source in observed_sources for source in case["deps"][terminal]):
                continue
            expected = case["costs"][case["source_count"] + terminal]
            branch_cost = 0.0
            labels = {}
            for assignment in posterior:
                label = terminal_label(assignment, case["deps"][terminal], terminal, case["salt"])
                labels[label] = labels.get(label, 0) + 1
            for label, count in labels.items():
                next_terminal = tuple(sorted((*terminal_tuple, (terminal, label))))
                branch_cost += count / len(posterior) * future(source_tuple, next_terminal)
            candidates.append(expected + branch_cost)
        return min(candidates) if candidates else float("inf")

    source_observed = {}
    terminal_observed = {}
    cost = 0.0
    queries = 0
    while True:
        known = known_labels(tuple(sorted(source_observed.items())), tuple(sorted(terminal_observed.items())))
        safe, choice = certificate(known, case["weights"], case["terminal_count"])
        if safe:
            return {"action": choice, "target": target, "cost": cost, "queries": queries, "certificate": True}
        source_tuple = tuple(sorted(source_observed.items()))
        terminal_tuple = tuple(sorted(terminal_observed.items()))
        candidates = []
        posterior = possible(source_tuple, terminal_tuple)
        for source in range(case["source_count"]):
            if source in source_observed:
                continue
            value = case["costs"][source]
            for bit in (0, 1):
                branch = [assignment for assignment in posterior if assignment[source] == bit]
                if branch:
                    value += len(branch) / len(posterior) * future(tuple(sorted((*source_tuple, (source, bit)))), terminal_tuple)
            candidates.append((value, "source", source))
        for terminal in range(case["terminal_count"]):
            if terminal in known or not all(source in source_observed for source in case["deps"][terminal]):
                continue
            value = case["costs"][case["source_count"] + terminal]
            labels = {}
            for assignment in posterior:
                label = terminal_label(assignment, case["deps"][terminal], terminal, case["salt"])
                labels[label] = labels.get(label, 0) + 1
            for label, count in labels.items():
                value += count / len(posterior) * future(source_tuple, tuple(sorted((*terminal_tuple, (terminal, label)))))
            candidates.append((value, "terminal", terminal))
        if not candidates:
            return {"action": target, "target": target, "cost": cost, "queries": queries, "certificate": False}
        _, kind, index = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        if kind == "source":
            source_observed[index] = case["new"][index]
            cost += case["costs"][index]
        else:
            terminal_observed[index] = case["new_labels"][index]
            cost += case["costs"][case["source_count"] + index]
        queries += 1


def run_h1(case: dict) -> dict:
    """One-step posterior rollout using the same strict certificate."""
    target = aggregate(case["new_labels"], case["weights"])
    assignments = list(itertools.product((0, 1), repeat=case["source_count"]))

    def posterior(source_observed, terminal_observed):
        return [a for a in assignments if all(a[i] == v for i, v in source_observed.items()) and all(terminal_label(a, case["deps"][i], i, case["salt"]) == v for i, v in terminal_observed.items())]

    def finish_cost(assignment, source_observed, terminal_observed):
        known = valid_cached_terminals(case, source_observed)
        known.update(terminal_observed)
        for terminal in sorted((i for i in range(case["terminal_count"]) if i not in known), key=lambda i: (case["weights"][i] / max(case["costs"][case["source_count"] + i], 1e-6), i), reverse=True):
            known[terminal] = terminal_label(assignment, case["deps"][terminal], terminal, case["salt"])
            safe, _ = certificate(known, case["weights"], case["terminal_count"])
            if safe:
                break
        cached = valid_cached_terminals(case, source_observed)
        return sum(case["costs"][case["source_count"] + i] for i in known if i not in cached)

    def estimate(source_observed, terminal_observed):
        candidates = posterior(source_observed, terminal_observed)
        if not candidates:
            return float("inf")
        unknown_source_cost = sum(case["costs"][i] for i in range(case["source_count"]) if i not in source_observed)
        return unknown_source_cost + float(np.mean([finish_cost(a, dict(enumerate(a)), terminal_observed) for a in candidates]))

    source_observed = {}
    terminal_observed = {}
    cost = 0.0
    queries = 0
    while True:
        known = valid_cached_terminals(case, source_observed)
        known.update(terminal_observed)
        safe, choice = certificate(known, case["weights"], case["terminal_count"])
        if safe:
            return {"action": choice, "target": target, "cost": cost, "queries": queries, "certificate": True}
        posterior_now = posterior(source_observed, terminal_observed)
        candidates = []
        for source in range(case["source_count"]):
            if source in source_observed:
                continue
            value = case["costs"][source]
            for bit in (0, 1):
                branch = [a for a in posterior_now if a[source] == bit]
                if branch:
                    next_source = dict(source_observed); next_source[source] = bit
                    value += len(branch) / len(posterior_now) * estimate(next_source, terminal_observed)
            candidates.append((value, "source", source))
        for terminal in range(case["terminal_count"]):
            if terminal in known or not all(source in source_observed for source in case["deps"][terminal]):
                continue
            value = case["costs"][case["source_count"] + terminal]
            labels = {}
            for a in posterior_now:
                label = terminal_label(a, case["deps"][terminal], terminal, case["salt"])
                labels[label] = labels.get(label, 0) + 1
            for label, count in labels.items():
                next_terminal = dict(terminal_observed); next_terminal[terminal] = label
                value += count / len(posterior_now) * estimate(source_observed, next_terminal)
            candidates.append((value, "terminal", terminal))
        _, kind, index = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        if kind == "source":
            source_observed[index] = case["new"][index]
            cost += case["costs"][index]
        else:
            terminal_observed[index] = case["new_labels"][index]
            cost += case["costs"][case["source_count"] + index]
        queries += 1

def run(seed: int, cases_per_family: int = 500) -> dict:
    rng = np.random.default_rng(seed)
    records = []
    for family in FAMILY_SPECS:
        for drift in (0.0, 0.1, 0.4, 1.0):
            for regime in ("uniform", "heterogeneous"):
                for edit_count in (1, 2):
                    for _ in range(cases_per_family):
                        case = make_case(rng, family, drift, regime, edit_count)
                        for method in METHODS_H1:
                            if method == "topology_voi_exact_dp":
                                result = run_dp(case)
                            elif method == "topology_voi_h1":
                                result = run_h1(case)
                            else:
                                result = run_static(case, method)
                            result.update({"family": family, "drift": drift, "regime": regime, "edit_count": edit_count, "method": method})
                            records.append(result)
    return {"schema": "mab2_p3_strict_graph_voi_cpu_v1", "seed": seed, "records": records}


def summarize(payload: dict) -> dict:
    records = payload["records"]
    out = {}
    for method in METHODS_H1:
        rows = [r for r in records if r["method"] == method]
        out[method] = {
            "n": len(rows),
            "action_exact": float(np.mean([r["action"] == r["target"] for r in rows])),
            "mean_cost": float(np.mean([r["cost"] for r in rows])),
            "mean_queries": float(np.mean([r["queries"] for r in rows])),
            "certificate_rate": float(np.mean([r["certificate"] for r in rows])),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.seed, args.cases)
    payload["summary"] = summarize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

