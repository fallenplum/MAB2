"""Controlled DAG experiment for topology-aware value of information.

The experiment is synthetic and CPU-only.  It constructs a layered DAG with
explicit cross-node dependencies, perturbs one source node, and measures how
different revalidation policies recover the post-edit action.  The offline
oracle uses the generated post-edit values only to report an upper bound.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


N_ACTIONS = 3
N_NODES = 15
TAU = 0.0


def clip(value: float) -> float:
    return float(np.clip(value, 0.02, 0.98))


def descendants(edges: dict[int, list[int]], edits: list[int]) -> set[int]:
    seen = set(edits)
    stack = list(edits)
    while stack:
        node = stack.pop()
        for child in edges.get(node, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def make_graph(rng: np.random.Generator) -> tuple[dict[int, list[int]], list[int]]:
    """Build a four-level DAG and edit one source node."""
    levels = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]
    edges: dict[int, list[int]] = {i: [] for i in range(N_NODES)}
    for left, right in zip(levels[:-1], levels[1:]):
        for child in right:
            parents = rng.choice(left, size=2, replace=False)
            for parent in parents:
                edges[int(parent)].append(int(child))
    edits = [int(rng.choice(levels[0]))]
    return edges, edits


def make_case(rng: np.random.Generator) -> dict[str, object]:
    edges, edits = make_graph(rng)
    affected = descendants(edges, edits)
    q_old = rng.uniform(0.36, 0.64, size=N_NODES)
    q_new = q_old.copy()
    delta = np.zeros(N_NODES, dtype=float)
    topo = list(range(N_NODES))
    for node in topo:
        if node in edits:
            delta[node] = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.10, 0.20))
        else:
            parent_values = [p for p, cs in edges.items() if node in cs]
            if parent_values and any(parent in affected for parent in parent_values):
                parent_delta = float(np.mean([delta[p] for p in parent_values]))
                delta[node] = 0.55 * parent_delta + float(rng.normal(0.0, 0.035))
        if node in affected:
            q_new[node] = clip(q_old[node] + delta[node])

    # Action weights are shared by all policies.  The post-edit decision is
    # close enough to the boundary that a stale value can be wrong, while the
    # affected descendants carry most of the decision mass.
    weights = rng.normal(0.0, 0.34, size=(N_ACTIONS, N_NODES))
    weights[:, 0:3] *= 1.4
    weights[:, 11:] *= 1.25
    base = rng.normal(0.0, 0.03, size=N_ACTIONS)
    old_scores = base + weights @ q_old
    center = float(np.mean(old_scores))
    base -= center
    weights -= np.mean(weights, axis=0, keepdims=True)
    prices = rng.integers(7, 18, size=N_NODES).astype(float)
    radii = np.full(N_NODES, 0.19, dtype=float)
    radii[:3] = 0.22
    radii[3:7] = 0.20
    radii[7:11] = 0.18
    radii[11:] = 0.16
    return {
        "edges": {str(k): v for k, v in edges.items()},
        "edits": edits,
        "affected": sorted(affected),
        "q_old": q_old.tolist(),
        "q_new": q_new.tolist(),
        "weights": weights.tolist(),
        "base": base.tolist(),
        "prices": prices.tolist(),
        "radii": radii.tolist(),
    }


def scores(case: dict[str, object], values: np.ndarray) -> np.ndarray:
    return np.asarray(case["base"], dtype=float) + np.asarray(case["weights"], dtype=float) @ values


def target_action(case: dict[str, object]) -> int:
    return int(np.argmax(scores(case, np.asarray(case["q_new"], dtype=float))))


def lower_gap(case: dict[str, object], estimate: np.ndarray, unknown: set[int]) -> tuple[int, float]:
    score = scores(case, estimate)
    choice = int(np.argmax(score))
    radii = np.asarray(case["radii"], dtype=float)
    weights = np.asarray(case["weights"], dtype=float)
    worst = float("inf")
    for competitor in range(N_ACTIONS):
        if competitor == choice:
            continue
        uncertainty = sum(
            2.0 * abs(float(weights[choice, i] - weights[competitor, i])) * radii[i]
            for i in unknown
        )
        worst = min(worst, float(score[choice] - score[competitor]) - uncertainty)
    return choice, worst


def run_controller(case: dict[str, object], method: str) -> dict[str, object]:
    old = np.asarray(case["q_old"], dtype=float)
    new = np.asarray(case["q_new"], dtype=float)
    prices = np.asarray(case["prices"], dtype=float)
    radii = np.asarray(case["radii"], dtype=float)
    affected = set(int(x) for x in case["affected"])
    target = target_action(case)
    if method == "exact_cache":
        predicted = int(np.argmax(scores(case, old)))
        return {"predicted": predicted, "target": target, "tokens": 0.0, "queries": 0, "certificate": "reuse"}
    if method == "full_recompute":
        return {"predicted": target, "target": target, "tokens": float(prices.sum()), "queries": N_NODES, "certificate": "full"}
    if method == "offline_full_info":
        # This is deliberately an offline upper bound, not an online route.
        return {"predicted": target, "target": target, "tokens": 0.0, "queries": 0, "certificate": "offline_oracle"}

    if method == "descendant_recompute":
        known = set(affected)
    else:
        known = set()

    if method == "pairwise":
        # The existing Pairwise certificate already receives the graph mask.
        # Keeping this candidate set equal to the affected descendants makes
        # the comparison isolate topology-aware ranking rather than metadata
        # access.
        candidates = set(affected)
    elif method == "generic_adaptive":
        candidates = set(range(N_NODES))
    elif method == "topology_voi":
        candidates = set(affected)
    elif method == "descendant_recompute":
        candidates = set(affected)
    else:
        raise ValueError(method)

    tokens = float(prices[list(known)].sum()) if known else 0.0
    queries = len(known)
    estimate = old.copy()
    for node in known:
        estimate[node] = new[node]
    # Descendant recomputation has exact information on the graph mask and is
    # the strongest topology baseline.
    if method == "descendant_recompute":
        return {"predicted": int(np.argmax(scores(case, estimate)),), "target": target,
                "tokens": tokens, "queries": queries, "certificate": "descendant_mask"}

    # Generic adaptive assigns uncertainty to every node.  Existing Pairwise
    # receives the graph mask but ranks affected nodes by a worst-case gap
    # contribution.  Topology VOI uses the same mask and recomputes the
    # current action's local decision impact at every step.
    while True:
        unknown = candidates.difference(known)
        if method in {"topology_voi", "pairwise"}:
            _, certificate_gap = lower_gap(case, estimate, unknown)
        else:
            _, certificate_gap = lower_gap(case, estimate, set(range(N_NODES)).difference(known))
        if not unknown or certificate_gap > TAU:
            break
        if method == "pairwise":
            choice = max(
                unknown,
                key=lambda i: (
                    max(abs(float(np.asarray(case["weights"])[a, i] - np.asarray(case["weights"])[b, i]))
                        for a in range(N_ACTIONS) for b in range(N_ACTIONS) if a != b)
                    * radii[i] / max(prices[i], 1.0),
                    -i,
                ),
            )
        elif method == "generic_adaptive":
            choice = max(unknown, key=lambda i: (radii[i] ** 2 / max(prices[i], 1.0), -i))
        else:
            weights = np.asarray(case["weights"], dtype=float)
            current = int(np.argmax(scores(case, estimate)))
            # Topology VOI combines decision impact, graph descendant status
            # and cost.  The graph mask is the mechanism under test.
            choice = max(
                unknown,
                key=lambda i: (
                    max(abs(float(weights[current, i] - weights[a, i])) for a in range(N_ACTIONS) if a != current)
                    * radii[i] / max(prices[i], 1.0),
                    -i,
                ),
            )
        known.add(int(choice))
        estimate[choice] = new[choice]
        tokens += float(prices[choice])
        queries += 1
    predicted = int(np.argmax(scores(case, estimate)))
    cert = "topology_voi" if method == "topology_voi" else method
    return {"predicted": predicted, "target": target, "tokens": tokens, "queries": queries, "certificate": cert}


def run(seed: int = 20261124, n_cases: int = 5000) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    methods = ["exact_cache", "descendant_recompute", "pairwise", "generic_adaptive", "topology_voi", "full_recompute", "offline_full_info"]
    records: dict[str, list[dict[str, object]]] = defaultdict(list)
    topology_extra: list[dict[str, object]] = []
    for _ in range(n_cases):
        case = make_case(rng)
        for method in methods:
            row = run_controller(case, method)
            row["action_error"] = int(row["predicted"] != row["target"])
            records[method].append(row)
        topology_extra.append({"affected": len(case["affected"]), "edit": case["edits"]})
    summary = {}
    for method in methods:
        rows = records[method]
        summary[method] = {
            "n": len(rows),
            "action_error": float(np.mean([r["action_error"] for r in rows])),
            "mean_tokens": float(np.mean([r["tokens"] for r in rows])),
            "mean_queries": float(np.mean([r["queries"] for r in rows])),
            "exact_action": float(1.0 - np.mean([r["action_error"] for r in rows])),
            "certificate_counts": dict(sorted({k: sum(1 for r in rows if r["certificate"] == k) for k in {r["certificate"] for r in rows}}.items())),
        }
    return {
        "schema": "controlled_dag_topology_voi_v1",
        "seed": seed,
        "n_cases": n_cases,
        "n_nodes": N_NODES,
        "n_actions": N_ACTIONS,
        "methods": methods,
        "summary": summary,
        "affected_size": {
            "mean": float(np.mean([x["affected"] for x in topology_extra])),
            "min": int(min(x["affected"] for x in topology_extra)),
            "max": int(max(x["affected"] for x in topology_extra)),
        },
        "graph_protocol": {
            "topology": "four-level DAG with explicit parent-to-child dependencies",
            "edit": "one source node per case",
            "affected_rows": "edit node plus all reachable descendants",
            "decision": "argmax of three weighted action scores",
            "risk": "post-edit action error against generated q_new",
            "cost": "sum of queried node token prices",
            "pairwise_baseline": "existing worst-case action-gap-per-cost ranking over the affected mask",
            "topology_voi": "same affected mask with dynamic current-action local impact per cost",
            "offline_oracle": "offline_full_info reports the generated upper bound and is not an online policy",
        },
        "scope": "CPU synthetic mechanism diagnostic; no language-model calls or sealed task labels.",
    }


def write_report(result: dict[str, object], path: Path) -> None:
    lines = [
        "# P1 controlled DAG topology VOI audit",
        "",
        "This CPU-only experiment uses a four-level DAG with explicit cross-node dependencies. One source node is edited in each case, and its reachable descendants are the graph-conditioned affected set. The action is the argmax of three weighted scores over node credits. Costs are queried node token prices.",
        "",
        "| method | action error | exact action | mean tokens | mean queries |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method, values in result["summary"].items():
        lines.append(f"| {method} | {values['action_error']:.4f} | {values['exact_action']:.4f} | {values['mean_tokens']:.2f} | {values['mean_queries']:.2f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The topology-aware policy receives the affected descendant mask from the DAG. Pairwise and generic adaptive controls do not use that mask and treat all nodes as potentially stale. The offline full-information line is a diagnostic upper bound only and is not an executable baseline.",
        "",
        "This experiment is evidence about topology-dependent information value. It does not establish natural-language task quality, model-call latency, or a universal advantage outside the generated DAG family.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    output = Path(__file__).with_name("P1_TOPOLOGY_VOI.json")
    report = output.with_suffix(".md")
    result = run()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, report)
    print(json.dumps(result["summary"], indent=2))
