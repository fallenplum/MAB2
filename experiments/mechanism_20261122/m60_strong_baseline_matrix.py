"""Fair M60 baseline matrix on the frozen conditional-MCQA panels.

Every controller receives the same old certificate transcript, post-edit
source distribution, relay outputs, prices and sealed reference.  Only
post-edit query actions are charged.  The exact warm-cache controller never
queries after the edit.  The row-refresh oracle is an explicit restricted
oracle over changed-row queries, and a separate necessary-cost lower bound is
reported when the warm cache is insufficient.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from blind_scheduler import Audit  # noqa: E402
from conditional_real_score import (  # noqa: E402
    baseline_recheck,
    cost,
    graph_manifest_changed_rows,
    guarded,
    old_history,
    replay,
)
from conditional_schema import validate_conditional_full  # noqa: E402
from ctdr import truth  # noqa: E402

TAU = 0.01
METHODS = (
    "stale",
    "exact_warm_cache",
    "reweighted_old_history",
    "ctdr_transport_only",
    "ctdr_transport_selective_ranking",
    "full_ctdr",
    "max_uncertainty",
    "periodic",
    "random",
    "full_restart",
    "oracle_row_refresh",
)


def exact_warm_cache(new_source, base, labels, history):
    """Reweight the paid old transcript and forbid every new query."""
    audit = Audit(new_source, base, np.ones(4), np.ones((4, 4)), "joint", tau=TAU)
    for action, value in history:
        audit.update(action, value)
    return set(audit.decision() or []), []


def _audit_from_history(new_source, base, history):
    audit = Audit(new_source, base, np.ones(4), np.ones((4, 4)), "joint", tau=TAU)
    for action, value in history:
        audit.update(action, value)
    return audit


def _row_refresh_prediction(new_source, base, new_relay, history, rows):
    """Evaluate a changed-row subset without charging a query twice."""
    stale_rows = {int(i) for i, _ in rows}
    filtered_history = [(action, value) for action, value in history
                        if int(action[0]) not in stale_rows]
    audit = _audit_from_history(new_source, base, filtered_history)
    # Old terminal cells belonging to a refreshed row are stale.  The row
    # update replaces the row certificate and clears its conditional cells.
    for i in sorted(stale_rows):
        audit.update((i, None), new_relay[i])
    return set(audit.decision() or [])


def oracle_row_refresh(new_source, base, new_relay, labels, history,
                       changed_rows, tokens, target):
    """Return restricted row-refresh optimum and a valid necessary lower bound."""
    changed = tuple(sorted(set(int(i) for i in changed_rows)))
    row_prices = {i: int(tokens[7 + i]) for i in range(4)}
    warm_pred, _ = exact_warm_cache(new_source, base, labels, history)
    lower_bound = 0 if warm_pred == target else min(row_prices[i] for i in changed)
    best = None
    best_rows: tuple[int, ...] = ()
    for size in range(len(changed) + 1):
        for subset in itertools.combinations(changed, size):
            pred = _row_refresh_prediction(new_source, base, new_relay, history,
                                           [(i, None) for i in subset])
            if pred == target:
                spent = sum(row_prices[i] for i in subset)
                if best is None or spent < best:
                    best, best_rows = spent, tuple(subset)
        if best is not None:
            break
    if best is None:
        # The restricted oracle cannot certify the target without conditional
        # cell actions.  Full restart remains a finite feasible upper bound.
        best = int(sum(int(tokens[7 + i]) for i in range(4)))
        best_rows = tuple(changed)
    return best_rows, int(best), int(lower_bound)


def _record(pred, actions, target, gold, tokens):
    return {
        "pred": sorted(int(x) for x in pred),
        "target": sorted(int(x) for x in target),
        "actions": [[int(i), None if j is None else int(j)] for i, j in actions],
        "tokens": int(cost(actions, tokens)),
        "exact_set": int(set(pred) == set(target)),
        "action_error": int((gold in pred) != (gold in target)),
        "false_drop": int(bool(set(target) - set(pred))),
        "false_keep": int(bool(set(pred) - set(target))),
    }


def _summarize(rows):
    n = len(rows)
    return {
        "n": n,
        "exact_set": float(np.mean([r["exact_set"] for r in rows])),
        "action_error": float(np.mean([r["action_error"] for r in rows])),
        "false_drop": float(np.mean([r["false_drop"] for r in rows])),
        "false_keep": float(np.mean([r["false_keep"] for r in rows])),
        "queries": float(np.mean([len(r["actions"]) for r in rows])),
        "tokens": float(np.mean([r["tokens"] for r in rows])),
        "budget_tokens": float(np.mean([r.get("budget_tokens", 0.0) for r in rows])),
        "unused_budget_tokens": float(np.mean([r.get("unused_budget_tokens", 0.0) for r in rows])),
    }


def run(manifests: list[Path], output: Path):
    all_rows: list[dict[str, Any]] = []
    by_density: dict[str, list[dict[str, Any]]] = {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        density = manifest_path.stem.split("_rows")[-1].split("_")[0]
        density_rows: list[dict[str, Any]] = []
        for panel, folder_name in manifest["panels"].items():
            folder = Path(folder_name)
            if not folder.exists() and (Path("MAB2_recovered") / folder).exists():
                folder = Path("MAB2_recovered") / folder
            sealed = json.loads((folder / "SEALED.json").read_bytes())
            for worker in manifest.get("workers", [0, 1, 2]):
                worker_rows = [json.loads(line) for line in (folder / f"WORKER{worker}.jsonl").read_bytes().splitlines()]
                for row in worker_rows:
                    parsed = validate_conditional_full(row["full"])
                    old_source, new_source, base = parsed["source"]
                    old_relay, new_relay = parsed["old_relay"], parsed["new_relay"]
                    labels = parsed["labels"]
                    history = old_history(old_source, base, old_relay, labels)
                    target = set(np.flatnonzero(truth(new_source, base, new_relay, labels, tau=TAU)))
                    stale_target = set(np.flatnonzero(truth(old_source, base, old_relay, labels, tau=TAU)))
                    gold = int(sealed[row["key"]])
                    changed_rows = tuple(graph_manifest_changed_rows(manifest))
                    oracle_rows, oracle_cost, oracle_lb = oracle_row_refresh(
                        new_source, base, new_relay, labels, history,
                        changed_rows, row["tokens"], target)
                    methods = {
                        "stale": (stale_target, []),
                        "exact_warm_cache": exact_warm_cache(new_source, base, labels, history),
                        "reweighted_old_history": replay(old_source, new_source, base, old_relay, labels,
                                                          row["tokens"], history),
                        "ctdr_transport_only": replay(old_source, new_source, base, old_relay, labels,
                                                       row["tokens"], history, radius=True),
                        "ctdr_transport_selective_ranking": replay(old_source, new_source, base, old_relay,
                                                                    labels, row["tokens"], history,
                                                                    radius=True, token_aware=True),
                        "full_ctdr": guarded(old_source, new_source, base, old_relay, new_relay,
                                             labels, row["tokens"], history, changed_rows, True),
                        "max_uncertainty": baseline_recheck(new_source, base, new_relay, labels,
                                                             row["tokens"], history, "max_uncertainty",
                                                             int(row["key"][:8], 16)),
                        "periodic": baseline_recheck(new_source, base, new_relay, labels,
                                                      row["tokens"], history, "periodic"),
                        "random": baseline_recheck(new_source, base, new_relay, labels,
                                                   row["tokens"], history, "random", int(row["key"][:8], 16)),
                        "full_restart": replay(new_source, new_source, base, new_relay, labels,
                                                row["tokens"]),
                        "oracle_row_refresh": (set(target), [(i, None) for i in oracle_rows]),
                    }
                    rec = {
                        "density": density,
                        "panel": panel,
                        "worker": worker,
                        "key": row["key"],
                        "oracle_rows": list(oracle_rows),
                        "oracle_row_opt_tokens": oracle_cost,
                        "oracle_lower_bound_tokens": oracle_lb,
                        "methods": {name: _record(pred, actions, target, gold, row["tokens"])
                                    for name, (pred, actions) in methods.items()},
                    }
                    common_budget = rec["methods"]["full_restart"]["tokens"]
                    for method_record in rec["methods"].values():
                        method_record["budget_tokens"] = common_budget
                        method_record["unused_budget_tokens"] = common_budget - method_record["tokens"]
                    density_rows.append(rec)
                    all_rows.append(rec)
        by_density[density] = density_rows

    def aggregate(rows):
        return {name: _summarize([r["methods"][name] for r in rows]) for name in METHODS}

    result = {
        "schema": "m60_strong_baseline_matrix_v1",
        "protocol": {
            "same_old_transcript": True,
            "same_post_edit_source": True,
            "same_token_prices": True,
            "same_stopping_rule": "Audit.decision exact set, with method-specific legal action ordering",
            "cost_scope": "post_edit_incremental_query_tokens_only",
            "full_ctdr_mask_source": "graph_manifest.edge_to_relay_descendants, never sealed labels or row-level metadata",
            "oracle_definition": "restricted row-refresh optimum over changed relay rows",
            "oracle_lower_bound_definition": "zero if exact warm cache is exact, otherwise cheapest changed-row query",
        },
        "aggregate": aggregate(all_rows),
        "by_density": {k: aggregate(v) for k, v in by_density.items()},
        "rows": all_rows,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Public summary deliberately omits question keys, predictions, targets,
    # sealed labels and action traces.  It retains only aggregate diagnostics.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        grouped.setdefault((str(row["density"]), str(row["panel"])), []).append(row)
    public_panels = {}
    for (density, panel), rows in sorted(grouped.items()):
        public_panels[f"rows{density}_{panel}"] = {
            name: _summarize([r["methods"][name] for r in rows]) for name in METHODS
        }
    public = {
        "schema": "m60_strong_baseline_public_v1",
        "protocol": result["protocol"],
        "aggregate": result["aggregate"],
        "by_density": result["by_density"],
        "by_panel_density": public_panels,
        "privacy": "question keys, sealed labels, predictions, targets and action traces omitted",
    }
    public_path = output.with_name(output.stem + "_PUBLIC.json")
    public_path.write_text(json.dumps(public, indent=2), encoding="utf-8")
    md = [
        "# M60 strong baseline matrix",
        "",
        "All controllers receive the same old certificate transcript, post-edit source, relay outputs, token prices and stopping target. Costs are post-edit incremental query tokens only.",
        "",
        "| method | exact set | action error | false drop | false keep | queries | tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in result["aggregate"].items():
        md.append(f"| {name} | {item['exact_set']:.4f} | {item['action_error']:.4f} | {item['false_drop']:.4f} | {item['false_keep']:.4f} | {item['queries']:.4f} | {item['tokens']:.2f} |")
    md += [
        "",
        "## Oracle interpretation",
        "",
        "`oracle_row_refresh` is an offline row-refresh optimum. It is evaluated only over changed-row actions, so it is a reproducible restricted oracle rather than a global optimum over every cell-level schedule. `oracle_lower_bound_tokens` is a valid necessary-cost lower bound when the exact warm cache is wrong, and zero when the warm cache already matches the target.",
        "",
        "## Attribution",
        "",
        "The matrix distinguishes exact warm-cache reuse from reweighted old-history replay. `ctdr_transport_only` isolates the transport gate, `ctdr_transport_selective_ranking` adds token-aware ordering, and `full_ctdr` uses graph-conditioned changed-row revalidation plus selective ranking.",
    ]
    output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    public_md = [
        "# M60 strong baseline public summary",
        "",
        "This summary omits question keys, sealed labels, predictions, targets and action traces.",
        "",
        "| density/panel | method | exact set | action error | false drop | false keep | queries | tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, methods in public_panels.items():
        for name, item in methods.items():
            public_md.append(f"| {key} | {name} | {item['exact_set']:.4f} | {item['action_error']:.4f} | {item['false_drop']:.4f} | {item['false_keep']:.4f} | {item['queries']:.4f} | {item['tokens']:.2f} |")
    output.with_name(output.stem + "_PUBLIC.md").write_text("\n".join(public_md) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    root = Path("MAB2_recovered")
    manifests = [root / "server_rows1_three_model_manifest.json",
                 root / "server_rows2_three_model_manifest.json",
                 root / "server_rows4_three_model_manifest.json"]
    # Accept paths from the caller while keeping a runnable default.
    if len(sys.argv) > 1:
        manifests = [Path(x) for x in sys.argv[1:-1]]
        output = Path(sys.argv[-1])
    else:
        output = root / "experiments/mechanism_20261122/M60_STRONG_BASELINE_MATRIX.json"
    result = run(manifests, output)
    print(json.dumps(result["aggregate"], indent=2))

