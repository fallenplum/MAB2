from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from blind_scheduler import Audit
from ctdr import safe_transport, truth
from m60_module_ablation import decision_rank_next
from m60_schema import load_and_validate

TAU = 0.01
METHODS = ("stale", "source_only_ctdr", "tcr_row", "tcr_ranking", "cold_restart")


def build_history(old, base, cond, labels):
    audit = Audit(old, base, np.ones(4), np.ones((4, 4)), "joint", tau=TAU)
    history = []
    while audit.decision() is None:
        action = audit.next()
        i, j = action
        value = cond[i] if j is None else int(labels[i, j])
        audit.update(action, value)
        history.append((action, value))
    return history, set(audit.decision() or [])


def token_cost(actions, tokens):
    return int(sum(int(tokens[3 + i] if j is None else tokens[7 + i * 4 + j]) for i, j in actions))


def prices(tokens, token_aware):
    if not token_aware:
        return np.ones(4), np.ones((4, 4))
    row = np.asarray([tokens[3 + i] for i in range(4)], dtype=float)
    cell = np.asarray([[tokens[7 + i * 4 + j] for j in range(4)] for i in range(4)], dtype=float)
    return row, cell


def run_audit(new, base, cond_new, labels, tokens, history=(), changed_rows=(),
              mode="source_only_ctdr"):
    token_aware = mode in {"tcr_ranking"}
    row_prices, cell_prices = prices(tokens, token_aware)
    audit = Audit(new, base, row_prices, cell_prices, "joint", tau=TAU)
    changed = set(changed_rows)
    actions = []

    if mode == "tcr_row" or mode == "tcr_ranking":
        old_rows = {i for (i, j), value in history if j is None}
        for i in sorted(changed & old_rows):
            audit.update((i, None), cond_new[i])
            actions.append((i, None))

    for (i, j), value in history:
        if j is None and i in changed and mode in {"tcr_row", "tcr_ranking"}:
            continue
        audit.update((i, j), value)

    safe = False
    if mode == "source_only_ctdr":
        safe, *_ = safe_transport(old_source=None if False else _OLD_SOURCE, new_source=new,
                                  audit=audit, base=base, tau=TAU, label_count=8)

    while audit.decision() is None and (mode != "source_only_ctdr" or not safe):
        action = decision_rank_next(audit, 8) if mode == "tcr_ranking" else audit.next()
        if action is None:
            break
        i, j = action
        value = cond_new[i] if j is None else int(labels[i, j])
        audit.update(action, value)
        actions.append(action)
        if mode == "source_only_ctdr":
            safe, *_ = safe_transport(old_source=_OLD_SOURCE, new_source=new,
                                      audit=audit, base=base, tau=TAU, label_count=8)
    return set(audit.decision() or []), actions, bool(safe)


def source_only(old, new, base, cond_new, labels, tokens, history):
    global _OLD_SOURCE
    _OLD_SOURCE = old
    return run_audit(new, base, cond_new, labels, tokens, history=history, mode="source_only_ctdr")


def tcr(old, new, base, cond_new, labels, tokens, history, ranking=False, changed_rows=()):
    global _OLD_SOURCE
    _OLD_SOURCE = old
    mode = "tcr_ranking" if ranking else "tcr_row"
    return run_audit(new, base, cond_new, labels, tokens, history=history,
                     changed_rows=changed_rows, mode=mode)


def cold(new, base, cond_new, labels, tokens):
    global _OLD_SOURCE
    _OLD_SOURCE = new
    return run_audit(new, base, cond_new, labels, tokens, history=(), mode="cold_restart")


def metrics(pred, target, gold):
    return {
        "decision_error": int(pred != target),
        "false_drop": int(bool(target - pred)),
        "false_keep": int(bool(pred - target)),
        "gold_membership_error": int((gold in pred) != (gold in target)),
        "exact_set": int(pred == target),
    }


def run(manifest, output_path):
    alphas = (0.25, 0.50)
    row_sets = {"A": (1,), "ABCD": (0, 1, 2, 3)}
    rows_out = []
    for panel, folder_name in manifest["panels"].items():
        folder = Path(folder_name)
        sealed = json.loads((folder / "SEALED.json").read_bytes())
        for worker in range(3):
            rows = {r["key"]: r for r in map(json.loads, (folder / f"WORKER{worker}.jsonl").read_bytes().splitlines())}
            for key, row in rows.items():
                parsed = load_and_validate(row)
                source_rows = np.asarray(parsed["source_rows"], dtype=float)
                cond_old = np.asarray(parsed["relay_rows"], dtype=float)
                labels = np.asarray(parsed["terminal_labels"], dtype=int)
                old, new, base = source_rows
                history, old_target = build_history(old, base, cond_old, labels)
                for alpha in alphas:
                    for set_name, changed_rows in row_sets.items():
                        cond_new = cond_old.copy()
                        cond_new[list(changed_rows)] = ((1 - alpha) * cond_old[list(changed_rows)]
                                                        + alpha * base)
                        target = set(np.flatnonzero(truth(new, base, cond_new, labels, tau=TAU)))
                        gold = int(sealed[key])
                        result = {}
                        result["stale"] = (old_target, [], False)
                        result["source_only_ctdr"] = source_only(old, new, base, cond_new, labels, row["tokens"], history)
                        result["tcr_row"] = tcr(old, new, base, cond_new, labels, row["tokens"], history, ranking=False, changed_rows=changed_rows)
                        result["tcr_ranking"] = tcr(old, new, base, cond_new, labels, row["tokens"], history, ranking=True, changed_rows=changed_rows)
                        result["cold_restart"] = cold(new, base, cond_new, labels, row["tokens"])
                        for method, (pred, actions, safe) in result.items():
                            row_result = {"panel": panel, "worker": worker, "key": key, "alpha": alpha,
                                          "changed_rows": set_name, "method": method,
                                          "queries": len(actions), "tokens": token_cost(actions, row["tokens"]),
                                          "safe": int(safe), "target_size": len(target)}
                            row_result.update(metrics(pred, target, gold))
                            rows_out.append(row_result)
    aggregate = {}
    groups = sorted({(r["alpha"], r["changed_rows"]) for r in rows_out})
    for alpha, set_name in groups:
        aggregate[str(alpha) + ":" + set_name] = {}
        for method in METHODS:
            subset = [r for r in rows_out if r["alpha"] == alpha and r["changed_rows"] == set_name and r["method"] == method]
            aggregate[str(alpha) + ":" + set_name][method] = {
                "n": len(subset),
                "decision_error": float(np.mean([r["decision_error"] for r in subset])),
                "exact_set": float(np.mean([r["exact_set"] for r in subset])),
                "false_drop": float(np.mean([r["false_drop"] for r in subset])),
                "false_keep": float(np.mean([r["false_keep"] for r in subset])),
                "gold_membership_error": float(np.mean([r["gold_membership_error"] for r in subset])),
                "queries": float(np.mean([r["queries"] for r in subset])),
                "tokens": float(np.mean([r["tokens"] for r in subset])),
                "safe_rate": float(np.mean([r["safe"] for r in subset])),
            }
    output = {"schema": "conditional_relay_drift_diagnostic_v1", "aggregate": aggregate,
              "n_rows": len(rows_out), "panels": list(manifest["panels"]),
              "note": "Synthetic conditional drift on frozen M60 model-side vectors; not independent confirmation."}
    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    Path(output_path).with_name("CONDITIONAL_DRIFT_DIAGNOSTIC_DETAIL.json").write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    run(manifest, manifest["output"])

