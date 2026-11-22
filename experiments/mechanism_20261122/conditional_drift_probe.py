from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from blind_scheduler import Audit
from ctdr import safe_transport, truth
from m60_module_ablation import decision_rank_next, token_cost
from m60_schema import load_and_validate

TAU = 0.01


def build_history(old, base, cond, labels):
    audit = Audit(old, base, np.ones(4), np.ones((4, 4)), "joint", tau=TAU)
    history = []
    while audit.decision() is None:
        action = audit.next()
        if action is None:
            raise RuntimeError("old audit has no next action")
        i, j = action
        value = cond[i] if j is None else int(labels[i, j])
        audit.update(action, value)
        history.append((action, value))
    return history


def warm_replay(new, base, old_cond, labels, history):
    audit = Audit(new, base, np.ones(4), np.ones((4, 4)), "joint", tau=TAU)
    for action, value in history:
        audit.update(action, value)
    return audit


def source_only(old, new, base, old_cond, labels, history, use_radius):
    audit = warm_replay(new, base, old_cond, labels, history)
    safe = False
    if use_radius:
        safe, *_ = safe_transport(old, new, audit, base, tau=TAU, label_count=8)
    actions = []
    while audit.decision() is None and (not use_radius or not safe):
        action = audit.next()
        if action is None:
            break
        i, j = action
        value = old_cond[i] if j is None else int(labels[i, j])
        audit.update(action, value)
        actions.append(action)
        if use_radius:
            safe, *_ = safe_transport(old, new, audit, base, tau=TAU, label_count=8)
    return set(audit.decision() or []), actions, bool(safe)


def guarded_replay(old, new, base, old_cond, new_cond, labels, tokens, history, token_aware):
    row_prices = np.asarray([tokens[3 + i] for i in range(4)], dtype=float) if token_aware else np.ones(4)
    cell_prices = np.asarray([[tokens[7 + i * 4 + j] for j in range(4)] for i in range(4)], dtype=float) if token_aware else np.ones((4, 4))
    audit = Audit(new, base, row_prices, cell_prices, "joint", tau=TAU)
    row_history = {int(i): value for (i, j), value in history if j is None}
    cell_history = [((int(i), int(j)), value) for (i, j), value in history if j is not None]
    actions = []
    changed = [i for i in range(4) if not np.allclose(old_cond[i], new_cond[i], atol=1e-12)]
    for i in range(4):
        if i in changed:
            audit.update((i, None), new_cond[i])
            actions.append((i, None))
        elif i in row_history:
            audit.update((i, None), row_history[i])
    for action, value in cell_history:
        i, j = action
        audit.update(action, value)
    while audit.decision() is None:
        action = decision_rank_next(audit, 8) if token_aware else audit.next()
        if action is None:
            break
        i, j = action
        value = new_cond[i] if j is None else int(labels[i, j])
        audit.update(action, value)
        actions.append(action)
    return set(audit.decision() or []), actions, len(changed)


def cold_restart(new, base, new_cond, labels, tokens, token_aware):
    row_prices = np.asarray([tokens[3 + i] for i in range(4)], dtype=float) if token_aware else np.ones(4)
    cell_prices = np.asarray([[tokens[7 + i * 4 + j] for j in range(4)] for i in range(4)], dtype=float) if token_aware else np.ones((4, 4))
    audit = Audit(new, base, row_prices, cell_prices, "joint", tau=TAU)
    actions = []
    while audit.decision() is None:
        action = decision_rank_next(audit, 8) if token_aware else audit.next()
        if action is None:
            break
        i, j = action
        value = new_cond[i] if j is None else int(labels[i, j])
        audit.update(action, value)
        actions.append(action)
    return set(audit.decision() or []), actions


def drift_condition(cond, strength, changed_rows=None):
    # A declared relay-channel change. Columns remain A-D and rows retain the
    # same state names, so this probes conditional drift without changing the schema.
    rotated = np.roll(cond, 1, axis=1)
    mixed = cond.copy()
    rows = range(cond.shape[0]) if changed_rows is None else changed_rows
    for index in rows:
        mixed[index] = (1.0 - strength) * cond[index] + strength * rotated[index]
    return mixed / mixed.sum(axis=1, keepdims=True)


def summarize(items):
    n = len(items)
    return {
        "n": n,
        "action_error": float(np.mean([x["action_error"] for x in items])),
        "exact_set": float(np.mean([x["exact"] for x in items])),
        "false_drop": float(np.mean([x["false_drop"] for x in items])),
        "false_keep": float(np.mean([x["false_keep"] for x in items])),
        "queries": float(np.mean([len(x["actions"]) for x in items])),
        "tokens": float(np.mean([x["tokens"] for x in items])),
        "drift_rows": float(np.mean([x["drift_rows"] for x in items])),
        "mismatch": int(sum(x["action_error"] for x in items)),
    }


def run(manifest):
    outputs = {}
    for panel, folder_name in manifest["panels"].items():
        folder = Path(folder_name)
        sealed = json.loads((folder / "SEALED.json").read_bytes())
        panel_out = {}
        for worker in range(3):
            rows = [json.loads(line) for line in (folder / f"WORKER{worker}.jsonl").read_bytes().splitlines()]
            methods = {name: [] for name in ["stale", "source_only_ctdr", "tcr_guard", "tcr_token", "cold_restart"]}
            for row in rows:
                parsed = load_and_validate(row)
                source = np.asarray(parsed["source_rows"], dtype=float)
                old, new, base = source
                old_cond = np.asarray(parsed["relay_rows"], dtype=float)
                labels = np.asarray(parsed["terminal_labels"], dtype=int)
                history = build_history(old, base, old_cond, labels)
                new_cond = drift_condition(old_cond, float(manifest["strength"]), manifest.get("changed_rows"))
                target = set(np.flatnonzero(truth(new, base, new_cond, labels, tau=TAU)))
                gold = int(sealed[row["key"]])
                stale_pred = set(warm_replay(new, base, old_cond, labels, history).decision() or [])
                for method in methods:
                    if method == "stale":
                        pred, actions, drift_rows = stale_pred, [], 0
                    elif method == "source_only_ctdr":
                        pred, actions, _ = source_only(old, new, base, old_cond, labels, history, True)
                        drift_rows = 0
                    elif method == "tcr_guard":
                        pred, actions, drift_rows = guarded_replay(old, new, base, old_cond, new_cond, labels, row["tokens"], history, False)
                    elif method == "tcr_token":
                        pred, actions, drift_rows = guarded_replay(old, new, base, old_cond, new_cond, labels, row["tokens"], history, True)
                    else:
                        pred, actions = cold_restart(new, base, new_cond, labels, row["tokens"], False)
                        drift_rows = 4
                    action_error = int((gold in pred) != (gold in target))
                    methods[method].append({
                        "action_error": action_error,
                        "exact": int(pred == target),
                        "false_drop": int(gold in target and gold not in pred),
                        "false_keep": int(gold not in target and gold in pred),
                        "actions": actions,
                        "tokens": token_cost(actions, row["tokens"]),
                        "drift_rows": drift_rows,
                    })
            panel_out[str(worker)] = {method: summarize(values) for method, values in methods.items()}
        outputs[panel] = panel_out
    aggregate = {}
    for method in ["stale", "source_only_ctdr", "tcr_guard", "tcr_token", "cold_restart"]:
        vals = [outputs[p][str(w)][method] for p in outputs for w in range(3)]
        n = sum(v["n"] for v in vals)
        aggregate[method] = {k: sum(v[k] * v["n"] for v in vals) / n for k in vals[0] if k not in {"n", "mismatch"}}
        aggregate[method]["n"] = n
        aggregate[method]["mismatch"] = sum(v["mismatch"] for v in vals)
    return {"schema": "m60_conditional_drift_probe_v2", "strength": manifest["strength"], "changed_rows": manifest.get("changed_rows"), "panels": list(manifest["panels"]), "aggregate": aggregate, "panel_team": outputs}


if __name__ == "__main__":
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(manifest["output"])
    result = run(manifest)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    public = {"schema": result["schema"], "strength": result["strength"], "changed_rows": result["changed_rows"], "panels": result["panels"], "aggregate": result["aggregate"], "interpretation": {"stale": "Reuses old relay observations after declared conditional drift.", "source_only_ctdr": "Uses source-only radius and assumes old relay outputs remain valid.", "tcr_guard": "Requeries rows whose relay distributions changed, then reuses legal terminal observations.", "tcr_token": "Same guard with recorded token prices and decision-region ranking.", "cold_restart": "Queries the new conditional channel from an empty audit."}}
    output.with_name("M60_CONDITIONAL_DRIFT_PUBLIC.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))

