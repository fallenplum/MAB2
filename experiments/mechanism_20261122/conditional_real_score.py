from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from blind_scheduler import Audit
from conditional_schema import validate_conditional_full
from ctdr import safe_transport, truth
from m60_module_ablation import decision_rank_next

TAU = 0.01
METHODS = ("stale", "warm_reweighted", "source_only_ctdr", "tcr_guard", "tcr_token",
           "tcr_bound", "tcr_bound_token", "tcr_density_gate", "tcr_density_gate_token",
           "cold_restart", "cold_token")


def price_arrays(tokens, token_aware):
    if not token_aware:
        return np.ones(4), np.ones((4, 4))
    row = np.asarray([tokens[7 + i] for i in range(4)], dtype=float)
    cell = np.asarray([[tokens[11 + i * 4 + j] for j in range(4)] for i in range(4)], dtype=float)
    return row, cell


def cost(actions, tokens):
    return int(sum(int(tokens[7 + i] if j is None else tokens[11 + i * 4 + j]) for i, j in actions))


def old_history(old, base, relay, labels):
    audit = Audit(old, base, np.ones(4), np.ones((4, 4)), "joint", tau=TAU)
    history = []
    while audit.decision() is None:
        i, j = audit.next()
        value = relay[i] if j is None else int(labels[i, j])
        audit.update((i, j), value)
        history.append(((i, j), value))
    return history


def replay(old_source, new_source, base, relay, labels, tokens, history=(), radius=False, token_aware=False):
    row_prices, cell_prices = price_arrays(tokens, token_aware)
    audit = Audit(new_source, base, row_prices, cell_prices, "joint", tau=TAU)
    for action, value in history:
        audit.update(action, value)
    actions = []
    safe = False
    if radius:
        safe, *_ = safe_transport(old_source, new_source, audit, base, tau=TAU, label_count=8)
    while audit.decision() is None and (not radius or not safe):
        action = decision_rank_next(audit, 8) if token_aware else audit.next()
        if action is None:
            break
        i, j = action
        value = relay[i] if j is None else int(labels[i, j])
        audit.update(action, value)
        actions.append(action)
        if radius:
            safe, *_ = safe_transport(old_source, new_source, audit, base, tau=TAU, label_count=8)
    return set(audit.decision() or []), actions


def guarded(old_source, new_source, base, old_relay, new_relay, labels, tokens, history, changed_rows, token_aware):
    row_prices, cell_prices = price_arrays(tokens, token_aware)
    audit = Audit(new_source, base, row_prices, cell_prices, "joint", tau=TAU)
    changed = set(changed_rows)
    old_rows = {int(i): value for (i, j), value in history if j is None}
    cells = [((int(i), int(j)), value) for (i, j), value in history if j is not None]
    actions = []
    for i in range(4):
        if i in changed:
            audit.update((i, None), new_relay[i])
            actions.append((i, None))
        elif i in old_rows:
            audit.update((i, None), old_rows[i])
    for action, value in cells:
        audit.update(action, value)
    while audit.decision() is None:
        action = decision_rank_next(audit, 8) if token_aware else audit.next()
        if action is None:
            break
        i, j = action
        value = new_relay[i] if j is None else int(labels[i, j])
        audit.update(action, value)
        actions.append(action)
    return set(audit.decision() or []), actions


def bounded_decision(audit, unknown_rows, labels, label_count):
    """Return a decision only when every unresolved row has a safe bound."""
    pending = np.where(audit.known[:, None] & ~audit.seen, audit.w, 0.0)
    pending_low = float(np.minimum(pending, 0.0).sum())
    pending_high = float(np.maximum(pending, 0.0).sum())
    row_low = np.zeros(label_count, dtype=float)
    row_high = np.zeros(label_count, dtype=float)
    for index in unknown_rows:
        weight = float(audit.source[index])
        base_mass = np.zeros(label_count, dtype=float)
        present = np.zeros(label_count, dtype=bool)
        for hypothesis, label in enumerate(labels[index]):
            base_mass[int(label)] += audit.base[hypothesis]
            present[int(label)] = True
        row_low -= weight * base_mass
        row_high += weight * (present.astype(float) - base_mass)
    low = pending_low + row_low
    high = pending_high + row_high
    observed = np.zeros(label_count, dtype=float)
    for label, value in audit.observed.items():
        if 0 <= int(label) < label_count:
            observed[int(label)] = float(value)
    keep = observed + low > TAU + 1e-12
    drop = observed + high < TAU - 1e-12
    return set(np.flatnonzero(keep)) if np.all(keep | drop) else None


def bounded_guard(old_source, new_source, base, old_relay, new_relay, labels,
                  tokens, history, changed_rows, token_aware):
    """Guard transport with worst-case conditional-row intervals.

    Changed rows are queried only when their worst-case contribution can still
    cross the action threshold. This is a conservative optimization over
    unconditional row revalidation.
    """
    row_prices, cell_prices = price_arrays(tokens, token_aware)
    audit = Audit(new_source, base, row_prices, cell_prices, "joint", tau=TAU)
    changed = set(changed_rows)
    old_rows = {int(i): value for (i, j), value in history if j is None}
    old_cells = {}
    for (i, j), value in history:
        if j is not None:
            old_cells.setdefault(int(i), []).append(((int(i), int(j)), value))
    for i in range(4):
        if i not in changed and i in old_rows:
            audit.update((i, None), old_rows[i])
            for action, value in old_cells.get(i, ()):
                audit.update(action, value)
    actions = []
    unknown = set(changed)
    pred = bounded_decision(audit, unknown, labels, 8)
    while pred is None:
        if unknown:
            scores = {i: float(audit.source[i]) / max(float(row_prices[i]), 1e-12)
                      for i in unknown}
            i = max(scores, key=scores.get)
            audit.update((i, None), new_relay[i])
            actions.append((i, None))
            unknown.remove(i)
            for action, value in old_cells.get(i, ()):
                audit.update(action, value)
        else:
            action = decision_rank_next(audit, 8) if token_aware else audit.next()
            if action is None:
                break
            i, j = action
            value = new_relay[i] if j is None else int(labels[i, j])
            audit.update(action, value)
            actions.append(action)
        pred = bounded_decision(audit, unknown, labels, 8) if unknown else set(audit.decision() or [])
    return set(pred or []), actions


def density_gate(old_source, new_source, base, old_relay, new_relay, labels,
                 tokens, history, changed_rows, token_aware):
    """Use transport only for a singleton edit, otherwise cold restart."""
    if len(tuple(changed_rows)) <= 1:
        return guarded(old_source, new_source, base, old_relay, new_relay, labels,
                       tokens, history, changed_rows, token_aware)
    return replay(new_source, new_source, base, new_relay, labels, tokens,
                  token_aware=token_aware)


def summarize(items):
    n = len(items)
    return {"n": n,
            "action_error": float(np.mean([x["action_error"] for x in items])),
            "exact_set": float(np.mean([x["exact_set"] for x in items])),
            "false_drop": float(np.mean([x["false_drop"] for x in items])),
            "false_keep": float(np.mean([x["false_keep"] for x in items])),
            "queries": float(np.mean([len(x["actions"]) for x in items])),
            "tokens": float(np.mean([x["tokens"] for x in items]))}


def run(manifest):
    outputs = {}
    manifest_changed_rows = tuple(int(i) for i in manifest.get("changed_rows", [1]))
    for panel, folder_name in manifest["panels"].items():
        folder = Path(folder_name)
        sealed = json.loads((folder / "SEALED.json").read_bytes())
        panel_out = {}
        for worker in manifest.get("workers", [0, 1, 2]):
            rows = [json.loads(line) for line in (folder / f"WORKER{worker}.jsonl").read_bytes().splitlines()]
            methods = {name: [] for name in METHODS}
            for row in rows:
                changed_rows = tuple(int(i) for i in row.get("changed_rows", manifest_changed_rows))
                parsed = validate_conditional_full(row["full"])
                old_source, new_source, base = parsed["source"]
                old_relay, new_relay = parsed["old_relay"], parsed["new_relay"]
                labels = parsed["labels"]
                history = old_history(old_source, base, old_relay, labels)
                target = set(np.flatnonzero(truth(new_source, base, new_relay, labels, tau=TAU)))
                gold = int(sealed[row["key"]])
                result = {
                    "stale": (set(np.flatnonzero(truth(old_source, base, old_relay, labels, tau=TAU))), []),
                    "warm_reweighted": replay(old_source, new_source, base, old_relay, labels, row["tokens"], history),
                    "source_only_ctdr": replay(old_source, new_source, base, old_relay, labels, row["tokens"], history, radius=True),
                    "tcr_guard": guarded(old_source, new_source, base, old_relay, new_relay, labels, row["tokens"], history, changed_rows, False),
                    "tcr_token": guarded(old_source, new_source, base, old_relay, new_relay, labels, row["tokens"], history, changed_rows, True),
                    "tcr_bound": bounded_guard(old_source, new_source, base, old_relay, new_relay, labels, row["tokens"], history, changed_rows, False),
                    "tcr_bound_token": bounded_guard(old_source, new_source, base, old_relay, new_relay, labels, row["tokens"], history, changed_rows, True),
                    "tcr_density_gate": density_gate(old_source, new_source, base, old_relay, new_relay, labels, row["tokens"], history, changed_rows, False),
                    "tcr_density_gate_token": density_gate(old_source, new_source, base, old_relay, new_relay, labels, row["tokens"], history, changed_rows, True),
                    "cold_restart": replay(new_source, new_source, base, new_relay, labels, row["tokens"]),
                    "cold_token": replay(new_source, new_source, base, new_relay, labels, row["tokens"], token_aware=True),
                }
                for method, (pred, actions) in result.items():
                    methods[method].append({"action_error": int((gold in pred) != (gold in target)),
                                            "exact_set": int(pred == target),
                                            "false_drop": int(bool(target - pred)),
                                            "false_keep": int(bool(pred - target)),
                                            "actions": actions,
                                            "tokens": cost(actions, row["tokens"])})
            panel_out[str(worker)] = {method: summarize(values) for method, values in methods.items()}
        outputs[panel] = panel_out
    aggregate = {}
    for method in METHODS:
        vals = [outputs[p][str(w)][method] for p in outputs for w in manifest.get("workers", [0, 1, 2])]
        n = sum(v["n"] for v in vals)
        aggregate[method] = {"n": n}
        aggregate[method].update({k: sum(v[k] * v["n"] for v in vals) / n
                                  for k in vals[0] if k != "n"})
    return {"schema": "m60_conditional_real_replay_v2", "panels": list(manifest["panels"]),
            "changed_rows": list(manifest_changed_rows), "aggregate": aggregate, "panel_team": outputs}


if __name__ == "__main__":
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = run(manifest)
    output = Path(manifest["output"])
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    public = {k: result[k] for k in ("schema", "panels", "changed_rows", "aggregate")}
    public["interpretation"] = "Real conditional worker replay. Gold is used only for reporting."
    output.with_name("M60_CONDITIONAL_REAL_PUBLIC.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
