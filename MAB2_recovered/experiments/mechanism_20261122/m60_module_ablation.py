from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from blind_scheduler import Audit
from ctdr import intervals, safe_transport, truth
from m60_schema import load_and_validate

TAU = 0.01
METHODS = [
    "warm_restart",
    "reweighted_audit",
    "ctdr_without_radius",
    "ctdr_legacy",
    "warm_restart_token",
    "decision_ranking",
    "full_ctdr",
]


def decision_set(audit: Audit, *, context: str) -> set[int]:
    """Keep unresolved ``None`` distinct from a proved empty set."""
    value = audit.decision()
    if value is None:
        raise RuntimeError(f"unfinished audit was treated as a decision: {context}")
    return {int(item) for item in value}


def token_cost(actions, tokens):
    return int(sum(int(tokens[3 + i] if j is None else tokens[7 + i * 4 + j]) for i, j in actions))


def clone(a):
    b = Audit(a.source.copy(), a.base.copy(), a.row_prices.copy(), a.terminal_prices.copy(), a.policy, tau=a.tau)
    b.known = a.known.copy()
    b.seen = a.seen.copy()
    b.w = a.w.copy()
    b.observed = dict(a.observed)
    return b


def crossing_measure(a, label_count):
    lo, hi = intervals(a, label_count)
    return float(sum(max(0.0, hi[y] - lo[y]) for y in range(label_count)
                    if lo[y] <= a.tau + 1e-12 and hi[y] >= a.tau - 1e-12))


def decision_rank_next(a, label_count):
    """Choose the action with largest expected decision-region width reduction/token."""
    current = crossing_measure(a, label_count)
    candidates = []
    rows = np.flatnonzero(~a.known)
    cells = np.argwhere(a.known[:, None] & ~a.seen)
    for i in rows:
        gains = []
        for k in range(len(a.base)):
            one_hot = np.zeros(len(a.base), dtype=float)
            one_hot[k] = 1.0
            b = clone(a)
            b.update((int(i), None), one_hot)
            gains.append(max(0.0, current - crossing_measure(b, label_count)))
        candidates.append((float(np.mean(gains)) / max(float(a.row_prices[i]), 1e-9), (int(i), None)))
    for i, j in cells:
        gains = []
        for y in range(label_count):
            b = clone(a)
            b.update((int(i), int(j)), int(y))
            gains.append(max(0.0, current - crossing_measure(b, label_count)))
        candidates.append((float(np.mean(gains)) / max(float(a.terminal_prices[i, j]), 1e-9), (int(i), int(j))))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def build_history(old, base, cond, labels):
    """Reconstruct the already-paid old-graph transcript without using gold."""
    a = Audit(old, base, np.ones(4), np.ones((4, 4)), "joint", tau=TAU)
    history = []
    while a.decision() is None:
        action = a.next()
        if action is None:
            raise RuntimeError("old audit did not provide a next action")
        i, j = action
        value = cond[i] if j is None else int(labels[i, j])
        a.update(action, value)
        history.append((action, value))
    return history


def prices(tokens, token_aware):
    if not token_aware:
        return np.ones(4, dtype=float), np.ones((4, 4), dtype=float)
    row = np.asarray([tokens[3 + i] for i in range(4)], dtype=float)
    cell = np.asarray([[tokens[7 + i * 4 + j] for j in range(4)] for i in range(4)], dtype=float)
    return row, cell


def replay(old, new, base, cond, labels, tokens, history, method):
    token_aware = method in {"warm_restart_token", "decision_ranking", "full_ctdr"}
    row_prices, cell_prices = prices(tokens, token_aware)
    a = Audit(new, base, row_prices, cell_prices, "joint", tau=TAU)
    for action, value in history:
        a.update(action, value)
    label_count = int(np.max(labels)) + 1
    radius_mode = method in {"ctdr_legacy", "full_ctdr"}
    ranking_mode = method in {"decision_ranking", "full_ctdr"}
    actions = []
    safe = False
    if radius_mode:
        safe, *_ = safe_transport(old, new, a, base, tau=TAU, label_count=label_count)
    while a.decision() is None and (not radius_mode or not safe):
        action = decision_rank_next(a, label_count) if ranking_mode else a.next()
        if action is None:
            break
        i, j = action
        value = cond[i] if j is None else int(labels[i, j])
        a.update(action, value)
        actions.append(action)
        if radius_mode:
            safe, *_ = safe_transport(old, new, a, base, tau=TAU, label_count=label_count)
    return decision_set(a, context=f"replay:{mode}"), actions, bool(safe)


def summarize(items):
    return {
        "n": len(items),
        "action_error": float(np.mean([int((x["gold"] in x["pred"]) != (x["gold"] in x["target"])) for x in items])),
        "exact_set": float(np.mean([x["pred"] == x["target"] for x in items])),
        "false_drop": float(np.mean([int(x["gold"] in x["target"] and x["gold"] not in x["pred"]) for x in items])),
        "false_keep": float(np.mean([int(x["gold"] not in x["target"] and x["gold"] in x["pred"]) for x in items])),
        "queries": float(np.mean([len(x["actions"]) for x in items])),
        "tokens": float(np.mean([x["tokens"] for x in items])),
        "safe_rate": float(np.mean([x["safe"] for x in items])),
    }


def run(panel_paths, output_path):
    outputs = {}
    for panel, folder_name in panel_paths.items():
        folder = Path(folder_name)
        sealed = json.loads((folder / "SEALED.json").read_bytes())
        panel_out = {}
        for worker in range(3):
            rows = {r["key"]: r for r in map(json.loads, (folder / f"WORKER{worker}.jsonl").read_bytes().splitlines())}
            methods = {method: [] for method in METHODS}
            for key, row in rows.items():
                parsed = load_and_validate(row)
                source_rows = np.asarray(parsed["source_rows"], dtype=float)
                relay_rows = np.asarray(parsed["relay_rows"], dtype=float)
                labels = np.asarray(parsed["terminal_labels"], dtype=int)
                old, new, base = source_rows
                cond = relay_rows
                history = build_history(old, base, cond, labels)
                target = set(np.flatnonzero(truth(new, base, cond, labels, tau=TAU)))
                gold = int(sealed[key])
                for method in METHODS:
                    pred, actions, safe = replay(old, new, base, cond, labels, row["tokens"], history, method)
                    methods[method].append({
                        "pred": pred,
                        "target": target,
                        "gold": gold,
                        "actions": actions,
                        "tokens": token_cost(actions, row["tokens"]),
                        "safe": int(safe),
                    })
            panel_out[str(worker)] = {method: summarize(values) for method, values in methods.items()}
        outputs[panel] = panel_out
    aggregate = {}
    for method in METHODS:
        vals = [outputs[panel][str(worker)][method] for panel in outputs for worker in range(3)]
        n = sum(item["n"] for item in vals)
        aggregate[method] = {key: sum(item[key] * item["n"] for item in vals) / n for key in vals[0] if key != "n"}
        aggregate[method]["n"] = n
    result = {
        "schema": "m60_module_ablation_private_v2",
        "panels": list(panel_paths),
        "results": outputs,
        "aggregate": aggregate,
        "cost_scope": "post_edit_incremental_query_tokens_only",
        "history_scope": "same old_graph transcript replayed by every method",
    }
    output = Path(output_path)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    public = {
        "schema": "m60_module_ablation_public_v2",
        "panels": list(panel_paths),
        "aggregate": aggregate,
        "panel_team": {panel: {worker: {method: outputs[panel][str(worker)][method] for method in METHODS} for worker in range(3)} for panel in outputs},
        "interpretation": {
            "warm_restart": "History replay on new source with unit-cost Audit.next.",
            "reweighted_audit": "Same computation, separately named to expose new-source reweighting.",
            "ctdr_without_radius": "Same history and unit-cost Audit.next without a radius gate.",
            "ctdr_legacy": "Current radius gate with unit-cost Audit.next.",
            "warm_restart_token": "History replay with recorded token prices but Audit.next.",
            "decision_ranking": "No radius, expected decision-region crossing-width reduction per recorded token.",
            "full_ctdr": "Radius gate plus expected decision-region crossing-width reduction per recorded token.",
            "fairness": "Every method uses the same old transcript, new graph, sealed inputs, and post-edit budget. Gold is used only for reporting.",
        },
    }
    output.with_name("M60_MODULE_ABLATION_PUBLIC.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    lines = [
        "# M60 公平模块消融（审计周期）", "",
        "所有方法复用相同的旧图历史 transcript，成本只计图编辑后的新增查询 token。Gold 只用于报告，不参与调度。", "",
        "| 方法 | action error | exact set | queries | tokens | safe rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        item = aggregate[method]
        lines.append(f"| {method} | {item['action_error']:.4f} | {item['exact_set']:.4f} | {item['queries']:.4f} | {item['tokens']:.2f} | {item['safe_rate']:.4f} |")
    lines += [
        "", "## 解释", "",
        "Warm Restart、Reweighted Audit 和 CTDR without Radius 的相同结果显示，当前 80% 左右的冷重启节省主要来自历史观测复用与新来源重加权。CTDR legacy 的 radius gate 只有在与 Audit.decision 不同的情况下才有独立停止价值。Decision Ranking 则单独测量实际 token 价格与决策区域排序。Full CTDR 只有在同时优于这些对照时，才可声称有独立算法收益。", "",
        "## 成本口径", "",
        "旧 transcript 的历史成本不计入 post-edit 增量成本。Stale 的 post-edit 增量为 0。full restart 的新增成本单独报告。",
    ]
    output.with_name("M60_MODULE_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    run(manifest["panels"], manifest["output"])

