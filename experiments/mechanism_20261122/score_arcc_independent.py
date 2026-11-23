"""Score independent ARCC conditional worker outputs.

The raw gold map is preserved.  Two source records contain malformed labels
outside the four-option range.  The report therefore includes an unscored
view that excludes those records and a corrected view using an explicit,
reviewed correction map.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from blind_scheduler import Audit  # noqa: E402
from conditional_real_score import (  # noqa: E402
    TAU,
    baseline_recheck,
    bounded_decision,
    cost,
    decision_set,
    decision_from_history,
    old_history,
    price_arrays,
)
from conditional_schema import validate_conditional_full  # noqa: E402
from ctdr import safe_transport, truth  # noqa: E402
from m60_module_ablation import decision_rank_next  # noqa: E402


# The original gold map stores these two entries as negative values.  The
# corrections are derived from the visible four-option question text and are
# reported explicitly in every output.
GOLD_CORRECTIONS = {
    "510e0d46d05c7200521aeeeacc5629844e154e2d8f01c4e3d16437a9c60b7acd": {
        "raw": -15,
        "corrected": 1,
        "reason": "raw answerKey is numeric '2' and was parsed as ord('2')-ord('A')=-15; convert 1-based answer 2 to option B/index1",
    },
    "3601c17696b3578a2cf5e79ed40743acf07619b700f5b0c9d658bc3882ffb376": {
        "raw": -13,
        "corrected": 3,
        "reason": "raw answerKey is numeric '4' and was parsed as ord('4')-ord('A')=-13; convert 1-based answer 4 to option D/index3",
    },
}

METHODS = (
    "stale",
    "relay_refresh",
    "warm_reweighted",
    "max_uncertainty",
    "random_recheck",
    "periodic_recheck",
    "source_only_ctdr",
    "tcr_guard",
    "tcr_token",
    "tcr_bound",
    "tcr_bound_token",
    "cold_restart",
    "cold_token",
)


def generic_replay(old_source, new_source, base, relay, labels, tokens, history=(), radius=False, token_aware=False):
    """Replay with the observed ARCC label cardinality instead of a fixed 8."""
    label_count = int(np.max(labels)) + 1
    row_prices, cell_prices = price_arrays(tokens, token_aware)
    audit = Audit(new_source, base, row_prices, cell_prices, "joint", tau=TAU)
    for action, value in history:
        audit.update(action, value)
    actions = []
    safe = False
    if radius:
        safe, *_ = safe_transport(old_source, new_source, audit, base, tau=TAU, label_count=label_count)
    while audit.decision() is None and (not radius or not safe):
        action = decision_rank_next(audit, label_count) if token_aware else audit.next()
        if action is None:
            break
        i, j = action
        value = relay[i] if j is None else int(labels[i, j])
        audit.update(action, value)
        actions.append(action)
        if radius:
            safe, *_ = safe_transport(old_source, new_source, audit, base, tau=TAU, label_count=label_count)
    return decision_set(audit, context="arcc-replay"), actions


def generic_guarded(old_source, new_source, base, old_relay, new_relay, labels, tokens, history, changed_rows, token_aware):
    label_count = int(np.max(labels)) + 1
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
        action = decision_rank_next(audit, label_count) if token_aware else audit.next()
        if action is None:
            break
        i, j = action
        value = new_relay[i] if j is None else int(labels[i, j])
        audit.update(action, value)
        actions.append(action)
    return decision_set(audit, context="arcc-baseline"), actions


def generic_bounded_guard(old_source, new_source, base, old_relay, new_relay, labels, tokens, history, changed_rows, token_aware):
    label_count = int(np.max(labels)) + 1
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
    pred = bounded_decision(audit, unknown, labels, label_count)
    while pred is None:
        if unknown:
            scores = {i: float(audit.source[i]) / max(float(row_prices[i]), 1e-12) for i in unknown}
            i = max(scores, key=scores.get)
            audit.update((i, None), new_relay[i])
            actions.append((i, None))
            unknown.remove(i)
            for action, value in old_cells.get(i, ()):
                audit.update(action, value)
        else:
            action = decision_rank_next(audit, label_count) if token_aware else audit.next()
            if action is None:
                break
            i, j = action
            value = new_relay[i] if j is None else int(labels[i, j])
            audit.update(action, value)
            actions.append(action)
        pred = bounded_decision(audit, unknown, labels, label_count) if unknown else decision_set(audit, context="arcc-guard-no-unknown")
    if pred is None:
        raise RuntimeError("unfinished bounded ARC audit was treated as an empty decision")
    return {int(value) for value in pred}, actions


def values(source, base, relay, labels):
    weights = np.asarray(source)[:, None] * (np.asarray(relay) - np.asarray(base)[None, :])
    return np.asarray(
        [weights[np.asarray(labels) == label].sum() for label in range(int(np.max(labels)) + 1)]
    )


def selected_relay(old_relay, new_relay, actions):
    relay = np.asarray(old_relay, dtype=float).copy()
    for action in actions:
        i, j = action
        if j is None:
            relay[int(i)] = np.asarray(new_relay)[int(i)]
    return relay


def relay_refresh(old_relay, new_relay, tokens, changed_rows):
    """Refresh only graph rows whose relay interface changed.

    This policy intentionally does not claim a graph certificate.  It tests
    whether updating the affected relay values alone is sufficient for the
    downstream answer action at a much smaller incremental cost.
    """
    actions = [(int(i), None) for i in changed_rows]
    return actions


def summarize(records, gold_mode):
    if not records:
        return {"n": 0}
    keys = (
        "decision_exact",
        "answer_action_exact",
        "answer_accuracy",
        "tokens",
        "queries",
    )
    return {
        "n": len(records),
        **{key: float(np.mean([record[key] for record in records])) for key in keys},
        "gold_mode": gold_mode,
    }


def evaluate_row(item, raw_gold):
    parsed = validate_conditional_full(item["full"])
    source = parsed["source"]
    old_source, new_source, base = source
    old_relay, new_relay = parsed["old_relay"], parsed["new_relay"]
    labels = parsed["labels"]
    history = old_history(old_source, base, old_relay, labels)
    target = set(np.flatnonzero(truth(new_source, base, new_relay, labels, tau=TAU)))
    old_target = decision_from_history(old_source, base, history, context="stale")
    changed = tuple(int(value) for value in item.get("changed_rows", []))
    tokens = item["tokens"]

    result = {
        "stale": (old_target, []),
        "relay_refresh": (set(), relay_refresh(old_relay, new_relay, tokens, changed)),
        "warm_reweighted": generic_replay(old_source, new_source, base, new_relay, labels, tokens, history),
        "max_uncertainty": baseline_recheck(
            new_source, base, new_relay, labels, tokens, history, "max_uncertainty", int(item["key"][:8], 16)
        ),
        "random_recheck": baseline_recheck(
            new_source, base, new_relay, labels, tokens, history, "random", int(item["key"][:8], 16)
        ),
        "periodic_recheck": baseline_recheck(
            new_source, base, new_relay, labels, tokens, history, "periodic"
        ),
        "source_only_ctdr": generic_replay(
            old_source, new_source, base, new_relay, labels, tokens, history, radius=True
        ),
        "tcr_guard": generic_guarded(
            old_source, new_source, base, old_relay, new_relay, labels, tokens, history, changed, False
        ),
        "tcr_token": generic_guarded(
            old_source, new_source, base, old_relay, new_relay, labels, tokens, history, changed, True
        ),
        "tcr_bound": generic_bounded_guard(
            old_source, new_source, base, old_relay, new_relay, labels, tokens, history, changed, False
        ),
        "tcr_bound_token": generic_bounded_guard(
            old_source, new_source, base, old_relay, new_relay, labels, tokens, history, changed, True
        ),
        "cold_restart": generic_replay(new_source, new_source, base, new_relay, labels, tokens),
        "cold_token": generic_replay(new_source, new_source, base, new_relay, labels, tokens, token_aware=True),
    }

    raw_value = int(raw_gold)
    correction = GOLD_CORRECTIONS.get(item["key"])
    corrected_value = correction["corrected"] if correction else raw_value
    valid_raw = 0 <= raw_value < 4
    output = {}
    full_truth_answer = int(np.argmax(values(new_source, base, new_relay, labels)))
    for method, (prediction, actions) in result.items():
        relay = selected_relay(old_relay, new_relay, actions)
        predicted_answer = int(np.argmax(values(new_source, base, relay, labels)))
        common = {
            "decision_exact": float(set(prediction) == target),
            "answer_action_exact": float(predicted_answer == full_truth_answer),
            "tokens": float(cost(actions, tokens)),
            "queries": float(len(actions)),
            "predicted_answer": predicted_answer,
        }
        common["answer_accuracy_corrected"] = float(predicted_answer == corrected_value)
        common["answer_accuracy_raw"] = float(predicted_answer == raw_value) if valid_raw else None
        output[method] = common
    return {
        "key": item["key"],
        "raw_gold": raw_value,
        "corrected_gold": corrected_value,
        "valid_raw_gold": valid_raw,
        "correction": correction,
        "methods": output,
    }


def aggregate(rows, gold_mode):
    output = {}
    for method in METHODS:
        records = []
        for row in rows:
            if gold_mode == "unscored" and not row["valid_raw_gold"]:
                continue
            record = row["methods"][method]
            records.append(
                {
                    "decision_exact": record["decision_exact"],
                    "answer_action_exact": record["answer_action_exact"],
                    "answer_accuracy": (
                        record["answer_accuracy_corrected"]
                        if gold_mode == "corrected"
                        else record["answer_accuracy_raw"]
                    ),
                    "tokens": record["tokens"],
                    "queries": record["queries"],
                }
            )
        output[method] = summarize(records, gold_mode)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    rows = []
    for line in args.worker.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows.append(evaluate_row(item, gold[item["key"]]))
    result = {
        "schema": "arcc_independent_conditional_score_v1",
        "worker": str(args.worker),
        "n_rows": len(rows),
        "n_invalid_raw_gold": sum(not row["valid_raw_gold"] for row in rows),
        "invalid_raw_gold_keys": [row["key"] for row in rows if not row["valid_raw_gold"]],
        "correction_map": GOLD_CORRECTIONS,
        "aggregate_corrected": aggregate(rows, "corrected"),
        "aggregate_unscored": aggregate(rows, "unscored"),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    public = {
        key: result[key]
        for key in (
            "schema",
            "worker",
            "n_rows",
            "n_invalid_raw_gold",
            "invalid_raw_gold_keys",
            "correction_map",
            "aggregate_corrected",
            "aggregate_unscored",
        )
    }
    args.output.with_name(args.output.stem + "_PUBLIC.json").write_text(
        json.dumps(public, indent=2), encoding="utf-8"
    )
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()

