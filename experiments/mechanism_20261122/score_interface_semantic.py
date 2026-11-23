"""Score interface-drift workers against their full new-graph decision set.

This panel has no sealed task labels.  It therefore reports finite audit-set
recovery and incremental query-token cost only.  The input cohort is the
three-task 128-question development panel and is kept separate from service
quality claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from conditional_real_score import baseline_recheck, old_history, price_arrays
from conditional_schema import validate_conditional_full
from ctdr import truth
from score_arcc_independent import generic_guarded, generic_replay


def score(worker_path: Path, input_path: Path):
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    by_key = {row["key"]: row for row in raw}
    records = []
    for line in worker_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        parsed = validate_conditional_full(item["full"])
        old_source, new_source, base = parsed["source"]
        old_relay, new_relay, labels = parsed["old_relay"], parsed["new_relay"], parsed["labels"]
        tokens = item["tokens"]
        changed = by_key[item["key"]].get("interface_drift_rows", item.get("changed_rows", []))
        history = old_history(old_source, base, old_relay, labels)
        target = set(np.flatnonzero(truth(new_source, base, new_relay, tau=0.01)))
        methods = {
            "stale": (set(np.flatnonzero(truth(old_source, base, old_relay, tau=0.01))), []),
            "warm_reweighted": generic_replay(old_source, new_source, base, old_relay, labels, tokens, history),
            "max_uncertainty": baseline_recheck(new_source, base, new_relay, labels, tokens, history, "max_uncertainty", int(item["key"][:8], 16)),
            "random": baseline_recheck(new_source, base, new_relay, labels, tokens, history, "random", int(item["key"][:8], 16)),
            "periodic": baseline_recheck(new_source, base, new_relay, labels, tokens, history, "periodic"),
            "tcr_guard": generic_guarded(old_source, new_source, base, old_relay, new_relay, labels, tokens, history, changed, False),
            "tcr_token": generic_guarded(old_source, new_source, base, old_relay, new_relay, labels, tokens, history, changed, True),
            "cold_restart": generic_replay(new_source, new_source, base, new_relay, labels, tokens),
        }
        for method, (pred, actions) in methods.items():
            records.append({"method": method, "task": input_path.name.split("_interface_")[0],
                            "key": item["key"], "exact": float(pred == target),
                            "action": float(bool(target & pred) == bool(target)),
                            "tokens": float(sum(int(tokens[7 + i] if j is None else tokens[11 + i * 4 + j]) for i, j in actions)),
                            "queries": float(len(actions))})
    out = {}
    for method in sorted({x["method"] for x in records}):
        xs = [x for x in records if x["method"] == method]
        out[method] = {"n": len(xs), "exact": float(np.mean([x["exact"] for x in xs])),
                       "tokens": float(np.mean([x["tokens"] for x in xs])),
                       "queries": float(np.mean([x["queries"] for x in xs]))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = score(args.worker, args.input)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

