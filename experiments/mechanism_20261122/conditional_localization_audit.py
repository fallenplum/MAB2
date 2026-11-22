from __future__ import annotations

import json
from pathlib import Path

from conditional_manifest import derive_changed_rows


def signatures():
    return {str(i): {"prompt": f"p{i}", "model": "m", "upstream": [i]}
            for i in range(4)}


def run(output):
    old = {"relay_signatures": signatures()}
    cases = []
    for name, changed, new in (
        ("source_only", set(), {"relay_signatures": signatures()}),
        ("one_row_signature", {1}, {"relay_signatures": {**signatures(), "1": {"prompt": "p1-new"}}}),
        ("two_row_signature", {1, 3}, {"relay_signatures": {**signatures(), "1": {"prompt": "p1-new"}, "3": {"prompt": "p3-new"}}}),
        ("hidden_propagation_graph_delta", {2}, {"relay_signatures": signatures(), "graph_affected_rows": [2]}),
        ("missing_signature_fallback", {0, 1, 2, 3}, {"relay_signatures": {"0": signatures()["0"]}}),
    ):
        predicted = set(derive_changed_rows(old, new))
        cases.append({"case": name, "expected": sorted(changed), "predicted": sorted(predicted),
                      "false_skip": int(bool(set(changed) - predicted)),
                      "extra_rows": len(predicted - set(changed))})
    result = {"schema": "conditional_localization_audit_v1", "cases": cases,
              "note": "Deterministic manifest audit only; no model inference or labels."}
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run("conditional_localization_audit_public.json")
