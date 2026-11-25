"""Offline replay of matched E0 cache and certificate baselines.

This script is a scorer-side replay. It never changes the online run and does
not claim that an oracle or replay-only policy was deployable.
"""

from __future__ import annotations

import hashlib
import json
import re
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def group_id(topic: str, salt: str) -> str:
    return "e0g_" + hashlib.sha256((salt + "|" + topic).encode()).hexdigest()[:16]


def canonical(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def certificate(labels: list[int | None], threshold: float = 0.80) -> dict[str, object]:
    if not labels:
        return {"status": "INVALID", "decision": None, "n": 0, "required_support": 0, "supported": 0, "failed": 0, "unknown": 0}
    required = ceil(threshold * len(labels))
    supported = sum(label == 1 for label in labels)
    failed = sum(label == 0 for label in labels)
    unknown = sum(label is None for label in labels)
    if supported >= required:
        return {"status": "CERTIFIED", "decision": "PASS", "n": len(labels), "required_support": required, "supported": supported, "failed": failed, "unknown": unknown}
    if supported + unknown < required:
        return {"status": "CERTIFIED", "decision": "FAIL", "n": len(labels), "required_support": required, "supported": supported, "failed": failed, "unknown": unknown}
    return {"status": "UNRESOLVED", "decision": None, "n": len(labels), "required_support": required, "supported": supported, "failed": failed, "unknown": unknown}


def main() -> None:
    run = json.loads((ROOT / "results/E0_FULL_32.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "TOPIC_SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    salt = selection["salt"]
    old_events = run["old_events"]
    new_events = run["new_cold_events"]
    warm_events = run["new_warm_events"]

    old_by_raw = {event["fact_digest"]: event for event in old_events}
    old_by_canonical: dict[str, dict] = {}
    old_rows: dict[str, dict] = {}
    new_rows: dict[str, dict] = {}
    data_dir = ROOT / "external/factscore_official/data/data/labeled"
    for name, target in [("InstructGPT.jsonl", old_rows), ("ChatGPT.jsonl", new_rows)]:
        with (data_dir / name).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gid = group_id(row["topic"], salt)
                if gid not in {item["group_id"] for item in selection["groups"]}:
                    continue
                target[gid] = row

    for event in old_events:
        old_by_canonical.setdefault(canonical(event.get("fact_text", "")), event)
    # Raw event records intentionally carry only a digest. Build canonical keys
    # from the frozen old fact list and pair them in order within each group.
    for gid, row in old_rows.items():
        facts = [fact.get("text", "") for annotation in (row.get("annotations") or []) for fact in ((annotation or {}).get("human-atomic-facts") or [])]
        events = [event for event in old_events if event["group_id"] == gid]
        for fact, event in zip(facts, events):
            old_by_canonical.setdefault(canonical(fact), event)

    group_new_events = {}
    for event in new_events:
        group_new_events.setdefault(event["group_id"], []).append(event)
    group_warm_events = {}
    for event in warm_events:
        group_warm_events.setdefault(event["group_id"], []).append(event)

    rows = []
    for gid in sorted(group_new_events):
        events = group_new_events[gid]
        warm = group_warm_events[gid]
        old_decision = run["old_decisions"].get(gid)
        new_decision = run["new_cold_decisions"].get(gid)
        # The full run does not store fact text, so canonical replay uses the
        # paired frozen fact lists and event order.
        new_row = new_rows.get(gid)
        new_facts = [fact.get("text", "") for annotation in ((new_row or {}).get("annotations") or []) for fact in ((annotation or {}).get("human-atomic-facts") or [])]
        canonical_labels = []
        canonical_input = canonical_output = 0
        for index, event in enumerate(events):
            fact = new_facts[index] if index < len(new_facts) else ""
            cached = old_by_canonical.get(canonical(fact))
            if cached is not None:
                canonical_labels.append(cached.get("label"))
            else:
                canonical_labels.append(event.get("label"))
                canonical_input += event.get("input_tokens", 0)
                canonical_output += event.get("output_tokens", 0)
        strict_labels = []
        strict_input = strict_output = 0
        strict_decision = None
        for event in events:
            strict_labels.append(event.get("label"))
            strict_input += event.get("input_tokens", 0)
            strict_output += event.get("output_tokens", 0)
            current = certificate(strict_labels)
            if current["status"] == "CERTIFIED":
                strict_decision = current
                break
        if strict_decision is None:
            strict_decision = certificate(strict_labels)
        canonical_strict_labels = []
        canonical_strict_input = canonical_strict_output = 0
        canonical_strict_decision = None
        for index, event in enumerate(events):
            fact = new_facts[index] if index < len(new_facts) else ""
            cached = old_by_canonical.get(canonical(fact))
            if cached is not None:
                canonical_strict_labels.append(cached.get("label"))
            else:
                canonical_strict_labels.append(event.get("label"))
                canonical_strict_input += event.get("input_tokens", 0)
                canonical_strict_output += event.get("output_tokens", 0)
            current = certificate(canonical_strict_labels)
            if current["status"] == "CERTIFIED":
                canonical_strict_decision = current
                break
        if canonical_strict_decision is None:
            canonical_strict_decision = certificate(canonical_strict_labels)
        rows.append({
            "group_id": gid,
            "old": old_decision,
            "new_cold": new_decision,
            "new_warm": run["new_warm_decisions"].get(gid),
            "stale_action": old_decision,
            "canonical_full": certificate(canonical_labels),
            "strict_gate_cold": strict_decision,
            "canonical_strict_gate": canonical_strict_decision,
            "costs": {
                "new_cold_input_tokens": sum(e.get("input_tokens", 0) for e in events),
                "new_warm_input_tokens": sum(e.get("input_tokens", 0) for e in warm),
                "canonical_full_input_tokens_replay": canonical_input,
                "strict_gate_cold_input_tokens_replay": strict_input,
                "canonical_strict_gate_input_tokens_replay": canonical_strict_input,
                "new_cold_output_tokens": sum(e.get("output_tokens", 0) for e in events),
                "new_warm_output_tokens": sum(e.get("output_tokens", 0) for e in warm),
                "canonical_full_output_tokens_replay": canonical_output,
                "strict_gate_cold_output_tokens_replay": strict_output,
                "canonical_strict_gate_output_tokens_replay": canonical_strict_output,
            },
        })
    result = {
        "schema": "mab2_e0_baseline_replay_v1",
        "status": "REPLAY_DEVELOPMENT_ONLY",
        "online_labels_or_costs_changed": False,
        "methods": ["stale", "new_cold", "new_warm", "canonical_full_replay", "strict_gate_cold_replay", "canonical_strict_gate_replay"],
        "rows": rows,
        "caveats": ["Replay uses completed full-run labels and costs after the online run", "No strong flat/RAG run", "No independent confirmation", "Do not interpret replay-only policies as deployed algorithms"],
    }
    (ROOT / "results/E0_BASELINE_REPLAY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "groups": len(rows)}))


if __name__ == "__main__":
    main()

