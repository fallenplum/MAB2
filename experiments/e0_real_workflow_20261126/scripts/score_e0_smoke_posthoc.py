"""Post-hoc scorer for E0 smoke outputs.

This script is intentionally separate from the online runner. It is the only
component that reads human atomic-fact labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def labels_by_group(path: Path, group_ids: set[str], salt: str) -> dict[str, dict[str, list[str]]]:
    output: dict[str, dict[str, list[str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            group_id = "e0g_" + hashlib.sha256((salt + "|" + row["topic"]).encode()).hexdigest()[:16]
            if group_id not in group_ids:
                continue
            mapping: dict[str, list[str]] = {}
            for annotation in row.get("annotations") or []:
                for fact in annotation.get("human-atomic-facts", []):
                    text = fact.get("text")
                    label = fact.get("label")
                    if isinstance(text, str) and label in {"S", "NS", "IR"}:
                        mapping.setdefault(digest(text), []).append(label)
            output[group_id] = mapping
    return output


def score_events(events: list[dict], labels: dict[str, dict[str, list[str]]]) -> dict[str, object]:
    scored = []
    ambiguous = 0
    for event in events:
        options = labels.get(event["group_id"], {}).get(event["fact_digest"], [])
        if len(set(options)) != 1:
            ambiguous += 1
            continue
        target = 1 if options[0] == "S" else 0
        predicted = event.get("label")
        scored.append({"group_id": event["group_id"], "predicted": predicted, "target": target, "target_label": options[0]})
    comparable = [item for item in scored if item["predicted"] in {0, 1}]
    correct = sum(item["predicted"] == item["target"] for item in comparable)
    return {
        "comparable_facts": len(comparable),
        "ambiguous_or_missing_targets": ambiguous + len(scored) - len(comparable),
        "accuracy": correct / len(comparable) if comparable else None,
        "support_targets": sum(item["target"] == 1 for item in comparable),
        "unsupported_targets": sum(item["target"] == 0 for item in comparable),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--source-old", required=True)
    parser.add_argument("--source-new", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    ids = {item["group_id"] for item in selection["groups"][: run["groups"]]}
    old_labels = labels_by_group(Path(args.source_old), ids, selection["salt"])
    new_labels = labels_by_group(Path(args.source_new), ids, selection["salt"])
    result = {
        "schema": "mab2_e0_smoke_posthoc_v1",
        "online_run_status": run["status"],
        "labels_read_posthoc_only": True,
        "old": score_events(run["old_events"], old_labels),
        "new_cold": score_events(run["new_cold_events"], new_labels),
        "new_warm": score_events(run["new_warm_events"], new_labels),
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

