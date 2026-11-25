"""Run the first four-group E0 smoke after the official DB is available."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.e0_certificate import fixed_threshold_certificate
from src.factscore_e0_adapter import load_online_groups
from src.factscore_runtime import QwenFactVerifier, SQLiteTopicRetriever


def fact_key(topic: str, fact: str) -> str:
    return hashlib.sha256((topic + "\0" + fact).encode()).hexdigest()


def run_version(groups, retriever, verifier, history=None, cache_mode="cold"):
    history = history if history is not None else {}
    events = []
    decisions = {}
    for group in groups:
        labels = []
        for index, fact in enumerate(group.atomic_facts):
            key = fact_key(group.topic, fact)
            cache_hit = key in history and cache_mode in {"raw", "canonical"}
            if cache_hit:
                result = dict(history[key])
                result.update({"cache_hit": True, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0})
                evidence = []
            else:
                evidence = retriever.get_passages(group.topic, fact, k=5)
                result = verifier.verify(group.topic, fact, evidence)
                if cache_mode == "cold":
                    history[key] = dict(result)
            labels.append(result["label"])
            events.append({
                "group_id": group.group_id,
                "topic": group.topic,
                "fact_index": index,
                "fact_digest": hashlib.sha256(fact.encode()).hexdigest(),
                "cache_mode": cache_mode,
                "cache_hit": cache_hit,
                "evidence_ids": [item.evidence_id for item in evidence],
                **result,
            })
        decisions[group.group_id] = fixed_threshold_certificate(labels)
    return events, decisions, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selection = json.loads((ROOT / "TOPIC_SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    selected_ids = {item["group_id"] for item in selection["groups"][: args.limit]}
    data_dir = ROOT / "external/factscore_official/data/data/labeled"
    old = load_online_groups(data_dir / "InstructGPT.jsonl", selected_ids, selection["salt"], fact_source="human")
    new = load_online_groups(data_dir / "ChatGPT.jsonl", selected_ids, selection["salt"], fact_source="human")
    retriever = SQLiteTopicRetriever(Path(args.db))
    verifier = QwenFactVerifier(Path(args.model))
    started = time.time()
    old_events, old_decisions, history = run_version(old, retriever, verifier, cache_mode="cold")
    new_cold_events, new_cold_decisions, _ = run_version(new, retriever, verifier, cache_mode="cold")
    new_warm_events, new_warm_decisions, _ = run_version(new, retriever, verifier, history=history, cache_mode="raw")
    retriever.close()
    result = {
        "status": "PASS_E0_SMOKE_DEV",
        "groups": args.limit,
        "old_events": old_events,
        "new_cold_events": new_cold_events,
        "new_warm_events": new_warm_events,
        "old_decisions": {key: vars(value) for key, value in old_decisions.items()},
        "new_cold_decisions": {key: vars(value) for key, value in new_cold_decisions.items()},
        "new_warm_decisions": {key: vars(value) for key, value in new_warm_decisions.items()},
        "elapsed_seconds": time.time() - started,
        "model": str(args.model),
        "knowledge_source": str(args.db),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "groups": args.limit, "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

