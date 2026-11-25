"""Run an actual early-stopping E0 gate policy on the locked groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.e0_certificate import fixed_threshold_certificate
from src.factscore_e0_adapter import load_online_groups
from src.factscore_runtime import QwenFactVerifier, SQLiteTopicRetriever


def digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def canonical(text: str) -> str:
    return " ".join(text.strip().lower().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["cold", "raw", "canonical"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selection = json.loads((ROOT / "TOPIC_SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    ids = {item["group_id"] for item in selection["groups"]}
    data_dir = ROOT / "external/factscore_official/data/data/labeled"
    old = load_online_groups(data_dir / "InstructGPT.jsonl", ids, selection["salt"], fact_source="human", allow_missing=True)
    new = load_online_groups(data_dir / "ChatGPT.jsonl", ids, selection["salt"], fact_source="human", allow_missing=True)
    old_run = json.loads((ROOT / "results/E0_FULL_32.json").read_text(encoding="utf-8"))
    old_events = old_run["old_events"]
    raw_cache = {(event["group_id"], event["fact_digest"]): event for event in old_events}
    canonical_cache = {}
    for group in old:
        events = [event for event in old_events if event["group_id"] == group.group_id]
        for fact, event in zip(group.atomic_facts, events):
            canonical_cache[(group.group_id, canonical(fact))] = event
    retriever = SQLiteTopicRetriever(Path(args.db))
    verifier = QwenFactVerifier(Path(args.model))
    rows = []
    started = time.time()
    for group in new:
        labels = []
        events = []
        if not group.atomic_facts:
            rows.append({"group_id": group.group_id, "status": "INVALID_NO_FROZEN_FACTS", "decision": None, "events": [], "input_tokens": 0, "output_tokens": 0})
            continue
        for index, fact in enumerate(group.atomic_facts):
            cached = None
            if args.method == "raw":
                cached = raw_cache.get((group.group_id, digest(fact)))
            elif args.method == "canonical":
                cached = canonical_cache.get((group.group_id, canonical(fact)))
            if cached is not None:
                result = {"label": cached.get("label"), "cache_hit": True, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0, "evidence_ids": []}
            else:
                evidence = retriever.get_passages(group.topic, fact, k=5)
                result = verifier.verify(group.topic, fact, evidence)
                result.update({"cache_hit": False, "evidence_ids": [item.evidence_id for item in evidence]})
            labels.append(result.get("label"))
            events.append({"group_id": group.group_id, "fact_index": index, "fact_digest": digest(fact), **result})
            decision = fixed_threshold_certificate(labels)
            if decision.status == "CERTIFIED":
                break
        decision = fixed_threshold_certificate(labels)
        rows.append({"group_id": group.group_id, "status": decision.status, "decision": decision.decision, "certificate": vars(decision), "events": events, "input_tokens": sum(event.get("input_tokens", 0) for event in events), "output_tokens": sum(event.get("output_tokens", 0) for event in events)})
    retriever.close()
    result = {"schema": "mab2_e0_strict_gate_run_v1", "status": "PASS_E0_STRICT_GATE_DEV", "method": args.method, "rows": rows, "elapsed_seconds": time.time() - started, "model": str(args.model), "knowledge_source": str(args.db)}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "method": args.method, "groups": len(rows), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

