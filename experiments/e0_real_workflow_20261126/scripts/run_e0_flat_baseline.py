"""Run a matched group-level flat RAG baseline for E0."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.factscore_e0_adapter import load_online_groups
from src.factscore_runtime import SQLiteTopicRetriever, parse_verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selection = json.loads((ROOT / "TOPIC_SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    ids = {item["group_id"] for item in selection["groups"]}
    data_dir = ROOT / "external/factscore_official/data/data/labeled"
    groups = load_online_groups(data_dir / "ChatGPT.jsonl", ids, selection["salt"], fact_source="human", allow_missing=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    retriever = SQLiteTopicRetriever(Path(args.db))
    rows = []
    started = time.time()
    for group in groups:
        if not group.atomic_facts:
            rows.append({"group_id": group.group_id, "status": "INVALID_NO_FROZEN_FACTS", "decision": None, "input_tokens": 0, "output_tokens": 0, "evidence_ids": []})
            continue
        evidence_ids = []
        evidence_lines = []
        for index, fact in enumerate(group.atomic_facts):
            evidence = retriever.get_passages(group.topic, fact, k=1)
            evidence_ids.extend(item.evidence_id for item in evidence)
            evidence_lines.append(f"Fact {index+1}: {fact}\nEvidence: {evidence[0].text if evidence else ''}")
        prompt = (
            "You are a flat RAG quality evaluator. Decide whether at least 80 percent of the listed atomic facts "
            "are supported by their evidence. Reply exactly PASS or FAIL.\n\nTopic: " + group.topic + "\n\n" + "\n\n".join(evidence_lines) + "\n\nDecision:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        elapsed = time.perf_counter() - start
        generated = output[0, inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip().upper()
        parsed = parse_verdict(raw)
        rows.append({"group_id": group.group_id, "status": "CERTIFIED" if parsed is not None else "UNRESOLVED", "decision": "PASS" if parsed == 1 else "FAIL" if parsed == 0 else None, "raw_output": raw, "input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": int(generated.shape[0]), "latency_seconds": elapsed, "evidence_ids": evidence_ids})
    retriever.close()
    result = {"schema": "mab2_e0_flat_baseline_v1", "status": "PASS_E0_FLAT_DEV", "method": "group-level-flat-RAG", "rows": rows, "elapsed_seconds": time.time()-started, "model": args.model, "knowledge_source": args.db}
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({"status":result["status"],"groups":len(rows),"elapsed_seconds":result["elapsed_seconds"]}))


if __name__ == '__main__':
    main()

