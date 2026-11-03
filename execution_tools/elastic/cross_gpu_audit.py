"""Non-empty deterministic equivalence audit; never reads reference labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def signature(value):
    return {"question_id": value["question_id"], "question_fingerprint": value["question_fingerprint"],
            "mask": value["mask"], "seed": value["seed"], "protocol_hash": value["protocol_hash"],
            "answer": value["answer"],
            "nodes": {key: {"messages": node["messages"], "seed": node["seed"],
                             "message_metadata": node["message_metadata"],
                             "generation": {field: node["generation"][field]
                                            for field in ("text", "input_tokens", "output_tokens", "finish_reason")}}
                      for key, node in value["nodes"].items()}}


def executions(folder):
    return {path.name: signature(read(path)) for path in (folder / "qualification_executions").glob("*.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/pilot/s0_v3_20261103"
    references = executions(base)
    if len(references) != 10 or sum(len(x["nodes"]) for x in references.values()) != 110:
        raise ValueError("Missing or unexpected reference executions; empty equality is forbidden")
    checks = []
    for gpu in (1, 2, 4, 7):
        folder = root / "results/pilot" / f"elastic_qualification_gpu{gpu}_20261103"
        gate = read(folder / "QUALIFICATION.json") if (folder / "QUALIFICATION.json").exists() else {}
        actual = executions(folder)
        calls = len((folder / "physical_calls.jsonl").read_bytes().splitlines()) if (folder / "physical_calls.jsonl").exists() else 0
        checks.append({"gpu": gpu, "gate": gate.get("status"), "eligible": gate.get("eligible_for_G1"),
                       "calls": calls, "execution_files": len(actual), "reference_files": len(references),
                       "node_occurrences": sum(len(x["nodes"]) for x in actual.values()),
                       "exact_signature_equal": len(actual) == 10 and actual == references})
    passed = all(item["gate"] == "PASS" and item["eligible"] is True and item["calls"] == 96
                 and item["exact_signature_equal"] for item in checks)
    value = {"schema": "elastic_cross_gpu_qualification_v1", "status": "PASS" if passed else "PENDING_OR_FAILED",
             "observed_unix": time.time(), "labels_read": False, "checks": checks,
             "qualified_gpus": [0, 5, 6] + [item["gpu"] for item in checks if item["gate"] == "PASS" and item["exact_signature_equal"]],
             "reference_protocol_hash": read(base / "run_manifest.json")["protocol_hash"],
             "auditor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "boundary": "Equality on 10 engineering executions per new GPU, not a universal numerical proof or scientific result"}
    (root / "artifacts/audit/elastic_cross_gpu_qualification.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
