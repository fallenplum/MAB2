"""Aggregate P2 service shards into a redacted public summary."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean

METHODS = (
    "full_new_graph",
    "old_graph_reuse",
    "single_model",
    "retrieval_augmented_generation",
    "exact_cache",
    "graph_mask_recompute",
    "topology_voi_online",
    "full_information_voi_upper_bound",
)


def aggregate(paths: list[Path], split: str) -> dict:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    out = {"split": split, "n": len(rows), "methods": {}}
    for method in METHODS:
        records = [row["methods"][method] for row in rows]
        metric_keys = ("answer_accuracy", "answer_action_exact", "decision_exact")
        cost_keys = ("forward_calls", "input_tokens", "output_tokens", "wall_seconds", "cache_hits", "retrieval_calls")
        item = {key: mean(record["metrics"][key] for record in records) for key in metric_keys}
        item.update({key: mean(record["cost"][key] for record in records) for key in cost_keys})
        item["offline_only"] = bool(any(record.get("offline_only", False) for record in records))
        if method == "topology_voi_online":
            item["mean_queried_terminal_cells"] = mean(record.get("queried_terminal_cells", 0) for record in records)
            item["threshold"] = records[0].get("threshold")
        out["methods"][method] = item
    history = [row["history_cost"] for row in rows]
    out["old_transcript_setup"] = {
        key: mean(item[key] for item in history)
        for key in ("forward_calls", "input_tokens", "output_tokens", "wall_seconds")
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", nargs="+", type=Path, required=True)
    parser.add_argument("--heldout", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema": "mab2_p2_service_public_v1",
        "model": "/home/chen/KDBPT/KD/models/Qwen-0.5B",
        "microbatch": 1,
        "online_gold_access": False,
        "methods": list(METHODS),
        "splits": [aggregate(args.dev, "development"), aggregate(args.heldout, "held_out")],
        "private_shards": {"development": [path.name for path in args.dev], "held_out": [path.name for path in args.heldout]},
        "note": "No question keys, targets or per-row predictions are included. Full-information VOI is offline-only.",
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "dev": 16, "held_out": 16}))


if __name__ == "__main__":
    main()
