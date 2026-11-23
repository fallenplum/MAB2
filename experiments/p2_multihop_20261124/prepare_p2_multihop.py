"""Prepare a small, auditable QASC multi-hop service panel.

The script keeps answer keys out of the public manifest. It converts the
existing QASC two-fact records into an explicit source -> relay -> terminal
DAG and preserves both the raw evidence strings and a structured relay
interface. The resulting JSON is suitable for later model-forward runners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("MAB2_recovered/qasc_rows2_INPUT.json")
DEFAULT_OUTPUT = Path("experiments/p2_multihop_20261124/P2_MULTIHOP_INPUT.json")
DEFAULT_MANIFEST = Path("experiments/p2_multihop_20261124/P2_MULTIHOP_MANIFEST.json")


def _stable_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows
        if len(row.get("options", [])) == 8
        and row.get("first_fact")
        and row.get("second_fact")
        and (
            (isinstance(row.get("second_facts_new"), list) and len(row["second_facts_new"]) == 4)
            or (isinstance(row.get("new_relay_interface"), list) and len(row["new_relay_interface"]) == 4)
        )
    ]
    eligible.sort(key=lambda row: hashlib.sha256(row["key"].encode()).hexdigest())
    return eligible[:limit]


def _interface(row: dict[str, Any], branch: int, changed: bool) -> str:
    payload = {
        "schema": "mab2_multihop_interface_v1",
        "source_node": "S0",
        "relay_node": f"R{branch}",
        "terminal_node": "T",
        "depends_on": ["S0", f"R{branch}"],
        "hypothesis_slot": chr(65 + branch),
        "evidence_key": row["key"],
        "changed": changed,
        "return_format": "one hypothesis token A-D",
    }
    return "STRUCTURED_INTERFACE " + json.dumps(payload, sort_keys=True)


def build_row(row: dict[str, Any]) -> dict[str, Any]:
    new_facts = row.get("second_facts_new")
    if not isinstance(new_facts, list) or len(new_facts) != 4:
        new_facts = [row["second_fact"]] * 4
    base_interfaces = row.get("new_relay_interface")
    if not isinstance(base_interfaces, list) or len(base_interfaces) != 4:
        base_interfaces = [""] * 4
    changed_rows = [
        index for index, value in enumerate(new_facts)
        if value != row["second_fact"]
    ]
    changed_rows = sorted(set(changed_rows) | {index for index, value in enumerate(base_interfaces) if value})
    interfaces = [
        (base_interfaces[branch] or _interface(row, branch, True))
        if branch in changed_rows else ""
        for branch in range(4)
    ]
    source_terms = set(re.findall(r"[A-Za-z]{4,}", row["first_fact"].lower()))
    relay_terms = set(re.findall(r"[A-Za-z]{4,}", row["second_fact"].lower()))
    overlap = sorted(source_terms & relay_terms)
    return {
        "key": row["key"],
        "stem": row["stem"],
        "options": row["options"],
        "first_fact": row["first_fact"],
        "second_fact": row["second_fact"],
        "second_fact_new": row.get("second_fact_new", ""),
        "second_facts_new": new_facts,
        "control_fact": row.get("control_fact", ""),
        "evidence_linkage": {
            "source_relay_lexical_overlap": overlap,
            "overlap_is_diagnostic_only": True,
        },
        "raw_evidence": {
            "source_fact": row["first_fact"],
            "relay_fact_old": row["second_fact"],
            "relay_fact_new": new_facts,
            "control_fact": row.get("control_fact", ""),
        },
        "structured_interface": {
            "schema": "mab2_multihop_dag_v1",
            "nodes": ["S0", "R0", "R1", "R2", "R3", "T"],
            "edges": [
                ["S0", "R0"], ["S0", "R1"], ["S0", "R2"], ["S0", "R3"],
                ["R0", "R1"], ["R0", "R2"], ["R0", "R3"],
                ["R0", "T"], ["R1", "T"], ["R2", "T"], ["R3", "T"],
            ],
            "source_inputs": ["stem", "options", "raw_evidence.source_fact"],
            "relay_inputs": ["S0.output", "raw_evidence.relay_fact_new[i]"],
            "terminal_inputs": ["R0.output", "R1.output", "R2.output", "R3.output", "options"],
            "changed_relay_rows": changed_rows,
        },
        "new_relay_interface": interfaces,
        "changed_rows": changed_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    selected = _stable_rows(rows, args.limit)
    if len(selected) < args.limit:
        raise ValueError(f"only {len(selected)} eligible rows, requested {args.limit}")
    prepared = [build_row(row) for row in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prepared, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema": "mab2_p2_multihop_manifest_v1",
        "input_source": str(args.input),
        "input_rows": len(prepared),
        "selection": "sha256(key) ascending, first 32 eligible QASC rows",
        "split": {"development": 16, "held_out": 16},
        "model_path_server": "/home/chen/KDBPT/KD/models/Qwen-0.5B",
        "gold_path_private": "MAB2_recovered/interface_drift_rows1/qasc/SEALED.json",
        "gold_in_public_input": False,
        "dag": "S0 -> R0 -> {R1,R2,R3} -> T, with direct S0 edges retained",
        "methods": [
            "full_new_graph",
            "old_graph_reuse",
            "single_model",
            "retrieval_augmented_generation",
            "exact_cache",
            "graph_mask_recompute",
            "topology_voi_online",
            "full_information_voi_upper_bound",
        ],
        "online_information_rule": "methods may read old transcript, source edit and public graph manifest only; sealed labels are scoring-only",
        "upper_bound_rule": "full_information_voi_upper_bound is offline analysis and cannot be reported as an online method",
        "cost_fields": ["forward_calls", "input_tokens", "output_tokens", "wall_seconds", "cache_hits", "retrieval_calls"],
        "quality_fields": ["answer_accuracy", "answer_action_exact", "decision_exact", "false_drop", "false_keep", "recovery_rate"],
        "status": "prepared_only_pending_P1_gain_gate",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rows": len(prepared), "development": 16, "held_out": 16, "output": str(args.output), "manifest": str(args.manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
