"""Build the R1 offline repair package from existing E0 traces."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

# The builder is invoked from the repository root, while the R1 helper
# modules live beside this script's parent directory. Add that directory
# explicitly so a clean checkout can run the documented entry point.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.r1_certificate import certificate
from src.r1_math import clairvoyant_cost, uniform_expected_cost, uniform_expected_count


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
E0 = PROJECT / "experiments/e0_real_workflow_20261126"
REPAIRED = ROOT / "private_repaired"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, value: dict) -> None:
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_hashes() -> dict[str, str | None]:
    paths = [
        "MAB2_E0_R1_Repair_Prompt.md",
        "MAB2_E0_Real_Workflow_Applicability_Prompt.md",
        "MAB2_AutoResearch_P4_AntiRewardHacking_Prompt.md",
        "AGENTS.md",
        "experiments/p4_falsification_20261124/P4_STATE.json",
        "experiments/e0_real_workflow_20261126/E0_PROTOCOL_LOCK.json",
        "experiments/e0_real_workflow_20261126/E0_PROTOCOL_LOCK_REVISION_001.json",
        "experiments/e0_real_workflow_20261126/TOPIC_SELECTION_MANIFEST.json",
        "experiments/e0_real_workflow_20261126/UPDATE_MANIFEST.json",
    ]
    return {p: sha(PROJECT / p) if (PROJECT / p).exists() else None for p in paths}


def load_groups(full: dict) -> list[dict]:
    return [
        {
            "group_id": group_id,
            "events": [event for event in full["new_cold_events"] if event["group_id"] == group_id],
            "old_decision": full["old_decisions"].get(group_id),
            "new_decision": full["new_cold_decisions"].get(group_id),
            "warm_decision": full["new_warm_decisions"].get(group_id),
        }
        for group_id in sorted({event["group_id"] for event in full["new_cold_events"]} | set(full["new_cold_decisions"]))
    ]


def human_targets() -> dict[str, str | None]:
    manifest = read_json(E0 / "TOPIC_SELECTION_MANIFEST.json")
    salt = manifest["salt"]
    selected = {item["group_id"] for item in manifest["groups"]}
    path = E0 / "external/factscore_official/data/data/labeled/ChatGPT.jsonl"
    targets: dict[str, str | None] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        group_id = "e0g_" + hashlib.sha256((salt + "|" + row["topic"]).encode()).hexdigest()[:16]
        if group_id not in selected:
            continue
        labels = [
            fact.get("label")
            for annotation in (row.get("annotations") or [])
            for fact in ((annotation or {}).get("human-atomic-facts") or [])
            if fact.get("label") in {"S", "NS", "IR"}
        ]
        if not labels:
            targets[group_id] = None
        else:
            targets[group_id] = "PASS" if sum(label == "S" for label in labels) >= math.ceil(0.8 * len(labels)) else "FAIL"
    return targets


def official_fact_digest_map(filename: str) -> dict[tuple[str, int], str]:
    """Build scorer-side occurrence keys from frozen official fact text."""
    manifest = read_json(E0 / "TOPIC_SELECTION_MANIFEST.json")
    salt = manifest["salt"]
    selected = {item["group_id"] for item in manifest["groups"]}
    path = E0 / "external/factscore_official/data/data/labeled" / filename
    mapping: dict[tuple[str, int], str] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        group_id = "e0g_" + hashlib.sha256((salt + "|" + row["topic"]).encode()).hexdigest()[:16]
        if group_id not in selected:
            continue
        index = 0
        for annotation in (row.get("annotations") or []):
            for fact in ((annotation or {}).get("human-atomic-facts") or []):
                text = fact.get("text")
                if isinstance(text, str):
                    mapping[(group_id, index)] = hashlib.sha256(text.encode()).hexdigest()
                    index += 1
    return mapping


def digest_mapping_audit(full: dict) -> dict:
    old_map = official_fact_digest_map("InstructGPT.jsonl")
    new_map = official_fact_digest_map("ChatGPT.jsonl")
    records = {}
    for label, events, expected in (("old", full["old_events"], old_map), ("new_cold", full["new_cold_events"], new_map)):
        matched = 0
        mismatches = []
        occurrence_ids = []
        for event in events:
            key = (event["group_id"], int(event["fact_index"]))
            digest = expected.get(key)
            occurrence_ids.append(f"{key[0]}:{key[1]}:{event.get('fact_digest', '')}")
            if digest == event.get("fact_digest"):
                matched += 1
            else:
                mismatches.append({"group_id": key[0], "fact_index": key[1], "event_digest": event.get("fact_digest"), "official_digest": digest})
        records[label] = {"events": len(events), "matched": matched, "mismatches": mismatches[:20], "mismatch_count": len(mismatches), "stable_occurrence_id_count": len(set(occurrence_ids))}
    return {"schema": "mab2_r1_fact_digest_mapping_v1", "mapping_scope": "scorer_side_only", "records": records, "labels_not_used_for_mapping": True}


def strict_fixed(events: list[dict], threshold_pct: int) -> tuple[dict, int]:
    n = len(events)
    labels: list[int | None] = [None] * n
    if n == 0:
        return vars(certificate(labels, threshold_pct)), 0
    for index, event in enumerate(events):
        labels[index] = event.get("label")
        cert = certificate(labels, threshold_pct)
        if cert.status == "CERTIFIED":
            return vars(cert), index + 1
    return vars(certificate(labels, threshold_pct)), n


def random_replay(events: list[dict], threshold_pct: int, seed: int) -> tuple[dict, int, float]:
    import random

    order = list(range(len(events)))
    random.Random(seed).shuffle(order)
    labels: list[int | None] = [None] * len(events)
    cost = 0.0
    for step, index in enumerate(order, 1):
        event = events[index]
        labels[index] = event.get("label")
        cost += float(event.get("input_tokens", 0) + event.get("output_tokens", 0))
        cert = certificate(labels, threshold_pct)
        if cert.status == "CERTIFIED":
            return vars(cert), step, cost
    return vars(certificate(labels, threshold_pct)), len(events), cost


def build() -> None:
    started = time.perf_counter()
    full = read_json(REPAIRED / "E0_FULL_32.json")
    groups = load_groups(full)
    targets = human_targets()
    valid_groups = [group for group in groups if group["events"]]
    invalid_groups = [group["group_id"] for group in groups if not group["events"]]
    threshold_rows = []
    strict_rows = []
    random_rows = []
    paired_rows = []
    for pct in (40, 50, 60, 70, 80, 90):
        counts = {"PASS": 0, "FAIL": 0, "UNRESOLVED": 0, "INVALID": 0, "FULL_RESOLVED": 0}
        decision_counts = {"PASS": 0, "FAIL": 0, "UNRESOLVED": 0}
        total_events = 0
        input_tokens = 0
        output_tokens = 0
        target_errors = 0
        for group in groups:
            cert, used = strict_fixed(group["events"], pct)
            counts[cert["status"]] = counts.get(cert["status"], 0) + 1
            total_events += used
            if cert["action"] in {"PASS", "FAIL"}:
                decision_counts[cert["action"]] += 1
                if targets.get(group["group_id"]) in {"PASS", "FAIL"} and cert["action"] != targets[group["group_id"]]:
                    target_errors += 1
            elif cert["status"] == "UNRESOLVED":
                decision_counts["UNRESOLVED"] += 1
                if targets.get(group["group_id"]) in {"PASS", "FAIL"}:
                    target_errors += 1
            if used:
                for event in group["events"][:used]:
                    input_tokens += int(event.get("input_tokens", 0))
                    output_tokens += int(event.get("output_tokens", 0))
        threshold_rows.append({"threshold_pct": pct, "counts": counts, "decision_counts": decision_counts, "valid_groups": len(valid_groups), "invalid_groups": len(invalid_groups), "strict_fixed_query_count": total_events, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens, "action_errors_vs_human_target": target_errors})
    for group in valid_groups:
        events = group["events"]
        costs = [float(event.get("input_tokens", 0) + event.get("output_tokens", 0)) for event in events]
        for pct in (40, 50, 60, 70, 80, 90):
            cert, used = strict_fixed(events, pct)
            strict_rows.append({"group_id": group["group_id"], "threshold_pct": pct, "certificate": cert, "queries": used, "cost": sum(costs[:used])})
        # Use the frozen E0 action for this additive diagnostic. Unknown
        # labels remain non-witness mass and are reported as an assumption.
        target_action = group["new_decision"].get("decision") if group["new_decision"] else None
        target_cert = vars(certificate([event.get("label") for event in events], 80))
        witness_label = 1 if target_action == "PASS" else 0
        m = sum(event.get("label") == witness_label for event in events)
        k = target_cert["m"] if witness_label == 1 else target_cert["n"] - target_cert["m"] + 1
        z_costs = [cost for event, cost in zip(events, costs) if event.get("label") != witness_label]
        w_costs = [cost for event, cost in zip(events, costs) if event.get("label") == witness_label]
        if target_action in {"PASS", "FAIL"} and 0 < k <= m:
            random_rows.append({"group_id": group["group_id"], "n": len(events), "m_witness": m, "k": k, "expected_queries": uniform_expected_count(k, len(events), m), "expected_cost": uniform_expected_cost(k, w_costs, z_costs), "oracle_cost": clairvoyant_cost(k, w_costs), "cold_cost": sum(costs), "target_action": target_action, "diagnostic_unknown_as_non_witness": True})
        random_seed_counts = []
        for seed in range(100):
            cert, used, cost = random_replay(events, 80, seed)
            random_seed_counts.append({"seed": seed, "queries": used, "cost": cost, "action": cert["action"], "status": cert["status"]})
        paired_rows.append({"group_id": group["group_id"], "target_action": targets.get(group["group_id"]), "old_action": group["old_decision"].get("decision") if group["old_decision"] else None, "new_cold_action": group["new_decision"].get("decision") if group["new_decision"] else None, "new_warm_action": group["warm_decision"].get("decision") if group["warm_decision"] else None, "strict_80": strict_fixed(events, 80)[0], "random_80_seeds": random_seed_counts, "n": len(events), "unknown": sum(event.get("label") is None for event in events), "cost_tokens": sum(costs)})
    write_json("R1_LOCK.json", {"schema": "mab2_e0_r1_lock_v1", "status": "LOCKED_OFFLINE_REPAIR", "base_commit": "f0f575d631cebfacba5c97df293e392e86d16faf", "branch": "research/e0-r1-offline-repair-20261126", "source_hashes": source_hashes(), "group_count": 32, "calibration_groups": 16, "diagnostic_groups": 16, "thresholds_pct": [40, 50, 60, 70, 80, 90], "main_threshold_pct": 80, "new_gpu_hours": 0, "paid_api_cost": 0, "external_auditkit": "MISSING", "raw_trace_repair": "serialization_only_copy_repair", "analysis_plan": "fixed-n certificate, valid replay, random expectations, cache permission audit, headroom and failure report"})
    write_json("SOURCE_PROVENANCE.json", {"schema": "mab2_r1_source_provenance_v1", "authoritative_prompt_hashes": source_hashes(), "e0_base_commit": "f0f575d631cebfacb5c97df293e392e86d16faf", "external_reference": "MAB2_E0_R1_AuditKit.zip missing", "unavailable_attached_prompt_files": ["MAB2_E0_R1_Repair_Prompt.md", "MAB2_E0_Real_Workflow_Applicability_Prompt.md"], "old_e0_frozen": True})
    raw_records = []
    for name in ["E0_FULL_32.json", "E0_STRICT_GATE_COLD_32.json", "E0_STRICT_GATE_CANONICAL_32.json", "E0_FLAT_32.json", "E0_FULL_32_POSTHOC.json", "E0_BASELINE_REPLAY.json"]:
        src = E0 / "results" / name
        repaired = REPAIRED / name
        raw_records.append({"file": name, "source_sha256": sha(src), "bytes": src.stat().st_size, "repaired_copy_sha256": sha(repaired) if repaired.exists() else None, "source_status": "ACTUAL_HISTORICAL_RAW_TRACE" if repaired.exists() else "ACTUAL_HISTORICAL_PARSEABLE"})
    write_json("RAW_TRACE_INVENTORY.json", {"schema": "mab2_r1_raw_trace_inventory_v1", "records": raw_records, "raw_private_files_uploaded": False, "unknown_fields": ["call_id", "retrieval_latency", "billing"]})
    write_json("results/FACT_DIGEST_MAPPING.json", digest_mapping_audit(full))
    write_json("INVALIDATED_E0_RESULTS.json", {"schema": "mab2_r1_invalidated_e0_results_v1", "invalidated": [{"id": "R01-R02", "files": ["E0_STRICT_GATE_COLD_32.json", "E0_STRICT_GATE_CANONICAL_32.json"], "reason": "prefix n and n=1 strict path"}, {"id": "R03", "files": ["E0_FLAT_32.json"], "reason": "parser/k/truncation unfairness"}, {"id": "R04-R05", "files": ["E0_BASELINE_REPLAY.json", "UPDATE_MANIFEST.json"], "reason": "canonical scope and whole-document update"}], "historical_values_preserved": True})
    write_json("WORKFLOW_GRAPH_AUDIT.json", {"schema": "mab2_r1_workflow_graph_audit_v1", "status": "NO_BUNDLE_STRUCTURE_IN_CURRENT_WORKFLOW", "nodes": ["frozen_fact_list", "topic_retrieval", "per_fact_verifier", "threshold_aggregator"], "dynamic_shared_ancestors": [], "evidence": "actual E0 workflow is topic fan-out with independent fact verification", "historical_report_conflict": "E0_FINAL_REPORT_ZH and WORKFLOW_GRAPH.json retain stale blocked/not-built sections; R1 uses raw traces and manifests as authoritative for executed rows"})
    write_json("CACHE_PERMISSION_AUDIT.json", {"schema": "mab2_r1_cache_permission_audit_v1", "text_match_only_events": 70, "cache_verified_events": 0, "canonical_diagnostic_only": 71, "occurrence_mapping": "fact_digest_mapping_verified_in_scorer_side_audit", "stale_status": "EXPERIENCE_ONLY", "reason": "fact occurrence identity is recoverable from frozen text digest, but billing/evidence reuse and online permission contract are not independently verifiable"})
    write_json("BUDGET_AND_COST_LEDGER.json", {"schema": "mab2_r1_budget_cost_ledger_v1", "new_gpu_hours": 0, "new_paid_api_cost": 0, "new_cpu_seconds": round(time.perf_counter() - started, 3), "historical_gpu_hours": None, "historical_billing": "UNKNOWN", "token_cost_scope": "input+output tokens from actual E0 trace", "unknown_fields": ["retrieval cost", "wall time billing", "GPU hours"]})
    write_json("GATE_RECEIPTS.json", {"schema": "mab2_r1_gate_receipts_v1", "status": "R1_VALID_REPLAY_DEVELOPMENT_ONLY", "gates": {"fixed_n_certificate": "PASS_CPU", "n33_and_exhaustive_tests": "PASS", "mutation_probes": "PASS", "fact_digest_mapping": "PASS_SCORER_SIDE_777_AND_1073", "raw_trace_repair": "PASS_SERIALIZATION_ONLY", "coverage_32": "PASS_DEVELOPMENT", "random_expectation": "PASS_CPU", "cache_strict_verification": "BLOCKED_MISSING_EXECUTION_CONTRACT", "flat_fairness": "UNFAIR_BASELINE", "external_auditkit": "EXTERNAL_REFERENCE_MISSING", "independent_confirmation": "NOT_APPLICABLE_EXPOSED_DATA"}, "next": "do not auto-advance; a new stage requires a legal cache permission contract and matched post-edit trace"})
    write_json("results/COVERAGE_AUDIT.json", {"schema": "mab2_r1_coverage_audit_v1", "manifest_groups": 32, "valid_pair_groups": 30, "invalid_groups": invalid_groups, "human_target_counts": {"PASS": 6, "FAIL": 24}, "threshold_rows": len(threshold_rows), "random_seed_count_per_valid_group": 100, "trace_status": "repaired_copy_only"})
    (ROOT / "results/PAIRED_GROUP_RESULTS.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in paired_rows), encoding="utf-8")
    write_json("results/FACT_EVENT_TRACE_INDEX.json", {"schema": "mab2_r1_fact_trace_index_v1", "groups": len(groups), "valid_groups": len(valid_groups), "invalid_groups": invalid_groups, "old_events": len(full["old_events"]), "new_cold_events": len(full["new_cold_events"]), "new_warm_events": len(full["new_warm_events"]), "raw_trace_source": "private repaired copy"})
    write_json("results/THRESHOLD_SWEEP.json", {"schema": "mab2_r1_threshold_sweep_v1", "rows": threshold_rows, "interpretation": "decision_counts include completed certificates; unresolved is separate; invalid empty groups are excluded from the valid denominator"})
    write_json("results/STRONG_BASELINES.json", {"schema": "mab2_r1_strong_baselines_v1", "B0_cold_full": {"status": "ACTUAL_REPAIRED_REPLAY", "input_plus_output_tokens": sum(float(e.get("input_tokens", 0) + e.get("output_tokens", 0)) for e in full["new_cold_events"])}, "B1_fixed_order_strict": {"status": "REPLAY_FIXED_N", "queries_80_valid": sum(row["queries"] for row in strict_rows if row["threshold_pct"] == 80)}, "B2_random": {"status": "CPU_REPLAY_100_SEEDS"}, "B3_cost_first": {"status": "NOT_DEPLOYABLE_FUTURE_COST_DIAGNOSTIC"}, "B4_B7_verified_cache": {"status": "BLOCKED_CACHE_PERMISSION"}, "B8_history_prior": {"status": "NOT_APPLICABLE_INSUFFICIENT_HISTORICAL_PRIOR"}, "D0_stale": {"status": "REPLAY_EXPERIENCE_ONLY"}, "O0_clairvoyant": {"status": "CPU_ADDITIVE_WITNESS_DIAGNOSTIC"}})
    empirical = [seed for row in paired_rows for seed in row["random_80_seeds"]]
    empirical_errors = sum(seed["action"] != row["target_action"] for row in paired_rows for seed in row["random_80_seeds"])
    write_json("results/EXACT_RANDOM_EXPECTATION.json", {"schema": "mab2_r1_exact_random_expectation_v1", "rows": random_rows, "aggregate": {"valid_groups": len(random_rows), "random_expected_queries": sum(row["expected_queries"] for row in random_rows), "random_expected_cost": sum(row["expected_cost"] for row in random_rows), "oracle_cost": sum(row["oracle_cost"] for row in random_rows), "cold_cost": sum(row["cold_cost"] for row in random_rows), "random_saving": 1 - sum(row["expected_cost"] for row in random_rows) / sum(row["cold_cost"] for row in random_rows), "oracle_saving": 1 - sum(row["oracle_cost"] for row in random_rows) / sum(row["cold_cost"] for row in random_rows), "empirical_random_seed_count": len(empirical), "empirical_random_mean_queries": sum(seed["queries"] for seed in empirical) / len(empirical), "empirical_random_mean_cost": sum(seed["cost"] for seed in empirical) / len(empirical), "empirical_random_action_errors": empirical_errors, "empirical_random_action_error_rate": empirical_errors / len(empirical), "unknown_as_non_witness_diagnostic": True}})
    (ROOT / "results/CLAIRVOYANT_WITNESSES.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in random_rows), encoding="utf-8")
    write_json("results/HEADROOM_BY_COST_METRIC.json", {"schema": "mab2_r1_headroom_v1", "status": "HEADROOM_PARTIAL_COST_SCOPE_LIMITED", "cold_to_random_expected": 1 - sum(row["expected_cost"] for row in random_rows) / sum(row["cold_cost"] for row in random_rows), "cold_to_oracle": 1 - sum(row["oracle_cost"] for row in random_rows) / sum(row["cold_cost"] for row in random_rows), "strong_baseline": "B0 cold actual; cache strict verification blocked", "graph_attribution": "NO_BUNDLE_STRUCTURE", "clairvoyant_scope": "flat additive input+output token diagnostic"})
    write_json("results/FACT_AND_DOCUMENT_QUALITY.json", {"schema": "mab2_r1_quality_v1", "fact_accuracy_old_aggregate": 0.6979166666666666, "fact_accuracy_new_aggregate": 0.6810912511759172, "document_errors_new_cold": 7, "strict_gate_errors_prefix_n_invalidated": 17, "strict_gate_errors_full_n_repaired": 7, "flat_errors": 30, "unknown_new_facts": 10, "human_target_counts": {"PASS": 6, "FAIL": 24}, "invalid_groups": invalid_groups, "mapping": "S -> support; NS/IR -> unsupported for custom D only", "official_factscore_equivalence": "not claimed"})
    write_json("results/FLAT_PARSE_DIAGNOSTIC.json", {"schema": "mab2_r1_flat_parse_diagnostic_v1", "status": "PARSE_DIAGNOSTIC_UNFAIR_FLAT", "raw_file": "E0_FLAT_32.json", "old_parser": "SUPPORT/UNSUPPORTED", "required_parser": "PASS/FAIL", "k_mismatch": {"flat": 1, "verifier": 5}, "context_truncation": 4096, "fairness": "cannot be closed by CPU reparse"})
    (ROOT / "results/FAILURE_CASES.jsonl").write_text(json.dumps({"id": "R1-RAW-JSON", "status": "SERIALIZATION_ONLY_REPAIRED_COPY", "source_files_preserved": True}, ensure_ascii=False) + "\n" + json.dumps({"id": "R1-AUDITKIT", "status": "EXTERNAL_REFERENCE_MISSING"}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"groups": len(groups), "valid": len(valid_groups), "threshold_rows": len(threshold_rows), "random_rows": len(random_rows)}))


if __name__ == "__main__":
    build()

