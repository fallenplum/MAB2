"""P4-0 read-only integration audit. This deliberately records failures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CPU = ROOT / "experiments" / "autoresearch_strict_graph_voi_20261124" / "cycles" / "strict_graph_voi_cpu.py"
H2 = ROOT / "experiments" / "autoresearch_strict_graph_voi_20261124" / "cycles" / "h2_fast.py"
SERVICE = ROOT / "experiments" / "p2_multihop_20261124" / "p2_service_runner.py"


def check(name: str, status: str, evidence: str, reason: str) -> dict:
    return {"test": name, "status": status, "evidence": evidence, "reason": reason}


def main() -> None:
    cpu = CPU.read_text(encoding="utf-8")
    h2 = H2.read_text(encoding="utf-8")
    service = SERVICE.read_text(encoding="utf-8")
    rows = [
        check("T02_second_parent_dirty", "FAIL", "p2_service_runner.py:255,292,329,355", "dirty cells use cell[0] in changed although terminal prompts consume both relay indices"),
        check("T03_no_target_fallback", "FAIL", "strict_graph_voi_cpu.py:295 and h2_fast.py:130", "candidate has target fallback when no legal option exists"),
        check("T07_independent_ledger", "FAIL", "p2_service_runner.py:run_row Ledger", "ledger is instantiated inside candidate runner and is not an independent broker source"),
        check("T10_clean_imports", "UNVERIFIED", "G3_CLEAN_REPLAY.md", "network clean checkout and independent environment were unavailable"),
        check("T13_deployable_prior", "FAIL", "strict_graph_voi_cpu.py:terminal_label/new_labels", "ideal generator and future labels are available to the CPU candidate"),
        check("T14_unresolved_status", "FAIL", "strict_graph_voi_cpu.py:295 and h2_fast.py:130", "unresolved paths return target instead of a non-success status"),
        check("T15_algorithm_identity", "FAIL", "h2_fast.py versus p2_service_runner.py", "CPU H2 and real service certificate_voi do not share a planner/certifier core"),
        check("T16_complete_independent_ledger", "UNVERIFIED", "P3 summaries and local ledgers", "raw per-case broker traces and independently generated call ledger are absent"),
        check("T01_real_input_digest", "UNVERIFIED", "p2_service_runner.py", "runner records tokenizer lengths but no independent parent-input digest event log"),
        check("T05_public_state_boundary", "FAIL", "p2_service_runner.py:run_row", "candidate route receives full row, model and target context in one process"),
    ]
    payload = {
        "schema": "mab2_p4_g0_readonly_audit_v1",
        "status": "G0_INCONCLUSIVE_INVALIDATING_ISSUES",
        "formal_experiments_started": False,
        "gpu_started": False,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (CPU, H2, SERVICE)},
        "tests": rows,
        "decision": "STOP_FORMAL_CANDIDATE_WORK_UNTIL_BROKER_CACHE_CLOSURE_TARGET_FALLBACK_AND_ALGORITHM_IDENTITY_ARE_FIXED",
    }
    out = Path(__file__).parent / "G0_AUDIT.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0)


if __name__ == "__main__":
    main()

