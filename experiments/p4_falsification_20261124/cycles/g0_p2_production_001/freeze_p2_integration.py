from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
P4 = WORKSPACE / "experiments" / "p4_falsification_20261124"
P2 = WORKSPACE / "experiments" / "p2_multihop_20261124"
PROMPT = WORKSPACE / "MAB2_AutoResearch_P4_AntiRewardHacking_Prompt.md"
INPUT = P2 / "P2_MULTIHOP_INPUT.json"

CHECKPOINT = {
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
}
CYCLE_FILES = (
    "p4_core.py", "p4_online_runtime.py", "p4_policy_worker.py", "p4_p2_service_adapter.py",
    "audit_p4_call_ledger.py", "test_p4_zero_weight.py", "test_p4_p2_adapter.py",
    "test_p4_call_ledger_audit.py", "verify_p2_integration_lock.py", "remote_run_p2_adapter.py",
    "freeze_p2_integration.py", "run_p2_g0_local_gate.py", "test_p4_os_policy_sandbox.py",
    "test_p4_adversarial_boundary.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = json.loads(INPUT.read_text(encoding="utf-8"))
    row = rows[0]
    changed = row["changed_rows"]
    if not any(row["second_facts_new"][index] != row["second_fact"] or row["new_relay_interface"][index] for index in changed):
        raise SystemExit("locked development row has no real source edit")
    cycle_hashes = {name: sha(HERE / name) for name in CYCLE_FILES}
    p2_hashes = {name: sha(P2 / name) for name in ("p2_service_runner.py", "p2_logic.py")}
    payload = {
        "schema": "mab2_p4_p2_production_g0_lock_v1",
        "classification": "BOUNDED_G0_DEVELOPMENT_ONLY_NO_GOLD",
        "authoritative_prompt": "MAB2_AutoResearch_P4_AntiRewardHacking_Prompt.md",
        "authoritative_prompt_sha256": sha(PROMPT),
        "cycle_files_sha256": cycle_hashes,
        "p2_files_sha256": p2_hashes,
        "input_path": "experiments/p2_multihop_20261124/P2_MULTIHOP_INPUT.json",
        "input_sha256": sha(INPUT),
        "checkpoint_sha256": CHECKPOINT,
        "tokenizer_sha256": CHECKPOINT["tokenizer.json"],
        "row_index": 0,
        "changed_rows": changed,
        "real_edit_asserted_at_freeze": True,
        "physical_gpu": 4,
        "time_cap_seconds": 180,
        "repeat_count": 2,
        "formal_candidate_comparison": False,
        "gold_or_target_access": False,
        "budget_status": "P4_BUDGET_REMAINS_UNVERIFIED; this cap bounds one G0 probe only",
    }
    out = HERE / "P2_PRODUCTION_LOCK.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
