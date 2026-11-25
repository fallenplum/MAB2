from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
P4 = WORKSPACE / "experiments" / "p4_falsification_20261124"
P2 = WORKSPACE / "experiments" / "p2_multihop_20261124"
PROMPT = WORKSPACE / "MAB2_AutoResearch_P4_AntiRewardHacking_Prompt.md"
RESULTS = HERE / "results"
OLD_CYCLE = P4 / "cycles" / "g0_online_runtime_001"
CORE = P4 / "cycles" / "g0_broker_refactor"
KIT = P4 / "audit_kit_original" / "MAB2_P4_Research_Gate"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    env_python = Path(sys.executable)
    checks = [
        ("new_p2_adapter_zero_mass_policy_sandbox_and_adversarial_mutations", [str(env_python), "-m", "unittest", "-v", "test_p4_zero_weight", "test_p4_p2_adapter", "test_p4_call_ledger_audit", "test_p4_os_policy_sandbox", "test_p4_adversarial_boundary"], HERE),
        ("prior_online_runtime_tests", [str(env_python), "-m", "unittest", "-v", "test_p4_online_runtime.py"], OLD_CYCLE),
        ("prior_broker_mutations_and_exhaustive_certificates", [str(env_python), "-m", "unittest", "-v", "test_g0_core", "test_g0_mutations", "test_certificate_exhaustive", "test_public_state_future_output_mutation", "test_public_gate_replay"], CORE),
        ("researcher_public_kit", [str(env_python), "-m", "unittest", "-v", "test_verify_trace"], KIT),
        ("researcher_public_example", [str(env_python), "verify_trace.py", "--fixtures", "example_private_fixture.json", "--traces", "example_trace.json"], KIT),
    ]
    rows = []
    for name, command, cwd in checks:
        t0 = time.perf_counter()
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", check=False)
        out_path = RESULTS / f"{name}.stdout.txt"
        err_path = RESULTS / f"{name}.stderr.txt"
        out_path.write_text(result.stdout, encoding="utf-8")
        err_path.write_text(result.stderr, encoding="utf-8")
        rows.append({
            "name": name, "command": command, "cwd": str(cwd), "exit_code": result.returncode,
            "wall_seconds": time.perf_counter() - t0,
            "stdout_sha256": sha(out_path), "stderr_sha256": sha(err_path),
        })
    lock_verify = subprocess.run([
        str(env_python), str(HERE / "verify_p2_integration_lock.py"),
        "--lock", str(HERE / "P2_PRODUCTION_LOCK.json"), "--cycle", str(HERE), "--p2", str(P2),
        "--prompt", str(PROMPT), "--input", str(P2 / "P2_MULTIHOP_INPUT.json"),
    ], capture_output=True, text=True, encoding="utf-8", check=False)
    (RESULTS / "lock_verify.stdout.txt").write_text(lock_verify.stdout, encoding="utf-8")
    (RESULTS / "lock_verify.stderr.txt").write_text(lock_verify.stderr, encoding="utf-8")
    all_ok = lock_verify.returncode == 0 and all(item["exit_code"] == 0 for item in rows)
    receipt = {
        "schema": "mab2_p4_p2_g0_local_gate_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "DEVELOPMENT_ONLY",
        "status": "PASS_DEVELOPMENT_ONLY" if all_ok else "FAIL_DEVELOPMENT_ONLY",
        "python": sys.version,
        "workspace_lock_status": "PASS" if lock_verify.returncode == 0 else "FAIL",
        "checks": rows,
        "candidate_comparisons_started": False,
        "formal_candidate_experiments_started": False,
        "g0_closed": False,
        "remaining_blockers": [
            "actual P2 runner is covered by one exposed-row development integration probe only",
            "OS-enforced policy/scorer isolation and external blind signoff remain absent",
            "the complete production T01-T16 matrix, including T06/T09/T10/T12/T15/T16, is still open",
            "P4 aggregate GPU/CPU and money budgets remain unverified",
        ],
    }
    (RESULTS / "LOCAL_GATE_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
