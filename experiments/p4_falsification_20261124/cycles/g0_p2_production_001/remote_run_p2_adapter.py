from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/tmp/mab2_p4_p2_production_001")
UPLOAD = ROOT / "upload"
CYCLE = ROOT / "experiments" / "p4_falsification_20261124" / "cycles" / "g0_p2_production_001"
P2 = ROOT / "experiments" / "p2_multihop_20261124"
PROMPT = ROOT / "MAB2_AutoResearch_P4_AntiRewardHacking_Prompt.md"
MODEL = Path("/home/chen/KDBPT/KD/models/Qwen-0.5B")
RESULTS = CYCLE / "results"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in args.run_id):
        raise ValueError("run id must contain only letters, digits, underscore, and hyphen")
    start = time.perf_counter()
    CYCLE.mkdir(parents=True, exist_ok=True)
    P2.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock_path = UPLOAD / "P2_PRODUCTION_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result_path = RESULTS / f"P4_P2_PRODUCTION_ADAPTER_{args.run_id}.json"
    audit_path = RESULTS / f"CALL_LEDGER_AUDIT_{args.run_id}.json"
    receipt_path = RESULTS / f"RUNNER_RECEIPT_{args.run_id}.json"
    if any(path.exists() for path in (result_path, audit_path, receipt_path)):
        raise FileExistsError("run-id result artifact already exists; refusing to overwrite")
    for name in lock["cycle_files_sha256"]:
        shutil.copy2(UPLOAD / name, CYCLE / name)
    shutil.copy2(lock_path, CYCLE / lock_path.name)
    for name in lock["p2_files_sha256"]:
        shutil.copy2(UPLOAD / name, P2 / name)
    shutil.copy2(UPLOAD / "P2_MULTIHOP_INPUT.json", P2 / "P2_MULTIHOP_INPUT.json")
    shutil.copy2(UPLOAD / PROMPT.name, PROMPT)

    env = os.environ.copy()
    gpu_id = int(lock["physical_gpu"])
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"
    verify_command = [
        sys.executable, "verify_p2_integration_lock.py",
        "--lock", str(CYCLE / lock_path.name), "--cycle", str(CYCLE), "--p2", str(P2),
        "--prompt", str(PROMPT), "--input", str(P2 / "P2_MULTIHOP_INPUT.json"), "--model", str(MODEL),
    ]
    verify = subprocess.run(verify_command, cwd=CYCLE, env=env, capture_output=True, text=True, check=False)
    (RESULTS / f"LOCK_VERIFY_{args.run_id}.stdout.txt").write_text(verify.stdout, encoding="utf-8")
    (RESULTS / f"LOCK_VERIFY_{args.run_id}.stderr.txt").write_text(verify.stderr, encoding="utf-8")
    if verify.returncode != 0:
        receipt = {"status": "FAIL_LOCK_BEFORE_MODEL_LOAD", "verify_stdout": verify.stdout, "verify_stderr": verify.stderr, "gpu_forwards": 0}
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return verify.returncode

    test_command = [sys.executable, "-m", "unittest", "-v", "test_p4_zero_weight", "test_p4_p2_adapter", "test_p4_call_ledger_audit", "test_p4_os_policy_sandbox", "test_p4_adversarial_boundary"]
    tests = subprocess.run(test_command, cwd=CYCLE, env=env, capture_output=True, text=True, check=False)
    (RESULTS / f"PRE_GPU_TESTS_{args.run_id}.stdout.txt").write_text(tests.stdout, encoding="utf-8")
    (RESULTS / f"PRE_GPU_TESTS_{args.run_id}.stderr.txt").write_text(tests.stderr, encoding="utf-8")
    if tests.returncode != 0:
        receipt = {"status": "FAIL_PRE_GPU_TESTS", "test_exit_code": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr, "gpu_forwards": 0}
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return tests.returncode

    gpu_text = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    fields = [int(part.strip()) for part in gpu_text.split(",")]
    preflight = {"physical_gpu": fields[0], "utilization_percent": fields[1], "memory_used_mib": fields[2], "memory_total_mib": fields[3]}
    if preflight["physical_gpu"] != gpu_id or preflight["utilization_percent"] > 5 or preflight["memory_used_mib"] > 1024:
        receipt = {"status": "STOP_GPU_NOT_IDLE", "gpu_preflight": preflight, "gpu_forwards": 0}
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 3

    run_command = [
        sys.executable, "p4_p2_service_adapter.py", "--model", str(MODEL),
        "--input", str(P2 / "P2_MULTIHOP_INPUT.json"), "--lock", str(CYCLE / lock_path.name),
        "--output", str(result_path),
        "--worker", str(CYCLE / "p4_policy_worker.py"), "--row", str(lock["row_index"]),
        "--repeat-count", str(lock["repeat_count"]), "--gpu", str(gpu_id), "--cap-seconds", str(lock["time_cap_seconds"]),
    ]
    run = subprocess.run(run_command, cwd=CYCLE, env=env, capture_output=True, text=True, check=False)
    audit_code = 99
    audit_stdout = ""
    audit_stderr = ""
    if result_path.is_file():
        audit_command = [sys.executable, "audit_p4_call_ledger.py", "--input", str(result_path), "--output", str(audit_path)]
        audit = subprocess.run(audit_command, cwd=CYCLE, env=env, capture_output=True, text=True, check=False)
        audit_code, audit_stdout, audit_stderr = audit.returncode, audit.stdout, audit.stderr
        (RESULTS / f"CALL_LEDGER_AUDIT_{args.run_id}.stdout.txt").write_text(audit_stdout, encoding="utf-8")
        (RESULTS / f"CALL_LEDGER_AUDIT_{args.run_id}.stderr.txt").write_text(audit_stderr, encoding="utf-8")
    receipt = {
        "status": "PASS_BOUNDED_G0_DEVELOPMENT_ONLY" if run.returncode == 0 and audit_code == 0 else "FAIL_BOUNDED_G0_DEVELOPMENT_ONLY",
        "gpu_preflight": preflight,
        "adapter_exit_code": run.returncode,
        "independent_audit_exit_code": audit_code,
        "adapter_stdout": run.stdout,
        "adapter_stderr": run.stderr,
        "audit_stdout": audit_stdout,
        "audit_stderr": audit_stderr,
        "wall_seconds_including_setup": time.perf_counter() - start,
        "formal_candidate_comparison": False,
        "gold_or_target_access": False,
        "remaining_scope": "one exposed row only; not G0 closure or service-quality evidence",
    }
    (RESULTS / f"ADAPTER_{args.run_id}.stdout.txt").write_text(run.stdout, encoding="utf-8")
    (RESULTS / f"ADAPTER_{args.run_id}.stderr.txt").write_text(run.stderr, encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
