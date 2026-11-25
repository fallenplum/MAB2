"""Create the E0 audit records without touching historical P4/P3 artifacts.

This script intentionally records an asset-gated stop.  It does not download
data, inspect sealed labels, start a model, or infer a substitute task.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(relative: str, payload: dict) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    e0_prompt = PROJECT / "MAB2_E0_Real_Workflow_Applicability_Prompt.md"
    p4_prompt = PROJECT / "MAB2_AutoResearch_P4_AntiRewardHacking_Prompt.md"
    agents = PROJECT / "AGENTS.md"
    p4_state = PROJECT / "experiments/p4_falsification_20261124/P4_STATE.json"
    source_runner = PROJECT / "experiments/p2_multihop_20261124/p2_service_runner.py"

    source_hashes = {
        "MAB2_E0_Real_Workflow_Applicability_Prompt.md": sha256(e0_prompt),
        "MAB2_AutoResearch_P4_AntiRewardHacking_Prompt.md": sha256(p4_prompt),
        "AGENTS.md": sha256(agents),
        "experiments/p4_falsification_20261124/P4_STATE.json": sha256(p4_state),
        "experiments/p2_multihop_20261124/p2_service_runner.py": sha256(source_runner),
        "external_reference": {
            "E0_prompt_sha256_from_authoritative_audit": "edf3b5ab5ba3536e6c437b108bd7d866062af3eb7e27e73a26efa3876ec7b91f",
            "P4_prompt_sha256_from_authoritative_audit": "bd9524b188331255ad81b279126612ced9063f20b222abd64b95d5dfaea7a7c2",
            "AGENTS_sha256_from_authoritative_audit": "a8da9db5027f05b6087aceef2912828dd3952f6a796e603f167d41faae7ed596",
        },
    }

    write_json(
        "E0_PROTOCOL_LOCK.json",
        {
            "schema": "mab2_e0_protocol_lock_v1",
            "locked_at_utc": now,
            "status": "LOCKED_BLOCKED_ASSET",
            "cycle": "E0-real-workflow-applicability-20261126",
            "repository": "fallenplum/MAB2",
            "branch_intent": "research/e0-real-workflow-applicability-20261126",
            "base_commit": "0918cb5892a803ce8832de2a84734f68a0425d54",
            "source_hashes": source_hashes,
            "frozen_scope": [
                "E0-A real workflow value",
                "E0-B raw/canonical/semantic interface stability",
                "E0-C propagation and joint recovery",
                "E0-D fixed-threshold certificate and legal baselines",
                "E0-E offline headroom diagnosis",
            ],
            "explicit_exclusions": [
                "E1/E2/DGBR/Horizon-2 search",
                "changing the P4 claim or historical P4 status",
                "using HotpotQA, QASC, SciQ, 2Wiki or historical M60 as an E0 replacement",
                "sealed labels, future outputs, or target leakage",
                "new GPU or paid API execution before a separate E0 budget is recorded",
            ],
            "primary_task_contract": {
                "task": "FActScore-style long-form atomic fact verification",
                "required_groups": 32,
                "calibration_groups": 16,
                "diagnostic_groups": 16,
                "threshold_service": {"threshold": 0.80, "weights": "unit", "decision": "PASS/FAIL/UNRESOLVED"},
            },
            "analysis_plan": "Do not choose an alternative task after observing asset availability or expected outcomes.",
            "resource_policy": {
                "new_gpu_hours": "UNAUTHORIZED_FOR_E0_AT_LOCK",
                "paid_api_cost": 0,
                "cpu_asset_audit": "allowed",
                "remote_read_only_asset_audit": "user_authorized_but_execution_environment_blocked",
            },
        },
    )

    write_json(
        "ASSET_INVENTORY.json",
        {
            "schema": "mab2_e0_asset_inventory_v1",
            "audited_at_utc": now,
            "status": "BLOCKED_PRIMARY_ASSET",
            "local_workspace": {
                "root": str(PROJECT),
                "factscore_data": {"status": "NOT_FOUND"},
                "factscore_official_code": {"status": "NOT_MATERIALIZED"},
                "historical_assets": {
                    "status": "FOUND_BUT_INELIGIBLE_FOR_E0_PRIMARY",
                    "categories": ["QASC", "SciQ", "Commonsense", "2Wiki", "HotpotQA", "M60/P3 traces"],
                    "reason": "E0 freezes FActScore as the primary task and forbids substituting old traces.",
                },
            },
            "authorized_e_drive_read_only": {
                "root": "E:\\图结构演化",
                "status": "READ_ONLY_INVENTORY_COMPLETE",
                "assets": [
                    {
                        "kind": "dataset",
                        "path": "E:\\图结构演化\\data\\raw\\qasper\\fdc9d8214fbab5dd782958601db4d678e6934a54",
                        "status": "AVAILABLE_EXCLUDED_PRIMARY_TASK",
                        "license": "CC BY 4.0",
                        "metadata_sha256": "c5a5b7abb6a0a91b2d9b173cdf1224d3d6e400b3d658d322acb67e12305dd5bf",
                        "reason": "Qasper is document QA, not FActScore-style atomic factuality.",
                    },
                    {
                        "kind": "dataset",
                        "path": "E:\\图结构演化\\data\\raw\\longbench_v2\\2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
                        "status": "AVAILABLE_EXCLUDED_PRIMARY_TASK",
                        "metadata_sha256": "9fdd1a3ebe86507253c124a18e9f78c898ce6341c12990af17ab868b8f600c35",
                        "reason": "No frozen atomic-fact verification contract in the current E0 scope.",
                    },
                    {
                        "kind": "dataset",
                        "path": "E:\\图结构演化\\data\\raw\\hotpotqa\\1908d6afbbead072334abe2965f91bd2709910ab",
                        "status": "AVAILABLE_EXCLUDED_BY_PROMPT",
                        "metadata_sha256": "1cc3729536a8738b0fcd8e33accc849f95b8c3313b323359eafe bd0d035db5f6".replace(" ", ""),
                        "reason": "The E0 prompt explicitly excludes HotpotQA as the primary task.",
                    },
                    {
                        "kind": "model",
                        "path": "E:\\图结构演化\\models\\Qwen2.5-1.5B-Instruct",
                        "status": "AVAILABLE_FOR_LATER_LOCK",
                        "config_sha256": "98d2ff8cc47488d08a2b0b3ac4feb99ef210779b42bd48605f6b8e36acdbf670",
                    },
                    {
                        "kind": "model",
                        "path": "E:\\图结构演化\\models\\Qwen2.5-3B-Instruct",
                        "status": "AVAILABLE_FOR_LATER_LOCK",
                        "config_sha256": "eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b",
                    },
                ],
            },
            "server": {
                "status": "NOT_VERIFIED",
                "reason": "The requested read-only SSH probe was denied by the current execution environment before a remote command ran.",
                "no_remote_write_or_gpu_started": True,
            },
            "decision": "No legal FActScore primary asset was verified. Do not silently switch to Qasper/LongBench/HotpotQA.",
        },
    )

    write_json(
        "DATA_MANIFEST.json",
        {
            "schema": "mab2_e0_data_manifest_v1",
            "status": "BLOCKED_ASSET",
            "primary_task": "FActScore-style atomic factuality",
            "groups_locked": 0,
            "group_selection": "NOT_STARTED",
            "reason": "No legal primary FActScore data and knowledge-source bundle is available in the inspected local assets.",
            "excluded_candidates": ["Qasper", "LongBench v2", "HotpotQA", "historical QASC/SciQ/2Wiki"],
            "sealed_data_read": False,
            "gold_read": False,
        },
    )

    write_json(
        "UPDATE_MANIFEST.json",
        {
            "schema": "mab2_e0_update_manifest_v1",
            "status": "NOT_LOCKED_BLOCKED_ASSET",
            "old_version": None,
            "new_version": None,
            "update_family": None,
            "reason": "E0 requires the update rule to be frozen before observing outputs, but the primary task asset is unavailable.",
        },
    )

    write_json(
        "WORKFLOW_GRAPH.json",
        {
            "schema": "mab2_e0_workflow_graph_v1",
            "status": "NOT_BUILT_BLOCKED_ASSET",
            "nodes": [],
            "edges": [],
            "reason": "The graph must be inferred from real execution traces. No eligible E0 task trace is available.",
        },
    )

    write_json(
        "EXECUTION_VERSIONS.json",
        {
            "schema": "mab2_e0_execution_versions_v1",
            "status": "ASSET_AUDIT_ONLY",
            "local_python": "not_locked",
            "observed_server_environment_record": {
                "python": "3.11.9",
                "cuda_runtime": "12.8",
                "torch": "2.9.1",
                "transformers": "4.54.0",
                "source": "E:\\图结构演化\\artifacts\\audit\\server_environment.json",
            },
            "models": ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct"],
            "selection": "not frozen because the primary task and evidence source are unavailable",
        },
    )

    write_json(
        "E0_BUDGET_LEDGER.json",
        {
            "schema": "mab2_e0_budget_ledger_v1",
            "status": "NO_E0_EXECUTION_STARTED",
            "new_gpu_hours": 0,
            "new_cpu_hours": 0,
            "paid_api_cost": 0,
            "download_cost": 0,
            "remote_model_calls": 0,
            "historical_p4_gpu_hours": "not reused as E0 authorization",
            "resource_authorization": "User authorized E-drive/server access, but no separate E0 compute budget was recorded in the current prompt.",
        },
    )

    issues = [
        {"id": "E0-001", "severity": "BLOCKER", "status": "OPEN", "issue": "FActScore primary data and knowledge-source bundle not found in inspected local assets."},
        {"id": "E0-002", "severity": "BLOCKER", "status": "OPEN", "issue": "Official FActScore code/data was not materialized in the workspace; no download was completed."},
        {"id": "E0-003", "severity": "MAJOR", "status": "OPEN", "issue": "Server read-only SSH probe was denied by the execution environment before any remote command ran."},
        {"id": "E0-004", "severity": "MAJOR", "status": "OPEN", "issue": "Qasper, LongBench v2 and HotpotQA are available but are not authorized substitutes for the frozen primary task."},
        {"id": "E0-005", "severity": "MAJOR", "status": "OPEN", "issue": "No E0-specific compute budget or model-call authorization has been recorded."},
        {"id": "E0-006", "severity": "INFO", "status": "CLOSED", "issue": "Fixed-threshold CPU certifier was implemented and mutation-tested without target or gold access."},
    ]
    issue_path = ROOT / "ISSUE_REGISTRY.jsonl"
    issue_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in issues), encoding="utf-8")

    write_json(
        "GATE_RECEIPTS.json",
        {
            "schema": "mab2_e0_gate_receipts_v1",
            "status": "INCONCLUSIVE_BLOCKED_ASSET",
            "gates": {
                "source_hash_lock": "PASS",
                "fixed_threshold_cpu_certifier": "PASS",
                "primary_asset": "BLOCKED_ASSET",
                "budget_and_compute": "BLOCKED_RESOURCE_AUTHORIZATION",
                "workflow_trace": "NOT_RUN",
                "F_baselines": "NOT_RUN",
                "D_baselines": "NOT_RUN",
                "headroom": "NOT_RUN",
            },
            "p4_status_preserved": "G0_INCONCLUSIVE_INVALIDATING_ISSUES",
            "prohibited_claims": ["PASS_E0_DEV", "G0_PASS", "RESEARCH_SUCCESS", "ACL readiness"],
        },
    )

    progress = [
        {"timestamp_utc": now, "event": "E0 prompt and P4 state read", "status": "PASS"},
        {"timestamp_utc": now, "event": "D-drive and E-drive asset inventory", "status": "PRIMARY_FACTSCORE_NOT_FOUND", "notes": "Qasper/LongBench/HotpotQA/model assets found but not substituted."},
        {"timestamp_utc": now, "event": "E0 fixed-threshold certificate CPU tests", "status": "PASS", "tests": "empty task, ternary exhaustive cases, boundary, illegal labels"},
        {"timestamp_utc": now, "event": "Server read-only probe", "status": "BLOCKED_EXECUTION_ENVIRONMENT", "gpu_started": False},
        {"timestamp_utc": now, "event": "E0 stage decision", "status": "INCONCLUSIVE_BLOCKED_ASSET", "next": "obtain or explicitly approve a legal primary asset/protocol"},
    ]
    progress_path = ROOT / "PROGRESS_LOG.jsonl"
    progress_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in progress), encoding="utf-8")

    results = {
        "E0_A_WORKFLOW_VALUE.json": {"status": "BLOCKED_ASSET", "n_groups": 0, "reason": "No eligible FActScore workflow."},
        "E0_B_INTERFACE_STABILITY.json": {"status": "NOT_RUN", "reason": "No eligible old/new execution pairs."},
        "E0_C_PROPAGATION_JOINT_RECOVERY.json": {"status": "NOT_RUN", "reason": "No eligible workflow graph."},
        "E0_D_BASELINE_CERTIFICATE.json": {"status": "NOT_RUN", "reason": "CPU certifier only; no task traces or baseline executions."},
        "E0_E_HEADROOM.json": {"status": "NOT_RUN", "reason": "No fixed DAG and cost ledger."},
        "COVERAGE_AND_COST_AUDIT.json": {"status": "INCONCLUSIVE_BLOCKED_ASSET", "expected_groups": 32, "completed_groups": 0, "new_gpu_hours": 0, "paid_api_cost": 0},
    }
    for name, payload in results.items():
        write_json("results/" + name, {"schema": "mab2_e0_result_v1", **payload})

    paired = ROOT / "results/PAIRED_GROUP_RESULTS.jsonl"
    paired.write_text("", encoding="utf-8")
    failures = ROOT / "results/FAILURE_CASES.jsonl"
    failures.write_text(json.dumps({"status": "NO_GROUP_EXECUTIONS", "reason": "blocked before sampling"}, ensure_ascii=False) + "\n", encoding="utf-8")

    write_json(
        "E0_DECISION.json",
        {
            "schema": "mab2_e0_decision_v1",
            "status": "BLOCKED_ASSET_INCONCLUSIVE",
            "conclusion": "No evidence about real workflow reuse or headroom can be claimed before a legal FActScore primary asset is available.",
            "recommendation": "Obtain the official FActScore human-annotated data and legal knowledge-source bundle, or obtain an explicit researcher-approved alternative protocol. Do not start E1 automatically.",
            "p4_state_unchanged": True,
        },
    )

    (ROOT / "E0_FINAL_REPORT_ZH.md").write_text(
        """# E0 真实工作流适用性与收益空间验证

## 1. 一句话结论

E0 暂不能进入真实工作流实验：已完成 D/E 盘资产清点和固定阈值 CPU 证书验证，但没有核验到规约要求的 FActScore 主任务资产，因此状态为 `INCONCLUSIVE_BLOCKED_ASSET`，不是正结果，也不是 E1 授权。

## 2. 规约、谱系与资源

本轮使用 E0 根 prompt 作为唯一规约，并保留 P4 的 `G0_INCONCLUSIVE_INVALIDATING_ISSUES`。E 盘发现 Qasper、LongBench v2、HotpotQA、Qwen2.5-1.5B/3B 等资产。Qasper 和 LongBench 不是当前冻结的 FActScore 原子事实核查合同，HotpotQA 被 E0 明确排除，因此没有替代主任务。服务器只读探测在当前执行环境被拒绝，未启动远端 GPU。

## 3. F 与 D 主表

没有合法的 32 个实体/文档组、旧/新版本和独立知识源，F/D 的完整重跑、缓存、规范化缓存、早停和 stale 对照均未运行。任何成本或质量数字都会混合不匹配任务，故不报告。

## 4. 接口、传播与证书

已实现独立的固定阈值证书 CPU 核心，区分 `INVALID`、`CERTIFIED`、`UNRESOLVED` 和 `FULL_RESOLVED`，对空任务、三态穷举、阈值边界和非法标签进行了测试。该测试只验证实现边界，不提供真实工作流证据。

## 5. 剩余收益空间与下一步

当前没有固定 DAG、真实旧/新轨迹或合法成本账本，不能计算强基线后的 headroom。下一步只能是获取官方 FActScore 人工标注/知识源资产，或由研究者明确批准替代协议。完成前不得启动 E1、DGBR 或新的算法搜索。

## 6. 证据分类

- `DEVELOPMENT`: 固定阈值证书 CPU 单元/穷举测试。
- `REPLAY`: 无。
- `BLOCKED`: FActScore 主任务资产、远端只读探测和 E0 新计算预算。
- `INVALID`: 无新的 E0 结果；历史 P4/P3 负结果保持原状态。
""",
        encoding="utf-8",
    )

    (ROOT / "REPRODUCE.md").write_text(
        """# E0 复现说明

当前可复现的只有资产锁和 CPU 证书测试。

```powershell
cd D:\\develop\\CV\\多臂老虎机
python experiments/e0_real_workflow_20261126/tests/test_e0_certificate.py
```

真实 F/D 流程尚未运行。不得将 E 盘 Qasper、LongBench 或 HotpotQA 直接替换为主任务，也不得读取封存标签来补齐当前缺失资产。
""",
        encoding="utf-8",
    )

    # Hash all generated files except the hash index itself.
    hashes = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_SHA256.json":
            hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
    write_json("ARTIFACT_SHA256.json", {"schema": "mab2_e0_artifact_sha256_v1", "generated_at_utc": now, "files": hashes})


if __name__ == "__main__":
    main()


