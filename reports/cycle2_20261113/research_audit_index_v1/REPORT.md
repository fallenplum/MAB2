# 持续科研审计索引

这是运行中索引，不是项目终止或最终验收。旧 D2/D3 停止结论保留；用户授权的新周期继续探索。当前第四轮已真实启动，评分仍封存。

| 阶段 | 原始证据 | 有边界的判定 |
|---|---|---|
| G1/G2 original | [G1_STAGE_REPORT_20261104.md](../../../results/G1_STAGE_REPORT_20261104.md) | Nonadditivity is not a valid predictor or online policy guarantee |
| G2 original | [G2_STAGE_REPORT_20261105.md](../../../results/G2_STAGE_REPORT_20261105.md) | Frozen C2 prediction gate failed; no threshold changes |
| D2 | [D2_REPORT.md](../../../reports/autonomous_20261113/d2/D2_REPORT.md) | Intercept penalty induces baseline credit; C1_R1 action gain zero; no substitute winner |
| D3 | [D3_REPORT.md](../../../reports/autonomous_20261113/d3/D3_REPORT.md) | Graph-only order2 drift enrichment failed; candidate stop preserved |
| Format/sparsity | [REPORT.md](../../../reports/cycle2_20261113/g2_format_diagnostic/REPORT.md) | Clean outputs still have sparse observed credit; metric noise offset cannot reverse model ranking |
| Group edits | [REPORT.md](../../../reports/cycle2_20261113/group_intervention_diagnostic/REPORT.md) | Selected near-zero singleton effects do not establish harmful joint deletion |
| Staged paid feedback | [REPORT.md](../../../reports/cycle2_20261113/staged_feedback_screen/REPORT.md) | Complete base cost eliminates proposed screening advantage in tested class |
| Formal1 and R2 | [README.md](../../../reports/cycle2_20261113/r1_completed_development/README.md) | change/k2 fails point gate; independent-bank drift variance/cost gate fails |
| R3 CPU cache route | [README.md](../../../reports/cycle2_20261113/r3_completed_screen/README.md) | Bundled purchase route fails; structural calls are not GPU or service utility |
| Formal2 | [README.md](../../../reports/cycle2_20261113/hotpot_formal2_results_v1/README.md) | Current B worse than independent aggregation; sparse drift support insufficient |
| Formal3 | [ROOT_DECISION.json](../../../reports/cycle2_20261113/path_formal3_results_v1/ROOT_DECISION.json) | Controlled path D filter passes; system gain unsupported |
| Formal3 posthoc I/C/S | [REPORT.md](../../../reports/cycle2_20261113/path_posthoc_policy_v1/REPORT.md) | Targeted refresh equals zero-new-feedback protection; exposed development only |
| Formal4 contract | [SCIENTIFIC_DESIGN_FROZEN_V1.json](../../../protocol/cycle2_receiver_factorial_20261115/SCIENTIFIC_DESIGN_FROZEN_V1.json) | Outcome still sealed at checkpoint; fixed receiver/access nominee and gates |
| Formal4 qualification | [ENGINEERING_ACCEPTANCE.json](../../../reports/cycle2_20261113/receiver_factorial_qualification/ENGINEERING_ACCEPTANCE.json) | Actual regeneration and CPU replay establish engineering only |

## G2 失败的已验证部分与未解部分

原始 C2 风险高于 B0；D2 的常数响应控制定位了全系数正则化产生伪信用的机制。这属于原目标性质，不是旧实现偏离旧合同。取消截距惩罚改善风险，却没有让预定提名产生正动作收益。格式正常的子集仍稀疏，因而不能归结为格式错误；同时，观察信用为零不等于真实条件均值为零。跨种子风险与五种子均值 MSE 的差是同单元、同权重下的模型无关噪声项，不足以解释模型排名失败。

仍未建立：跨问题信用方向可预测性、完整成本下反馈净价值、恢复收益、跨任务泛化和正方法独立复现。没有用更大预测器掩盖这些缺口。

## 成本与资料完整性

COST_CHECKPOINT.json 将已闭合驻留小时、尚未发生的预约额度与历史未知费用分开。历史 G1/G2 有 GPU kernel、token 和调用账，但缺少完整驻留账及部分迁移在途成本；未知保留为 null，不伪造整个历史项目精确总额。

当前分支原先未继承另一历史分支的 R1/R2/R3 公开报告。现已从 7762ffd / acd4c27 按原字节补齐，恢复记录提供逐文件哈希；没有覆盖当前控制状态或修改旧报告。这是交付完整性修复，不是重跑实验。

## 验收边界

数值审计、真实回放与独立源码重算能支持计算对应性，不能冒充独立人工同行评审。有效正方法尚未成立。正式第五轮尚未提名或准入；第四轮结果不会自动使其他单元成为替补胜者。若后续达到项目级停止条件，交付受证据约束的负结果稿；否则继续自主研究。
