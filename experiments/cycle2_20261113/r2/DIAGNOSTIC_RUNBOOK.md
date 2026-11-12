# R2 I/C 诊断运行说明

已冻结合同：`FROZEN_DIAGNOSTIC_CONTRACT.json`，SHA256 `3e85f15ff4c653f533d7f4483000c3fcf8c8c2f11013619d18d3d191f3e2ff02`。原 draft 保留；程序拒绝合同字节变化。

当前状态：31 项诊断合成检查、18 项先前代数合成检查通过。未由本子任务读取真实 R1 I/C 或 S；尚无真实 R2 诊断结论，也未启动 R2 GPU。

## 输入与执行

R1 adapter 在服务动作冻结后、首次 S 标签访问前导出：

- `private/R2_IC_INPUTS.npz`：严格 16 个预注册键，只含 I/C 数值、token/call 成本、哈希标识、seed 与图结构。
- `private/R2_IC_RECEIPT.json`：绑定协议、执行器、模型、实际分析提交/源码、I/C 源文件哈希清单、动作冻结和 NPZ。
- `private/ACTIONS_FROZEN.json`：作为导出前的不可变信息边界，并逐项核对 I 增益哈希及所有 C 已购买反馈的增益哈希。

正式执行由主任务运行：

```powershell
python .\r2_ic_diagnostic.py --private-dir <R1正式分析private目录> --output <新的R2诊断JSON>
```

不需要 R1 `COMPLETE.json` 或 R1 public summary，也不读取 S 文件、原题、gold、模型文本和模型权重。程序不执行外部命令或模型推理。若只做 adapter 人工 fixture 集成，加 `--synthetic-fixture`；该选项只接受显式 `SYNTHETIC_FIXTURE_ONLY` 的导出身份，输出同时标为纯合成。正式模式拒绝此类身份。

验证：

```powershell
python .\test_r2_ic_diagnostic.py
```

## 结果解释

`ELIGIBLE_FOR_G4_RESOURCE_SCREEN` 只表示 G0–G3 的固定研发投入门通过；主任务还必须完成 G4 的真实吞吐、覆盖和剩余预算验收。程序没有 GPU 启动权限，也不把候选失败解释为项目终止。

`INCONCLUSIVE_LOW_SUPPORT`、`INCONCLUSIVE_CURRENT_CANDIDATE` 或 `NO_GO_CURRENT_BANK_AND_QUOTA_OR_DECISION_VALUE` 都应保留，不能改阈值、换 m/n、择优子组或用通用 CV 的成绩替代主候选过门。风险分母过小及 bootstrap 异常复制全部保留和计数。

成本结果只称“初始化+反馈的实测token成本”。S24 共同上限加和是敏感性分析，未来服务的实际 token/GPU 成本尚未观测。全部结论仍是 R1 I/C 开发数据复用；后续独立 pilot 和独立确认另有要求。

## 审查边界

本程序独立校验导出 hash、协议参数、seed/题目不重复、成本结构、I/C 增益与已冻结动作证据的绑定。它不重新打开原始图文件、不重新执行评分器，因此绝对 C 回答质量和原始模型执行正确性仍依赖已验收的 adapter/原档审计。哈希一致性不等于科学可重复性或预测效果。

bootstrap 共 8 个 episode、每折仅 6 题；区间没有覆盖率或显著性保证。通用 CV 系数的估计口径、零方差处理、二折行序、模型大小和所有投入阈值在真实数值读取前已冻结。
