# E0-R1 复现入口

## 公共 CPU 检查

在仓库根目录执行：

```powershell
python -m unittest discover -s experiments/e0_r1_offline_repair_20261126/tests -v
python experiments/e0_r1_offline_repair_20261126/scripts/run_mutation_probes.py
```

这两步不需要 GPU、网络、模型权重或私有原题，检查固定 `n` 证书、三值标签边界、Fact/Document parser 隔离、随机期望数学和突变行为。

## 历史轨迹回放

完整 R1 builder 需要本地私有 E0 资产：

```powershell
python experiments/e0_r1_offline_repair_20261126/scripts/repair_trace_json.py
python experiments/e0_r1_offline_repair_20261126/scripts/build_r1_artifacts.py
```

`repair_trace_json.py` 只从 `experiments/e0_real_workflow_20261126/results/` 生成 `private_repaired/` 副本，不改写源 JSON；原题、答案、逐题证据和模型权重不进入公开提交。没有这些私有资产时，builder 会按缺失输入失败，不能用合成结果替代历史回放。

## 结果入口

- `R1_REPORT_ZH.md`：中文审计与结论。
- `results/THRESHOLD_SWEEP.json`：完整 n 阈值敏感性。
- `results/EXACT_RANDOM_EXPECTATION.json`：精确随机期望、100 seeds/组经验汇总和 clairvoyant 诊断。
- `results/FACT_DIGEST_MAPPING.json`：scorer-side occurrence 映射审计。
- `results/MUTATION_TESTS.json`：突变测试。
- `GATE_RECEIPTS.json`：R1 门状态和未闭合项。

所有成本均为历史轨迹中的 input+output token 加和；检索、计费、延迟和 GPU 时长字段缺失时保持 `UNKNOWN`，不进行推断。

