# 条件 relay 漂移诊断合同

## 目的

测试 source-only certificate transport 在 relay conditional channel 发生变化时是否会复用失效的旧 row 观测，并测试按受影响 relay 行重验证的 TCR 机制是否能在同一反馈预算下恢复正确决策。

## 固定构造

使用 M60 四个脱敏面板的模型侧 vectors 和旧图 transcript，仅作机制诊断，不作为新的独立任务确认。新图 source 使用冻结的 `new_source`。对 relay 行使用固定变换

`cond_new[i] = (1-alpha) * cond_old[i] + alpha * base`

其中 `alpha` 取 0.25 和 0.50，受影响行集合分别为 `{A}` 和 `{A,B,C,D}`。终端标签保持冻结，以隔离 conditional channel 漂移。每个条件的 target 使用 `cond_new` 重新计算。

## 对照

- stale，完全复用旧图 decision。
- source-only CTDR，复用全部旧 row 和 cell 观测，只使用 source 漂移半径。
- TCR-row，知道受影响行集合，重查这些 row 的新 conditional distribution，复用其合法 terminal labels，再用 Audit.next。
- TCR-ranking，TCR-row 初始化后按决策区域宽度除以真实 token 价格排序。
- cold restart，新图全部重新审计。

## 预注册指标

action error、exact decision-set、false drop、false keep、post-edit token cost、相对 cold restart 的 token saving、按 alpha 和受影响行集合分解。不得使用 gold 选择行集合或调参。若 source-only CTDR 在受影响行集合下错误而 TCR-row 在至少一个 alpha 和两个面板上修复，进入真实条件通道模型推理。若 TCR 不优于 cold 或只在单一面板有效，冻结该候选并转向机制负结果。

该诊断不改变 M60 正式结果，也不打开 sealed test。
