# M60 公平模块消融（审计周期）

所有方法复用相同的旧图历史 transcript，成本只计图编辑后的新增查询 token。Gold 只用于报告，不参与调度。

| 方法 | action error | exact set | queries | tokens | safe rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm_restart | 0.0000 | 1.0000 | 0.9453 | 98.21 | 0.0000 |
| reweighted_audit | 0.0000 | 1.0000 | 0.9453 | 98.21 | 0.0000 |
| ctdr_without_radius | 0.0000 | 1.0000 | 0.9453 | 98.21 | 0.0000 |
| ctdr_legacy | 0.0000 | 1.0000 | 0.9453 | 98.21 | 0.4783 |
| warm_restart_token | 0.0000 | 1.0000 | 0.9497 | 96.63 | 0.0000 |
| decision_ranking | 0.0000 | 1.0000 | 0.9991 | 99.24 | 0.0000 |
| full_ctdr | 0.0000 | 1.0000 | 0.9991 | 99.24 | 0.4787 |

## 解释

Warm Restart、Reweighted Audit 和 CTDR without Radius 的相同结果显示，当前 80% 左右的冷重启节省主要来自历史观测复用与新来源重加权。CTDR legacy 的 radius gate 只有在与 Audit.decision 不同的情况下才有独立停止价值。Decision Ranking 则单独测量实际 token 价格与决策区域排序。Full CTDR 只有在同时优于这些对照时，才可声称有独立算法收益。

## 成本口径

旧 transcript 的历史成本不计入 post-edit 增量成本。Stale 的 post-edit 增量为 0。full restart 的新增成本单独报告。
