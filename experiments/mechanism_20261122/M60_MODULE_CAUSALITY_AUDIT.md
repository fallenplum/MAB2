# M60 模块因果归因审计

正式四面板模块消融共 2304 个条件实例。Warm Restart、Reweighted Audit 和 CTDR without Radius 都达到 exact set 1.0，平均逻辑 token 为 98.2057。Full CTDR 同样达到 exact set 1.0，但平均 token 为 99.2448。Decision Ranking 的平均 token 也为 99.2448。

因此，当前正式面板没有识别出漂移半径或决策排序带来的独立决策收益或成本收益。相对于 cold restart 的节省主要来自历史观测复用与新图重加权审计。`module_causal_necessity` 标记为 `unproven_negative_ablation`。

该结果保留为负消融，不支持把 Full CTDR 宣称为独立优于 Warm Restart 的算法。后续若继续研究，应使用新的独立任务质量协议和真正的可学习复检策略。

