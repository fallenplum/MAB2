# D36 后的证据与新颖性矩阵

| 方向 | 近邻已有工作 | 本项目可检验的增量 | 本项目实测结果 |
| --- | --- | --- | --- |
| 图结构稀疏化 | AgentPrune 学习节省 token 的通信图；AgentDropout 动态去除节点。 | 在同一已部署图发生变化后，用有限预算判定何时值得复检旧边信用。 | D34 的免费结构规则相对最强固定身份仅 +0.00005 效用，无法归因为可转移的结构保护效果。[AgentPrune](https://arxiv.org/abs/2410.02506) · [AgentDropout](https://aclanthology.org/2025.acl-long.1170/) |
| 自洽和汇合 | ICML 对辩论、集成和自洽比较资源与准确率；Smoothie 在无标签路由中估计模型质量。 | 在相同初始化、答复和反馈预算下比较当前分歧与历史图变是否带来增量。 | 六个真实模型中，D33 的图变化权重与去掉该项选择相同服务结果，完整效用低于最强廉价基线。[讨论系统对照](https://proceedings.mlr.press/v235/smit24a.html) · [Smoothie](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e6b57a990462df5afa58d64ce2709db9-Abstract-Conference.html) |
| 回报变化与决策价值 | 成本归一的期望信息价值、逆概率修正和差值估计均为已有工具。 | 利用完全相同请求可复用的原图记录，只购买新增干预并在不同服务题上检验净收益。 | D35 的按冲击上界有空间。D36 支付复检后六个同预算比较中五负一平。没有达到独立确认条件。[本项目预注册](../../protocol/cycle3_20261116/D36_PAID_PANEL.md) |
| 任务时间与图变化 | SituatedQA 为相同问题提供不同时间答案，TimeQA 对文章证据与变化事实作对齐。 | 各自固定提问时间与图结构作交叉干预，避免把任务变化误记为边信用。 | D32 有 9 篇当前信用改变，但最多纠正 1 篇。需要读者区分标注证据位置、世界时间和文档版本。[SituatedQA](https://aclanthology.org/2021.emnlp-main.586/) · [TimeQA](https://arxiv.org/abs/2108.06314) |

该矩阵补充既有文献树。它支持有边界的实验问题，不能据此认定优先权、达到 SOTA 或外部算法已完成同预算复现。作者侧写作与检查使用 [Scientific Agent Skills](https://arxiv.org/abs/2609.00065)。
