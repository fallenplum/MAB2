# 最终文献树与新颖性结论

来源树、逐篇定位与先前假设保留：[原始 Literature Tree](../../autonomous_20261113/LITERATURE_TREE.md)、[原始 Novelty Matrix](../../autonomous_20261113/NOVELTY_MATRIX.md)、[第五轮前合并树](../consolidated_20261115/TREE_AND_NOVELTY.md)、[12 条书目记录](../consolidated_20261115/BIBLIOGRAPHIC_LEDGER.json)、[经典 KG 来源](../decision_value_foundation_20261115/REVIEW.md)。原 23 篇与增量有重叠，不能相加当唯一篇数；定向综述未宣称穷尽或完成所有外部算法复现。

| 分支 | 核对的一手近邻 | 本项目与其区别 | 最终证据与新颖性处置 |
|---|---|---|---|
| 拓扑生成、节点与边剪枝 | AgentPrune、AgentDropout、G-Designer、AGP | 外生图变化后复检既有边信用 | 不以剪枝/省节点为创新 |
| 传播、证据访问和压缩 | Information Propagation、Information Bottleneck、iAgents、Reliability Limits | 显式控制原始证据访问与单模型基线 | 第3/4轮机制存在但质量不足；不将理想 Bayes 支配定理套到有限 LLM |
| 结构搜索与归因 | CATS、E2、SafeSieve、Graph-GRPO（阅读层次见原树） | 固定预测器与图编辑后反馈选择 | 条件信用并非全新概念；新机制优势未建立 |
| 主动采集与净决策价值 | Active Graph Learning v3、DIVA、CABAI、Frazier et al. 2008 KG | 以 paid cache 与执行闭包估计边复检价格 | EVSI/成本归一化/缓存均已有基础；第5轮四门失败 |
| 证据审计 | 本项目 I/C/S、免费对照、真实已测成本 | 将信用变化、决策效用、成本明确拆开检验 | 可提供可复算的限定负结果，不宣称方法 SOTA |

经典 KG 来源：[SIAM DOI 10.1137/070693424](https://epubs.siam.org/doi/10.1137/070693424)。已核元数据和指定假设/定义段落，不宣称全部证明核验；其正态独立总体假设不适用于本项目的全部条件。

[Active Graph Learning v3](https://arxiv.org/html/2605.05703v3) A.4 已描述重叠子集预测复用，E.2 指出图更新会改变信息价值，主实验仍为单轮。因此不能声称首次缓存或首次提出变化的信息价值。所有来源支持范围与原记录一致，不因负结果而修改先前阅读深度。

最终判定：新方法创新性和有效性均未达到原正方法论文标准。对冻结候选失败的确定性判断置信度高；对未来一般图条件维护的成败不作概率估计。优势是透明且有强反例控制；劣势是一个任务、有限图族、未完整复现外部 SOTA、未完成独立人类来源核验。文献定义和证据边界优先于“冲击 ACL”的目标措辞。
