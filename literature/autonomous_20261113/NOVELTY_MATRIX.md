# MAB2 Novelty Matrix — 2026-11-13

**新颖性判断：PROVISIONAL；未通过覆盖/独立审查门。** 下表“未见”仅针对所读章节，不是全文不存在证明。

| 原文 | 已有机制与定位 | MAB2必须额外证明的差别 | 主要公平性/解释风险 |
|---|---|---|---|
| [agentprune](https://proceedings.iclr.cc/paper_files/paper/2025/file/bbc461518c59a2a8d64e70e2c38c4a0e-Paper-Conference.pdf) | 空间/时间图mask经任务回报策略梯度与低秩正则优化，再一次TopK剪枝；多query模式在少量任务上学习后固定图。（PDF pp.1-10; Sections 2, 3.1-3.4, 4.1-4.5; Eqs.(7)-(13)） | 候选区别是图编辑后历史边信用的重新测量及恢复，并将购买反馈计入长期完整预算。 | 本文已有训练阶段成本分析，不能称所有先前研究均忽略训练成本。 |
| [agentdropout](https://aclanthology.org/2025.acl-long.1170/) | 按通信轮分别学习邻接权重，先按加权度删节点，再重新学习并删边；40个训练样本，之后使用学习后的拓扑。（PDF pp.2-7 (printed pp.24014-24019); Sections 3.1-3.4, 4.1-4.3; Eqs.(1)-(8)） | “dynamic”主要是不同推理轮有不同保留结构；MAB2需检验跨未来问题的信用失效与预算化重新测量，而非仅轮级dropout。 | Eq.(4)符号与相邻描述存在intra/inter歧义，适配前需代码核对。 |
| [safesieve](https://ojs.aaai.org/index.php/AAAI/article/view/40236) | 角色语义/专家评分与历史成功贡献逐步混合，用0-extension聚类剪枝；Eq.(11)剪后重新标准化分数并调整历史权重。（arXiv v3: SafeSieve: Integrated Pruning Strategy, Eqs.(2)-(11); Experiment Setup; Tables 1-2） | 候选区别必须具体到是否购买哪条边的新反事实、如何恢复删除边及同完整预算下的决策收益；单说图变化后校正不足。 | 官方主路径9文件已部分核读：历史分数计入成功图中存在的边，非逐边反事实；聚类/训练主路径与论文存在差异，未benchmark复现。 |
| [e2_explainer](https://arxiv.org/abs/2608.12921) | 固定其他通道做单边删除，五次独立执行估计任务差与语义熵；边/节点预算子图作为监督，离线蒸馏为部署时预测器。（arXiv v2: Problem Formulation; E2-Explainer; Eqs.(3)-(17)） | 本文部署不再干预；MAB2可研究连续图编辑后选择性复检的长期价值。本文budget约束结构数量，与购买反馈预算不同。 | 正文Budget-Aware Subgraph Selection称rho为保留比例，附录Pruning counts Eq.(29)称其为删除比例；适配时须核代码，不能默选一种约定。 |
| [masrubric](https://arxiv.org/abs/2602.23258) | 离线从失败轨迹挖掘判据；在线逐消息检索和审核，按满足率选择传递、修改或扣留。（arXiv v3: Sections 3.1-3.3, 4.1; Appendix A.1 and A.6; Eqs.(3)-(5); Table 7） | 其判据是内容审核而非图边信用重新测量；MAB2应将复检开销与未来图决策收益明确挂钩。 | 同一arXiv旧版题名AgentDropoutV2；不得与ACL2025 AgentDropout混为同一方法。 |
| [graph_grpo](https://aclanthology.org/2026.findings-acl.1010/) | 每query采样16图，统计边存在条件下成功率并在组内标准化，训练GAT拓扑策略；部署时阈值选边。（PDF v2 pp.1-7; Sections 3.1-3.3, 4.1-4.4; Eqs.(3)-(7); Algorithm 1） | MAB2需要验证同一边随其他边编辑而失效及主动重新测量，不应把组内归一化或边信用学习称首创。 | S_ij=P(success|edge present)不是保持其他边不变的paired删除差；“counterfactual”用词须核公式。 |
| [nash_credmas](https://aclanthology.org/2026.findings-acl.1975/) | 离线AIPW标签训练上下文价值网络，在线按预测净收益/成本选择下一轮代理panel。（PDF pp.1-6 (printed pp.39640-39645); Sections 3.1-3.3, 4.1-4.3, 5; Eqs.(1)-(5); Algorithm 1） | MAB2候选单位是通信边与复检反馈选择，且须把学习、复检、评分、服务全部计成本。 | Eq.(3)写的是单处理潜在结果形式，不能仅凭该式推出处理差。 |
| [cats](https://arxiv.org/abs/2610.29292) | 固定supergraph中采样候选，LLM比较同任务轨迹并分配代理带符号信用，更新入边并逐步删除，最终部署固定团队。（arXiv v1 Sections 1-6; Appendix A.1-A.3; Tables 1,2,5,6） | 其任务是离线团队搜索；MAB2需展示图条件历史信用失效、选择购买新反事实和恢复对未来服务收益的独立作用。 | 正文明确matched candidate evaluations不等于matched compute。 |
| [cabai](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_193.pdf) | 固定置信下按异质抽样成本识别最高均值臂；成本归一化信息分配与停止规则。（PDF pp.1-4; Sections 1-3; Assumptions 1-2; Theorem 1） | 可迁移的是成本与信息目标分离；边信用会随图变化，服务/测量交错，需另证假设。 | 稳定臂分布、reward/cost独立、正有界成本等假设不可默认适用于LLM配对边差。 |

| [agents_that_matter](https://arxiv.org/html/2605.27621v2) | 归因取决于联盟分布、干预算子与效用；完整联盟冗余会使LOO接近零，而较小联盟可能有用。（v2 §§3–5，尤其§5.2） | MAB2需验证图编辑后的历史边信用失效及复检对未来服务的作用，不能仅把稀疏LOO重新解释称为新方法。 | 所读正文不等于附录/代码复现；删除与提示修改属于不同归因目标。 |
| [active_learning_structure](https://arxiv.org/html/2605.05703v1) | 已知参考答案的训练池中主动选任务，再执行昂贵MAS rollout优化图；Fig.1匹配8倍MAS token，Table1分列Select/Train/Test。（§§4.1–4.4,5.1–5.2） | 候选差别是图编辑后按边购买复检、未来任务流及恢复决策，不能宣称首个主动图优化或首个计采集成本。 | root独立核验仅覆盖上述窄主张；附录与代码未核，未知项不得当作不存在。 |

## 候选主张与验伪

**候选而非已证主张：** 图编辑后按预先确定的风险/信息价值选择复检，并允许恢复被删边，能在服务、初始化、复检、评分共用预算下改善未来质量。

- 最强反对意见：这可能只是SafeSieve、E2、CATS及cost-aware active learning的组合。需用图变化×复检策略的受控证据证明额外机制。
- 优势：无需追逐新预测网络；与导师问题一致，能先用旧开发数据反证。
- 风险：稀疏信用与噪声可能让风险排序近似随机；主动采集的花费可能超过服务节省。
- 核心对照：固定模型池/初始化/价值预测器结构/更新规则/完整预算，仅改不复检、随机、周期、最大不确定性、图变化驱动、成本归一化。每策略只能用自身合法购买的反馈。
- 附加外部基线：经算子与权限审计的AgentPrune/AgentDropout、SafeSieve、E2式离线剪枝；CATS式归因需预算匹配或标成不同信息/模型预算。适配版本不能冒充作者原版复现。
- 决策指标：未来回答质量、错删/恢复、累计完整成本的服务收益；MSE、符号、排序、校准作为诊断。
- 停止/降级：独立数据未显示优于随机/周期的决策价值，或优势依赖隐藏额外反馈，则降级/放弃机制主张；保留负结果。

## 已识别的理论移植边界

CABAI的稳定臂、reward/cost独立及正有界成本假设需重新审查。Nash-CredMAS中单潜在结果AIPW、可加目标、成本惩罚后的单调性与ratio贪心之间有待澄清，不能引用其定理替代MAB2证明。Graph-GRPO的条件成功率也不等于控制其他边的反事实差。

## 下一项文献验收

后续需核CCPO、TOCA正文及Agents that Matter v2附录/代码；CATS/E2官方代码尚未核实。SafeSieve部分主执行路径与合成算子检查已完成，但无benchmark复现；CATS/E2相关技术章节及预算权限审计已完成，案例排除。继续由独立读者核关键比较句。检索未饱和之前不写“首个”“现有工作均未考虑”。
