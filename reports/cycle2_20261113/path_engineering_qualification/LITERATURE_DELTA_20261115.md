# Formal-3 运行期间的 Literature Tree / Novelty Matrix 增量

检索时间：2026-11-15。检索范围：通信反事实、预算化信息采集、校准和图变化；这是定向补充，不是穷尽综述。Google 查询页面经工具访问失败；实际发现来自联网搜索，核验来自 ACL Anthology 与 arXiv 一手页面。未将镜像、搜索摘要或预印本自动视为会议录用。

查询：`multi agent communication credit revalidation`；`multi-agent counterfactual communication pruning LLM budget`；`value of information feedback graph LLM agents`。未向搜索服务发送私有数据、题目、模型输出或凭据。

## 文献树增量

- **拓扑干预与信息传播**：Shen et al., *Understanding the Information Propagation Effects of Communication Topologies in LLM-based Multi-Agent Systems*, EMNLP 2025, pp. 12347–12361. [Anthology](https://aclanthology.org/2025.emnlp-main.623/), [原文](https://aclanthology.org/2025.emnlp-main.623.pdf)。通过改变代理输出分析传播，提出 EIB-Learner。我们的边删除与历史信用维护不同，但不能声称首次研究拓扑条件下的反事实影响。
- **反事实图校准**：Huang et al., *Counterfactual Graph for Multi-Agent LLM Calibration*, arXiv:2605.30653v1, 2026-05-28，[原文](https://arxiv.org/html/2605.30653v1)。CAGE-Cal 比较通信前后依赖；附录 F.1 的 CAGE-Select 执行所有候选拓扑后选答案。用于比较时必须计入全部候选执行成本，不能当成零成本预路由。未核实正式会议录用。
- **决策价值驱动的采集**：Lee et al., *Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs*, arXiv:2610.32181v1, 2026-10-23，[原文](https://arxiv.org/html/2610.32181v1)。DIVA/DIVA-U 用模型给出的概率构建检查的净决策价值与停止规则。其附录指出成本感知停止解释了相当部分收益，成本化 EIG 是必要强对照。仅把不确定性除以成本不构成充分新颖性。
- **消息选择的外部重验与负结果**：Xue et al., *SafeRoute-MAS: Coverage-Aware and Risk-Calibrated Message Routing for Multi-Agent Scientific Reasoning*, arXiv:2611.13454v1, 2026-11-10，[原文](https://arxiv.org/html/2611.13454v1)。保留全部或删一条消息；缓存上的正收益未建立外部端点选择器优势，资格规则为回顾性分析。这里的区分是历史信用在真实图变更后的适用性及有限复检，不能把缓存动作选择重新命名为创新。
- **信息奖励训练**：Chen et al., *Training LLMs for Multi-Agent Collaboration with Verifiable Information Rewards*, arXiv:2610.17893v1, 2026-10-13，[一手摘要](https://arxiv.org/abs/2610.17893)。CoSIR 以可验证环境中的团队 surprisal 训练通信行为；本次只核验摘要，尚未审计全部实现。它削弱“信息价值可用于消息信用”这种泛化新颖性主张；与固定小模型下的在线复检不是同一实验问题。

## 新颖性与必要对照

| 研究主张 | 已有覆盖 | 本项目尚需证明 | 必须保留的强对照 |
|---|---|---|---|
| 图改变后边价值改变 | 拓扑传播、反事实校准已有研究 | 固定实际模型下，旧信用导致可测决策错误 | 同种子无图变更、完整图和实际删边重执行 |
| 主动反馈优于随机 | 一般决策价值采集已有研究 | 相同反馈与完整服务成本下改善后续图决策 | 随机、周期、最大不确定性、成本感知 EIG |
| 图条件维护有独立价值 | 拓扑与信息依赖先验已被使用 | 超过廉价路径保护与一般新鲜反馈模型 | 免费结构规则、同信息权限的中心化路由、去图变化特征 |
| 低成本更高回答质量 | 系统路由与消息剪枝已有研究 | 同预算优于单模型和独立聚合，含采集开销 | SINGLE_M/Q、INDEPENDENT3_M、keep-all、成本匹配随机 |

## 对下一轮的约束

本轮盲收集尚未评分；以下不是数据支持的新结论。若路径机制筛选通过，也只允许审查新的候选合同。先在已曝光开发档案上核对信用方向稳定性、错误删边与可恢复量，不能直接声称方法有效。若未通过，保留既定筛选结论，冻结该实例；区分“图变化没有可用信号”“信号不稳定”“决策已被免费结构规则解决”三种原因，避免继续扩大同一失败实例。

候选设计的优势是能够将图变化、信用老化与反馈价值拆开验证；风险是图结构先验可能已经决定最优动作，使主动采集只增加成本。任何后续采集器都必须包含零反馈基线，不得要求每个方法花满预算后再宣布净收益。

所有新实验仍按点估计和种子分解报告，不计算区间或仅为区间运行 bootstrap。文献中的统计方式不改变本项目授权。
