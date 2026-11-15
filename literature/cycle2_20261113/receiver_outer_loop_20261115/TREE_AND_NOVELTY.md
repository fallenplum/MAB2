# 接收端与反馈必要性的文献增量

检索日 2026-11-15。实际工具为联网搜索和一手站点，不声称 Google 查询成功；定向覆盖 ACL / EMNLP / ICML / ICLR / NeurIPS，非穷尽综述。没有向搜索服务发送任何项目私有数据。本次没有复现作者 benchmark。

| 分支及核验来源 | 已有能力 / 阅读深度 | 对 MAB2 的约束 |
|---|---|---|
| [G-Designer, ICML 2025](https://proceedings.mlr.press/v267/zhang25cu.html) | 官方元数据和摘要；变分图自编码器产生任务条件拓扑 | 新预测器或条件拓扑本身不足以构成信用维护创新 |
| [GTD, ACL 2026](https://aclanthology.org/2026.acl-long.1764/) | 官方摘要；多目标代理引导迭代图扩散 | 不能声称首次代理价值指导图编辑；本次未完整核代码 |
| [MessageSieve, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.2179/) | 官方元数据/摘要；固定图上用状态卡和消息比较删减文本，不增加 LLM 调用 | 必须保留零反馈内容/结构规则，不能强迫所有方法付费查询 |
| [AgentPrune](https://arxiv.org/abs/2410.02506) | 一手摘要；时空通信图一次剪枝；ICLR 2025 官方 PDF 搜索命中，直接 PDF 被浏览器验证拦截，本次不冒称全文已读 | 剪边省 token 已有先例；算法适配与作者原版复现分开 |
| [Evolving Orchestration, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/f1320d2e2842169c6fc89dcbd80e94d0-Paper-Conference.pdf) | 官方 PDF 摘要、§3.2 与附录 A 的局部段落；RL 编排及成本分析 | 固定小模型编排不等于首次动态协作；完整训练开销不能隐去 |
| [iAgents, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0534abc9e6db91683d82186ef0d68202-Abstract.html) | 官方摘要；信息不对称、人际网络通信与 InformativeBench | 受控分区不能冒充真实隐私/通信部署；引入新任务需新合同 |
| [DIVA, arXiv 2610.32181v1](https://arxiv.org/html/2610.32181v1) | 摘要与正文决策价值/停止说明复核；前轮局部附录记录保留 | “信息必须改变决策并值得成本”已有直接先例；需胜过停止和成本化信息增益 |
| [SafeRoute-MAS, arXiv 2611.13454v1](https://arxiv.org/html/2611.13454v1) | 摘要、§III-IV；缓存选择与外部重验分开，保留回顾性资格边界 | 缓存上的正确动作选择或小正点差不足以证明部署收益 |

树结构：**拓扑生成**（G-Designer/GTD）→ **结构或消息筛选**（AgentPrune/MessageSieve）→ **动态编排与信息访问**（Evolving Orchestration/iAgents）→ **反馈价值与外部资格**（DIVA/SafeRoute-MAS）。这些分支不是按时间或方法优劣排序。

仍有价值但未证实的 MAB2 问题是：在真实图变更、同样可得的信息及有限完整预算下，哪些历史信用需要重新取证；这种取证能否超过廉价保护、无需复检的强协作算子和普通反馈策略。第三轮已支持一项受控信用变化点估计，尚未支持付费复检必要性。第四轮结果未开放时，所有关于接收端改动有效的陈述仍为假设。

优势：候选归因能被上游固定的二乘二实验与无反馈策略反驳。风险：更好的接收端可能直接消除通信依赖，或免费结构规则已足够解决变化；任一种情况都应缩小或放弃主动复检主张。新颖性整体评级仍为低到中，不能据此承诺 ACL 投稿质量或录用。
