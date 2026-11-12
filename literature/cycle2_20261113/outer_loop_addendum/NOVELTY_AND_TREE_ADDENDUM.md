# 外层研究补充：直接反事实、条件信用与运行时消息

原30条文献快照保留。本补充增加7个条目，4篇仅核读摘要、3篇核读指定章节；未声称检索饱和或完成独立复现。原文方法的效果不是MAB2实验结果。

| 研究对象 | 已核实近邻 | MAB2仍须证明 |
|---|---|---|
| c3_exact_trace | [The Trace Is the State: Exact Credit Assignment for LLM Agent Teams](https://arxiv.org/html/2603.06859v4) | Predictor-free executed credit, shared-prefix savings, and state restoration are precedented. MAB2 targets maintenance across graph edits and subsequent graph decisions, not policy-gradient training. |
| filtered_feedback_marginal | [Marginal-Contribution Policy Gradients under Filtered Feedback for Multi-Agent LLMs](https://arxiv.org/html/2604.22785v2) | Conditional credit and budgeted correction are not standalone novelty. The stated allocation assumes independent Bernoulli evaluations and an expected-cost constraint; this is not a proof for a hard cap with shared caches or future graph-edit utility. |
| messagesieve | [MessageSieve: Runtime Message Pruning for Accurate and Efficient LLM-Based Multi-Agent Communication](https://aclanthology.org/2026.emnlp-main.2179/) | Message-sensitive communication control has a direct precedent. Exact replay equivalence and historical credit validity require separate definitions. Detailed implementation and comparisons remain unread. |
| vouch | [Vouch: Counterfactual Contracts for Multi-Agent Coordination](https://aclanthology.org/2026.findings-aacl.222/) | Auditable counterfactual commitments are not inherently new. It is not yet verified as a direct graph-credit baseline. |
| replay_noise | [Re-feeding Is Not Replaying: Measuring Replay Noise in Counterfactual Token-Credit Estimation](https://arxiv.org/abs/2606.15621) | Local replay checks cannot establish arbitrary engines or hardware are equivalent. MAB2 must keep its actual replay checks, fixed inference conditions and numerical limitations explicit. |
| edge_multierror | [EDGE: Error Dependency Graph-Guided Multi-Error Attribution in Multi-Agent LLM Systems](https://aclanthology.org/2026.findings-emnlp.387/) | Graph-guided causal verification is adjacent prior art; this abstract does not establish an adaptive credit-revalidation budget policy. |
| cachescout | [Learning Agent Execution for KV-Cache Management in Agentic Serving](https://arxiv.org/abs/2608.14624) | Serving KV-cache reuse differs from complete-output reuse across counterfactual graph executions. No equivalence of these cost models is assumed. |

C3 v4已改变题名与作者列表，应以固定版本元数据引用。背景条件信用的定义、直接反事实执行、控制变量估计和预算纠错均不能各自充当本项目独立创新。当前R1仍是机制可行性实验；任何后续方法须有新的独立数据、公平信息/执行预算、图演化后的决策收益和明确的适用边界。

后续候选C的探索问题是：当实际消息未改变时，能否用可核验的执行等价性减少不必要复检，并把节省的反馈转给真正不确定的图编辑？这只是待分析假设；所有对照必须共享同一精确缓存，不能把系统缓存优化单独算成采样策略收益。
