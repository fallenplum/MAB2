# MAB2 Literature Tree — 2026-11-13

**状态：PROVISIONAL。** 这是按科学问题组织的关系树，分支表示本审计的概念关联，不表示已验证的逐篇引用关系。23篇记录中11篇达到相关章节阅读；0篇计作全篇深读或独立复现。

## 文献树

- 图作为可优化协作系统：GPTSwarm（ICML 2024，摘要）→ AgentPrune（ICLR 2025，章节）→ AgentDropout（ACL 2025，章节）。
- 历史信用与演化剪枝：SafeSieve（AAAI 2026，章节）；Graph-GRPO（Findings ACL 2026，章节）；CATS（2026预印本，章节）。
- 条件/反事实归因：COMA（AAAI 2018，摘要）；E2-Explainer、Agents that Matter v2（2026预印本，章节）；CCPO、TOCA（2026预印本，摘要）。
- 反馈与图决策：Nash-CredMAS（Findings ACL 2026，章节）；MASRubric v3（2026预印本，章节）；Feedback Attribution（EMNLP 2021，摘要）。
- 信息获取预算：CPE（NeurIPS 2014，摘要）；Track-and-Stop（COLT 2016，摘要）；Active RL（2020 arXiv记录，摘要）；CABAI（RLC 2024，章节）；BAIwRC（AISTATS 2024，摘要）；Active Learning for Communication Structure Optimization（2026预印本，章节且窄主张经root独立核验）。
- 评估协议：Rethinking the Evaluation of Efficiency Methods for Multi-Agent Systems（EMNLP 2026，官方摘要）。
- 预测/决策目标错配：Decision-Focused Learning（AAAI 2019，摘要）；SPO（Management Science 2022卷期，摘要）。

## 对当前方案的约束

已有研究覆盖“图剪枝”“历史反馈”“图条件边反事实”“成本意识”和“决策指标优于只看MSE”。MAB2候选贡献需收缩到图变化后信用重新测量的选择机制，以及完整服务预算下的后续决策收益。此差别尚未被新实验支持。

SafeSieve Eq.(11)已经做剪后分数和历史权重调整；E2的反事实标签已经依赖当前完整图；CATS已把归因成本列入搜索token。不能把这些工作描述为完全静态、图无关或全部不记成本。

## 阅读与检索边界

下载不等于精读；本轮原文只读与机制及实验约束有关的章节。原始全文为私人审计材料，不应把PDF全文上传研究代码仓库。arXiv本地网络连接失败时，使用官方HTML正文；对应记录local_fulltext=null。

检索未饱和：后期同义检索继续找到CATS与Agents that Matter，尚无足够证据宣称相关机制不存在。本稿是可审查初稿，不是新颖性验收通过。

## 逐篇来源

- **agentprune** — [Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems](https://proceedings.iclr.cc/paper_files/paper/2025/file/bbc461518c59a2a8d64e70e2c38c4a0e-Paper-Conference.pdf); 2025; ICLR 2025; `section_read`。
- **agentdropout** — [AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](https://aclanthology.org/2025.acl-long.1170/); 2025; ACL 2025 Long Papers; `section_read`。
- **safesieve** — [SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication](https://ojs.aaai.org/index.php/AAAI/article/view/40236); 2026; AAAI 2026; `section_read`。
- **e2_explainer** — [Discovering Efficient and Explainable Communication Topologies for LLM-based Multi-Agent Systems via Causal Inference](https://arxiv.org/abs/2608.12921); 2026; 正式venue未核实; `section_read`。
- **masrubric** — [MASRubric: Auditing Information Flow in Multi-Agent Systems with Failure-Distilled Pitfall Rubrics](https://arxiv.org/abs/2602.23258); 2026; 正式venue未核实; `section_read`。
- **graph_grpo** — [Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization](https://aclanthology.org/2026.findings-acl.1010/); 2026; Findings of ACL 2026; `section_read`。
- **nash_credmas** — [Nash-Pruned CredMAS: Dynamic Panel Pruning for VLM-MAS using Nash-based Selection and Doubly-Robust Credits](https://aclanthology.org/2026.findings-acl.1975/); 2026; Findings of ACL 2026; `section_read`。
- **cats** — [Credit Where Credit Is Due: Contrastive Attribution Topology Search for Multi-Agent Systems](https://arxiv.org/abs/2610.29292); 2026; 正式venue未核实; `section_read`。
- **cabai** — [Cost Aware Best Arm Identification](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_193.pdf); 2024; RLC 2024 / Reinforcement Learning Journal; `section_read`。
- **agents_that_matter** — [Agents that Matter: How Subtle Choices Shape Agent Attribution in Multi-Agent Systems](https://arxiv.org/abs/2605.27621); 2026; 正式venue未核实; `section_read`。
- **active_learning_structure** — [Active Learning for Communication Structure Optimization in LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2605.05703); 2026; 正式venue未核实; `section_read`。
- **bai_resource** — [Best Arm Identification with Resource Constraints](https://proceedings.mlr.press/v238/li24c.html); 2024; AISTATS 2024; `abstract_screened`。
- **active_rl** — [Active Reinforcement Learning: Observing Rewards at a Cost](https://arxiv.org/abs/2011.06709); 2020; 正式venue未核实; `abstract_screened`。
- **decision_focused** — [Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/3982); 2019; AAAI 2019; `abstract_screened`。
- **spo** — [Smart “Predict, then Optimize”](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922); 2022; Management Science 68(1):9-26; `abstract_screened`。
- **coma** — [Counterfactual Multi-Agent Policy Gradients](https://ojs.aaai.org/index.php/AAAI/article/view/11794); 2018; AAAI 2018; `abstract_screened`。
- **feedback_attribution** — [Feedback Attribution for Counterfactual Bandit Learning in Multi-Domain Spoken Language Understanding](https://aclanthology.org/2021.emnlp-main.91/); 2021; EMNLP 2021; `abstract_screened`。
- **track_stop** — [Optimal Best Arm Identification with Fixed Confidence](https://proceedings.mlr.press/v49/garivier16a.html); 2016; COLT 2016; `abstract_screened`。
- **gptswarm** — [GPTSwarm: Language Agents as Optimizable Graphs](https://proceedings.mlr.press/v235/zhuge24a.html); 2024; ICML 2024; `abstract_screened`。
- **cpe** — [Combinatorial Pure Exploration of Multi-Armed Bandits](https://papers.neurips.cc/paper_files/paper/2014/hash/d3ea0f3316d2da934d79b8b344eafee4-Abstract.html); 2014; NeurIPS 2014; `abstract_screened`。
- **ccpo** — [Counterfactual Credit Policy Optimization for Multi-Agent Collaboration](https://arxiv.org/abs/2603.21563); 2026; 正式venue未核实; `abstract_screened`。
- **toca** — [Turnover-Orthogonal Credit Assignment for Open-Team Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2610.02847); 2026; 正式venue未核实; `abstract_screened`。
- **mas_efficiency_eval** — [Rethinking the Evaluation of Efficiency Methods for Multi-Agent Systems](https://aclanthology.org/2026.emnlp-main.798/); 2026; EMNLP 2026; `abstract_screened`。

## 工具说明

本审计采用 research-rigor 的证据分级，以及 paper-lookup / literature-review 的来源与检索记录流程。未调用外部AI审稿/绘图服务或付费API。相关流程软件文献：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner. 2026. [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)，检索时最新v2；不将其视作MAB2技术效果证据。
