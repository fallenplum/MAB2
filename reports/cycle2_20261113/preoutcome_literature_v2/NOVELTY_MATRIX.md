# 增量新颖性矩阵

本表是反向核查，不是新颖性证书；阅读深度见 PAPERS.json 及并行报告。候选仅 H1/H2，以下所有标准技术都不得独立作为创新。

| 拟议说法 | 最强先例/已知结果 | 仍可研究的窄差异 | 必须拒绝的夸大 |
|---|---|---|---|
| 通过实际反事实验证协调主张 | [VOUCH，Findings IJCNLP-AACL 2026](https://aclanthology.org/2026.findings-aacl.222/)；EvoSteer 预印本的配对验证 | 图背景变化后旧边信用是否有付费修复价值 | 第一个反事实验证；验证通过等于答案正确 |
| 按通信影响定位重要边/节点 | [CAPE/TCTE，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.623/) | 配对有符号删留价值与旧信用动作错误 | 答案发生变化就表示通信有益；注入 gold 是免费信号 |
| 组内多图评分降低任务难度混杂 | [Graph-GRPO，Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1010/) | 直接配对条件信用估计与失效检测 | 边出现时奖励均值天然等于边的因果边际 |
| 查询条件稀疏拓扑/图价值代理 | [G-Designer，ICML 2025](https://proceedings.mlr.press/v267/zhang25cu.html)；[GTD，ACL 2026](https://aclanthology.org/2026.acl-long.1764/) | H1 的 graph×处理前内容信号是否增量识别旧方向错误 | 仅换图预测网络；以前不计训练/总成本 |
| 剪去冗余或有害时空通信 | [AgentPrune，ICLR 2025](https://openreview.net/pdf?id=LkzuPorQ5L) | 旧剪枝在新的背景下何时需要复检 | 剪枝本身是新方法；未全文核实就声称对方完全静态 |
| 自适应采反馈、提前停止、控制错误接受 | [aLTT，ICML 2025](https://proceedings.mlr.press/v267/zecchin25a.html) | 固定图信用的合适估计目标及同完整成本实证 | confidence sequence/alpha spending 新定理；每题无伤害保证 |
| 信息收益除以共享/可变成本 | [Zheng 等，NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/2adee8815dd939548ee6b2772524b6f2-Abstract.html)；[CAET，AISTATS 2025](https://proceedings.mlr.press/v258/wu25c.html)；[SH-RR，AISTATS 2024](https://proceedings.mlr.press/v238/li24c.html) | 实际图后缀成本模型与公平工程效果 | 所有传统 bandit 都是固定等成本；除以成本就新颖 |
| 是否值得为未来收益询问反馈 | [Active Teacher Selection，TMLR 2026](https://openreview.net/pdf?id=9OEy68av40) | H2：明确服务时段内旧信用排序能否改变查询/重学/不问的取舍 | 机会成本/VOI 新原则；无限 H 摊销后宣称节省 |
| 有限银行侧信息与已知集更新 | [Shekhar 等，UAI 2023](https://proceedings.mlr.press/v216/shekhar23a.html) | 合法共享缓存下的应用与预算核算 | HT/CV/条件均值恒等式新理论；有限银行等于未来分布 |

## 旧失败排除表

| 不再提名的路线 | 原因 |
|---|---|
| 更大图预测器替代 Ridge | 没有新的决策信息假设；不解决当前主要机制问题 |
| β=1 old-A + paired(B−A) | 旧 R2 已直接测试并未通过固定风险门，不可换成“residual correction”重新提名 |
| dirty 优先/同 gap_bundle 加一项 | 旧 R1/R3 主候选未通过；新增词汇不构成实质新路线 |
| oracle 证书节约即算法节约 | 旧 R3 oracle 使用银行真值；不能替换已失败可执行算法 |
| receiver 回答改变作为免费主动信号 | 该差异需要干预生成；必须先支付并与所有基线共享同权限 |
| 合法 exact-cache 或共享 anchor 就算新理论 | 既有系统与统计先例充分；旧银行额外复用空间小 |
| 当前 Hotpot 私有证据分片/换 A、F1、seed、edge | 违反现合同；会改变信息结构或事后选择结果 |

## 可保留的主张等级

- **工程级：** 可审计的同算子、同信息、完整费用验证。可有价值，但不冒称通用新算法。
- **科学机制级：** 独立证明 graph×处理前信息识别旧动作错误，且复检带来增量服务效用。现无新结果支持。
- **算法/理论级：** 必须额外证明超出上述强先例的技术与保证；本轮没有建立，不应预写成论文贡献。
