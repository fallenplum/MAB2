# 机制研究的文献树与新颖性矩阵

本轮检索聚焦一个可检验的区别：受控改写一个代理的消息时，图能传播多大的影响；保持发送策略不变而删除一条边时，图实际损失多少。前者适合压力测试，后者决定是否值得为日常图编辑购买新反馈。研究记录采用定向引文检索，不能据此断言没有更近的先例。

|研究分支与代表工作|已有贡献或可复用对照|本研究增加的检验和限制|
|---|---|---|
|通信剪枝。AgentPrune，ICLR 2025|论文针对时空图学习通信掩码。作者实现同时支持让决策节点阅读所有代理和只阅读最后节点。|固定代码版本验证连接函数，在表格数值外部任务中采用作者的 FinalRefer 模板。两个局部接口的复现不能替代原剪枝训练的成绩复现。[论文](https://arxiv.org/abs/2410.02506) · [连接实现](https://github.com/yanweiyue/AgentPrune/blob/c544dd6a1858c02c6d5d371d23c6e6ff55e0be21/AgentPrune/graph/graph.py)|
|动态图构建。AgentDropout，ACL 2025；G-Designer，ICML 2025|学习删减代理或依任务设计通信拓扑，已有准确率和成本比较。|本轮只能评价给定图、给定接口和参考问题上的复检成本与可证决策。不能声称超越其原始端到端系统。[AgentDropout](https://aclanthology.org/2025.acl-long.1170/) · [G-Designer](https://proceedings.mlr.press/v267/zhang25cu.html)|
|因果传播。Shen 等，EMNLP 2025|CAPE/TCTE 在已答对或已答错的队列上强制改变局部代理回答，研究错误和正确信息传播。|M47/M49 保持外部读出干预与内部边干预分明，实测来源概率加权和强制条件信用。两种指标均可有效，但目标量不同。训练 EIB-Learner 与本研究的计算并不等价。[CAPE](https://aclanthology.org/2025.emnlp-main.623/)|
|贝叶斯网络敏感性。CPT 直径与 TV 鲁棒性|随机核对输入分布差异的收缩、CPT 行间差异、期望有界奖励的 TV 控制已有理论。|使用固定根状态混合权重计算本次具体边和读出可见状态的界。常数与锐性属于经典总变差原理。贡献须依靠独立实证，不能将一般定理署名为新算法。[直径分析](https://arxiv.org/abs/2407.04667) · [bnmonitor](https://manueleleonelli.github.io/bnmonitor/)|
|有限预算精确推断。Horvitz 等的 bounded conditioning|按子问题对结论的潜在影响安排计算，逐步缩小决策区间。|M44/M46/M48 比较按费用的界宽策略、逐来源完整消元、先查中继行和联合调度。当前证书面向有限消息与指定阈值；未以新名字宣称发明渐进有界推断。[原论文](https://arxiv.org/abs/1304.1512)|
|抽样和方差。嵌套条件 MC；有符号测度估计|已知复采上游和下游的方差/成本分解，已知对正负测度作采样以及方差降低。|M37/M38 检查哪类复采在特定图上更有效，跨接口失效的排序如实记录。[嵌套条件 MC](https://arxiv.org/abs/1402.0243) · [有符号测度](https://jmlr.org/papers/volume21/19-346/19-346.pdf)|
|呈现与投票。选项顺序影响，NAACL 2024；顺序缓解与多数投票|重排答案选项可能改变单模型表现，跨顺序投票是已有基线。|M41 固定源和接收器各自的边际，仅改变布局是否同步。两个任务表现不同，不能把单模型位置偏差当作新发现。[选项顺序](https://aclanthology.org/2024.findings-naacl.130/) · [重排投票](https://aclanthology.org/2025.findings-ijcnlp.127/)|
|自然语言共识与来源。ConsensAgent；Copies or Sources?|已有工作研究附和、代理共识与来源依赖。|M43 对同题同模型的两个真实来源实现作交叉，保持消息边际和读出形式相同。自然文本的统一传播放大/衰减门没有通过。[ConsensAgent](https://aclanthology.org/2025.findings-acl.1141/) · [Copies or Sources?](https://arxiv.org/abs/2610.06192)|

优势是能在同一实际条件核上分别核验强制传播、自然收益、区间保证和核验成本。风险是有限消息可能过于简单，贪心数值读出只涵盖一项解码设置，TFLOPs、线上延迟与逐请求 token 成本不能互相替代。自由文本改变证书所需的可见状态，不能把抽取出的同一答案当作完全相同的输入。
