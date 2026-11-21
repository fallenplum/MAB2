# 时间问题、版本化证据与图条件信用

| 分支与代表工作 | 已有贡献 | 本项目需要额外识别的效应 |
| --- | --- | --- |
| 开放检索时间问答，Zhang 与 Choi，EMNLP 2021 | 对同一问题给出不同时间条件与答案；文章及数据说明区分问题时间和检索语料时间。训练实例含 Natural Questions 来源。 | 必须比较同信息的带日期直接回答与普通检索。时间适配本身已有研究。[论文](https://aclanthology.org/2021.emnlp-main.586/) · [作者数据](https://github.com/mikejqzhang/SituatedQA) |
| 时间敏感阅读理解，Chen 等，NeurIPS 数据集与基准轨道 2021 | 将变化事实与维基百科文章对齐，由人工核对，从中构造带时间约束的问题。 | 单篇回顾性文章允许读取多个时期，不能冒充不同时点发布的两份语料。需检验固定同一时间时换图能否独立改变消息价值。[论文](https://arxiv.org/abs/2108.06314) · [作者数据](https://github.com/wenhuchen/Time-Sensitive-QA) |

检索日期 2026-11-21。定向检索词为 temporal situated questions、time-sensitive QA、Wikipedia source passages、dated questions。以上是数据构造与信息预算的邻近工作，覆盖范围不足以证明新颖性。旧项目中的图剪枝与反馈采集矩阵继续适用。
