# D27b 作者原始编号的补充核对

D27 已经记录：100 个 subtle 变体有显式 parent；两套改写的 parent 字段均缺失。作者的实现对两个改写集使用与 HumanEval 相同的 raw-input 求值路径，论文说明原题与两套改写复用测试。该证据提示另一种可检验的数据连接途径。[作者评估代码](https://github.com/evo-eval/evoeval/blob/d5ca3104ec30b99f1076f51d4476eb4c3f29effa/evoeval/evaluate.py#L149-L168) · [论文第 3 节及表 1](https://arxiv.org/html/2403.19114v1)

在检查改写 ID 的实际取值前固定第二种标准。改写题的完整 task_id 必须与原版标识逐字相同，且对应 entry_point 逐字相同。subtle 仍使用显式 parent。每组出现一次才可进入交集。既不剥离编号前缀，也不依靠行号或题面匹配。只使用编号、函数入口名与发布的 parent，零模型调用。该结果与 D27 的父字段统计并列保留。即使可配对，仍需另行核验曝光和协议，不能由元数据推断图复检有效。
