# 已解封冻结评价的聚合报告

本报告仅按预定范围展示固定结果；不新增检验、择优条件或准入。全部 384 题、11 条件保留。

## 冻结主要结果

D1–D4 的估计、区间、支持数及门均直接引用冻结评估输出，缺失则为 NA；没有重新抽样或重算判据。区间是原 Bonferroni 近似 bootstrap，不是有限样本精确保证。

| 对比 | 估计 | 原区间 | 非零支持 | 支持门 |
|---|---:|---|---:|---|
| D1: B minus no-edge | -0.046875 | [-0.0963542, 0.00260417] | 60 | True |
| D2: B minus independent12 | -0.0677083 | [-0.117188, -0.0208333] | 58 | True |
| D3: Mean signed delta | -0.00520833 | [-0.0364583, 0.0260417] | 58 | True |
| D4: Mean cross-seed delta product | 0.0182292 | [-0.00260417, 0.0442708] | 10 | False |

D1–D3 使用 EM 比例差；D4 使用 EM 比例的平方单位，保留负值，不开平方。

| 冻结判据字段 | 原结果 |
|---|---|
| L1 | B_DISFAVORED |
| L2 | INCONCLUSIVE |
| signed_direction | INCONCLUSIVE |
| action | STOP_CURRENT_B_DIRECTION |
| theta_small_effect_exclusion | NOT_PERMITTED_BY_THIS_BOOTSTRAP_CONTRACT |
| theta_nonzero_supported | False |
| theta_negative_point_retained | False |
| GPU_admission_granted | False |
| method_success_claim | False |
| theta_width_divided_by_0_0025 | 18.75 |
| observed_point_frontier | NOT_DOMINATED |

STOP_CURRENT_B_DIRECTION 等字段只约束当前 B 路线。即使门通过，也只允许讨论独立新合同下的复检机制；不证明主动反馈胜过随机、不证明异构模型因果优势、不代表论文证据齐备。失败后继续既有外循环，项目停止仍服从总合同。

## 全部条件的描述性结果

| 条件 | n | EM | F1 | 输入拒绝置零 | aggregate格式失败 | 生成结束原因 |
|---|---:|---:|---:|---:|---|---|
| S1/A | 384 | 0.510417 | 0.598254 | 0 | NA | NA |
| S1/B | 384 | 0.479167 | 0.57691 | 0 | NA | NA |
| S1/NO_EDGE | 384 | 0.526042 | 0.611458 | 0 | NA | NA |
| S1/A_MINUS_E | 384 | 0.510417 | 0.600098 | 0 | NA | NA |
| S1/B_MINUS_E | 384 | 0.484375 | 0.58042 | 0 | NA | NA |
| S1/INDEPENDENT12 | 384 | 0.546875 | 0.621307 | 0 | NA | NA |
| S1/SINGLE_M | 384 | 0.424479 | 0.514781 | 0 | NA | NA |
| S2/A | 384 | 0.513021 | 0.60205 | 0 | NA | NA |
| S2/B | 384 | 0.484375 | 0.578553 | 0 | NA | NA |
| S2/A_MINUS_E | 384 | 0.518229 | 0.61102 | 0 | NA | NA |
| S2/B_MINUS_E | 384 | 0.494792 | 0.593764 | 0 | NA | NA |

全局 aggregate 格式失败计数：59；已完成评分条件数：4224；输入拒绝数：0。
逐条件格式失败及生成结束原因未由冻结评估输出提供。NA 表示不可用，不能解释为零或未发生；报告器不从原文补抽答案。

## 有符号信用与成本

gamma_A=A−(A−e)，gamma_B=B−(B−e)，delta=gamma_B−gamma_A。两个 seed 分列，跨 seed 乘积保留负值；全部题、四边等权。符号计数和固定离散直方图只作描述，不替代 D3/D4 或原支持门。F1 的零按原浮点值精确比较，不另调容差。

完整逻辑成本仅有 S1 的 B、no-edge、independent12、single-M 四方法，按模型分别列调用、输入/输出 token 和 CUDA 事件秒；包括复用节点的完整服务费用。其余七条件为 NA。独立12匹配调用及模型分配，不保证 token 或设备时间相等。

物理研究调用、缓存节省、驻留、初始化、等待和失败费用的数值未由冻结评估输出提供，全部另列 NA。相关证据仅保留在私有回执，本公开报告没有覆盖全部成本；CUDA 事件秒不是物理驻留，也不按逻辑调用估算 GPUh。

固定长度层未出现在该输出中，因此不生成长度分组。两个 seed 不代表全部采样变化，单任务不支持普遍系统结论。

## 图表来源与范围

图只读取本次已净化的聚合表。所有条件顺序固定，负值与缺失值保留；描述性 EM/F1 和成本没有另造误差条。主要区间图只使用上述原区间。PNG/PDF 及 CSV/JSON 均不含逐题行、标识、原题、答案或来源指纹。
