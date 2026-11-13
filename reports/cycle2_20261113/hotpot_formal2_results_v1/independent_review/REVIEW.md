# Hotpot 已解封评价数组：独立统计与结论核对

**结论：独立复算通过；当前预定 B 路线的系统优势不受支持，图条件信用变化证据不足。**
这不是新实验，也不是对整个 Graph-Conditional Communication Credit Revalidation 研究主题的停止判断。

## 1. 复核范围与可信程度

只读取已正式解封的评价数组、评价回执、评分批准元数据及冻结科学合同/源码。没有读取标签库、原题、逐请求生成、INPUT、MATRIX 或任何新数据；没有运行模型、GPU、修改 E 盘或 Git。没有导入或调用生产统计与前沿函数。

预注册共 384 题，4 个固定边层各 96 题，两个生成 seed 为 26111301/26111302。11 个条件共 4,224 个完整终态，评价状态全部为严格聚合器评分；评分审核元数据记录 0 输入溢出、0 运行失败。这里独立检查了终态覆盖与绑定，**没有重新读取原生成核验答案评分**。

按冻结 PCG64(261113103) 一次生成 (10000,4,96) 抽样索引，保持题内全部 seed/图/对照配对。生产程序通过索引 gather 后求均值；本复核用各抽样中每题出现次数构成频数矩阵，再矩阵乘法求均值。抽样索引 SHA 完全一致。主指标用线性分位点 [0.00625,0.99375]，4 比较、family alpha 0.05 的近似 Bonferroni percentile bootstrap；这些不是有限样本精确覆盖保证。次指标的 [.025,.975] 仅描述性，不参与门。

**529 项真实输入算术/哈希/轴/分类检查全部通过**；最大绝对数值差 2.7755575615628914e-17。另有 4 项小型人工反例测试通过，覆盖信用符号、题内交叉积、分层频数抽样、低支持/退化门。对实现重放的一致性置信度高；统计总体解释限于合同假设、近似区间和当前任务/算子。

## 2. 预注册定义、符号与单位

冻结图轴为 [A,B,A_MINUS_E,B_MINUS_E]。对题 q、seed s：

- γ_A(q,s) = EM(A) − EM(A−e)，γ_B(q,s) = EM(B) − EM(B−e)。正信用表示保留该边提高回答正确性。
- δ(q,s) = γ_B(q,s) − γ_A(q,s)。正值表示该边在 B 下的信用高于在 A 下；负值表示相反。它不是 B 的整体系统收益。
- D1 = mean_q[EM(B,seed1) − EM(NO_EDGE)]。
- D2 = mean_q[EM(B,seed1) − EM(INDEPENDENT12)]。
- D3 = mean_q[(δ(q,seed1)+δ(q,seed2))/2]。
- D4 = θ̂ = mean_q[δ(q,seed1)δ(q,seed2)]，**先题内相乘再平均**，不是两个总体均值的乘积。

D1–D3 是 EM 比例差，可乘 100 表示百分点。D4 单位为 EM²，不乘 100 冒称准确率百分点，不截断负值，不从其点值推断已确认 RMS。若两个 seed 流在给定题/边后条件独立且各 δ 对同一固定随机算子的条件均值无偏，θ 对应条件平均漂移平方的题总体均值；两份固定 seed 数据本身不能经验性证明这些假设。

## 3. 四个主统计独立结果

| 统计 | 点估计 | 冻结校正区间 | 非零题簇（正/负） | 支持门 | 结论 |
|---|---:|---|---:|---|---|
| D1：B−NO_EDGE | −4.6875 pp | [−9.6354, +0.2604] pp | 60（21/39） | 通过 | 区间跨零；未支持优于 NO_EDGE，也不能据此称等价 |
| D2：B−INDEPENDENT12 | −6.7708 pp | [−11.71875, −2.08333] pp | 58（16/42） | 通过 | 固定近似区间完全低于零，支持本设置中 B 较差 |
| D3：平均有符号漂移 | −0.00520833 | [−0.03645833, +0.02604167] | 58（28/30） | 通过 | 未建立一致漂移方向 |
| D4：题内跨 seed 乘积 | +0.01822917 EM² | [−0.00260417, +0.04427083] EM² | 10（8/2） | **不通过** | 非零数 10 < 20；正点值不足以支持信用变化信号，区间仅描述 |

D3 的百分点表达为 −0.520833 pp，区间 [−3.645833,+2.604167] pp。D4 区间宽度 0.046875，为合同 0.0025 参考量的 18.75 倍；合同禁止用这种 bootstrap 区间自动排除小 θ。少事件与区间跨零意味着**不确定**，不能将非显著说成没有信用变化。

原冻结分类逐字段重放一致：

- L1 = `B_DISFAVORED`：D2 已满足原“任一质量区间上界 < 0 且支持门通过”的规则。
- L2 = `INCONCLUSIVE`：D3 无方向支持，D4 未满足支持门与下界条件。
- action = `STOP_CURRENT_B_DIRECTION`：停止当前 B、当前 full-context 信息结构与固定算子的自动升级。
- `method_success_claim=false`、`GPU_admission_granted=false`。

## 4. 全部 11 个条件的聚合均值

下面所有 A、编辑后图、seed2 与 F1 结果都是完整披露的描述量，不能替换唯一预定候选 B，也不能用于筛选 seed/边/子组救援。

| 条件 | EM 成功数 | EM | answer F1 |
|---|---:|---:|---:|
| S1/A | 196/384 | 51.0417% | 59.8254% |
| S1/B | 184/384 | 47.9167% | 57.6910% |
| S1/NO_EDGE | 202/384 | 52.6042% | 61.1458% |
| S1/A_MINUS_E | 196/384 | 51.0417% | 60.0098% |
| S1/B_MINUS_E | 186/384 | 48.4375% | 58.0420% |
| S1/INDEPENDENT12 | 210/384 | 54.6875% | 62.1307% |
| S1/SINGLE_M | 163/384 | 42.4479% | 51.4781% |
| S2/A | 197/384 | 51.3021% | 60.2050% |
| S2/B | 186/384 | 48.4375% | 57.8553% |
| S2/A_MINUS_E | 199/384 | 51.8229% | 61.1020% |
| S2/B_MINUS_E | 190/384 | 49.4792% | 59.3764% |

S1 的 B=184/384、NO_EDGE=202/384、INDEPENDENT12=210/384、SINGLE_M=163/384。B−SINGLE_M 点差 +21/384=+5.46875 pp 仅描述性；廉价单模型不能替代已预定的两个强基线。

## 5. 费用前沿复核与不能推出的结论

独立从评价文件的完整逻辑费用聚合重构了 8 维向量与所有支配比较：每模型分别 calls/input_tokens/output_tokens/CUDA_event_seconds。B、NO_EDGE、INDEPENDENT12 均为 Qwen 2,304 次 + Mistral 2,688 次，即每题 13 次逻辑调用；SINGLE_M 为 Qwen 0 次 + Mistral 384 次。缓存节点仍计入完整服务费用。未知没有置零；SINGLE_M 的 Qwen 全零对应真实零调用。

结果是 `NOT_DOMINATED`，没有支配见证。原因可逐项核对：

- NO_EDGE 比 B 多 18 个正确答案，但 Qwen 输出 token 多 6,297、Qwen CUDA 事件秒多约 341.27；并非 8 维成本全部更低。
- INDEPENDENT12 比 B 多 26 个正确答案，但 Qwen 输出 token 多 3,359、Qwen CUDA 事件秒多约 118.40、Mistral 输入 token 多 36,492；仍未满足全部成本维度不高于 B。
- SINGLE_M 各成本维度均更低，但少 21 个正确答案。

这是**有限观测点的描述性 Pareto 条件**。维度多且存在成本交叉，NOT_DOMINATED 很容易出现；它不是成本最优、总体支配检验、净系统收益或部署优势。共享硬件的 CUDA 时间有测量/争用影响，原规则不加事后容忍带。前沿不救 L1/L2，不能把 token 加权成事后收益。

本复核验证的是已经审计过的费用聚合的几何、数值和支配逻辑；**没有在本次重读节点/native 日志重新加总 token 或 CUDA 时间**。研究采集驻留小时、加载/等待/失败属于另一个实际物理费用账，不得与这些重计缓存的逻辑费用重复累计。删 peer 边仍保留 self-state 和 13 次服务调用，不能将采集缓存节省声称为部署少调用。

## 6. 科学结论边界与风险

**得到支持的结论**：在本次固定 384 题、完整 context、固定路由/预算下，B 相对预定 INDEPENDENT12 的 EM 差为负，满足原近似区间和支持门的 `B_DISFAVORED` 分类；代码、数组轴、bootstrap 和该分类一致。

**未得到支持的研究主张**：B 同时优于两种强基线；本设置下可据此升级为有效主动信用维护方法；维护/采集机制的净系统收益；任何异构模型的因果优势。此轮只测适用性，不是主动复检算法对照。

**仍不确定**：B 与 NO_EDGE 的总体差异方向、平均信用漂移方向、可重复的非零图条件信用变化规模、未来服务分布或其他合法图/任务/信息结构。不能把本次结果推成“通信普遍无效”或“整个项目停止”。未来研究须另立有界合同，本报告不提出或授权新门、新题、新算子或 GPU。

优势是预先固定任务/候选/强基线/seed/四主门，完整配对保留且全部负结果披露；风险是仅一个候选图/任务/路由与两 seed、D4 稀疏且检验力不足、bootstrap 仅近似、八维观测费用前沿不能替代实际部署测量。没有用 A/F1/子组进行事后救援。

## 7. 来源与复现

冻结定义来自 `protocol/cycle2_hotpot_applicability_20261113/SCIENTIFIC_DESIGN_FROZEN.json`，尤其 `statistics`、`evidence_rules` 与 `scientific_scope`；费用规则来自同目录 `cost_frontier_rule/PROSPECTIVE_RULE.md`。生产源码仅只读用于解释与 SHA 绑定，本独立程序不导入其实现。

| 输入别名 | 文件名 | SHA256 |
|---|---|---|
| evaluation | `PRIVATE_EVALUATION.json` | `0f63c65f6d60d8ca924ff1c9e814425b4e906a1658f1ce81abf24530b9d32270` |
| evaluation_receipt | `EVALUATION_RECEIPT.json` | `653a7df8c6eb0df7d6c7c23c1253c53219437578b6e8ea55e435e9c8254ca506` |
| score_review | `SCORE_REVIEW.json` | `f3482981cb9bbd11c08a746ae35620e7ccb5868ea31cfed46900ea1332887e36` |
| scientific_design | `SCIENTIFIC_DESIGN_FROZEN.json` | `e69b608aa065517934985b17261a7a225383439ed95bb8580b52ea4bcd29ab59` |
| statistics_source | `applicability_statistics.py` | `3417dc11bc583d72a2ec8c8cc139c6ee490aa039d8eb386c97f7e2aa52060bfe` |
| frontier_source | `point_frontier.py` | `6f641a3ff9c00afc2b045bc08435e42c602df5756747c58a7852f017da7bf7f9` |
| frontier_rule | `PROSPECTIVE_RULE.md` | `1411cad3d29fff2619b7754267270c9adf3c92b2d192a8bc16d5181d3befe800` |
| evaluation_adapter_source | `evaluation_adapter.py` | `f8ba38be9ce1ef6bbfee66c17606685b117d25c52b947e66937ea5574a6105b8` |
| postclosure_source | `postclosure.py` | `8f701f52ac7fa15ea39f55f2beb877ce400774cc88484e7280c45da322ccb544` |

独立结果 SHA256：`48efe45908d9afbd8c83fc07449dc6ced296d9fd161e2d65be9e52f6e0e79465`。

复现脚本 `independent_statistics_review.py` 接受 `--evaluation`、`--evaluation-receipt`、`--score-review`、`--source-root`、`--output-dir`；三个真实输入 SHA 已固定并 fail-closed。只在获得相同私有评价数组访问授权后运行。脚本拒绝覆盖已有 RESULTS。发布候选为本报告及三个聚合 CSV；无逐题数组、qid、私有本机路径或题文本。提供复算脚本不能代替私有原数据的受控可用性说明。
