# MAB2 下一周期机制与算法审查

## 判断

当前结果支持一个重要观察，但尚不足以单独形成 ACL 级创新。多标签证书成本、随机布尔函数验证和决策区域确定已有相近抽象。已有 LLM 工作也覆盖动态通信拓扑、边归因、消息裁剪和相对信用学习。因此论文不能把“all-label certificate”或“TV 区间”单独包装成新算法。

## 可保留的机制问题

图编辑后，旧 transcript 是否仍能证明新图的 keep/remove 动作，是一个图条件的证书传输问题。它区别于重新估计边信用，也区别于只最小化预测不确定性。应把主对象定义为 action-region membership。对阈值 tau，旧决定安全可运输的充分条件是每个标签的旧区间在图漂移半径扩大后仍不穿过 tau。

对

\[
\Delta_y(G)=\sum_i s_i(G)q_i(y;G),
\]

一个可审计的上界可以写成

\[
|\Delta_y(G')-\Delta_y(G)|
\leq \sum_i |s'_i-s_i|\,|q_i(y;G)|
 + \sum_i \min(s_i,s'_i)|q_i(y;G')-q_i(y;G)|.
\]

第二项必须由新反馈或保守上界支持。若旧 margin 大于该半径，直接复用是安全的。否则触发复检。

## 候选算法 CTDR

1. 为每条边和每个可能答案维护带符号信用区间。
2. 计算图编辑造成的 source-TV 漂移和已审计 relay-row 漂移。
3. 对仍跨越 keep/remove 阈值的标签计算 decision risk。
4. 以 `decision risk × graph exposure / next-query-cost` 排序反馈请求。
5. 如果所有标签均有安全 action-region 归属，直接复用旧 transcript。
6. 如果一个标签被确认，仍继续检查其他标签是否会导出相反图编辑。记录单标签服务目标与 all-label reusable audit 的区别。

## 必须通过的证据门

| 门 | 要求 |
|---|---|
| 理论 | 给出 safe-transport theorem、tightness/counterexample 和无漂移退化到 stale 的证明 |
| 算法 | CTDR 在相同初始化、反馈预算和模型调用下优于 stale、random、periodic、max-uncertainty、source-TV、oracle 六类对照 |
| 任务 | 至少三种任务，MCQA、数值 QA、抽取 QA。每个任务至少三个 checkpoint 或 seed |
| 主指标 | 最终图编辑 action accuracy、误删、误保、恢复率、逻辑 token 和服务收益。不得只报 certificate MSE 或 retrospective lower bound |
| 消融 | 去掉 graph drift、decision margin、row-level TV、terminal uncertainty、transcript reuse |
| 反例 | 人为构造 margin 小而漂移大、margin 大而漂移小、单标签容易但 all-label 不兼容的实例 |
| 独立性 | 新题目不用于策略调参。角色轮换只算一次独立题目 |

## 新颖性矩阵

| 方向 | 现有工作覆盖 | MAB2 可声称的差异 |
|---|---|---|
| SBFE | 付费测试、随机 Boolean 函数、adaptive greedy | 不是新 SBFE 算法，而是将 graph-edit drift 作为证书可传输性的条件变量 |
| Decision Region Determination | 测试直到落入一个决策区域 | MAB2 的区域属于 signed edge-credit action，区域在图编辑后迁移 |
| 动态拓扑 | AgentDropout、AMAS、GTD 学习或选择拓扑 | 现有工作通常直接优化新拓扑，MAB2 研究旧反馈何时仍可用及如何修复 |
| 因果边归因 | EIB-Learner、E2-Explainer、CATS | 不再宣称因果归因新颖性。差异是证书传输与预算化再验证 |
| runtime message pruning | MessageSieve 按消息实例裁剪 | MAB2 的决策是保留/删除通信边的证书维护，需用同成本 action utility 对照 |

## 研究停止条件

若 CTDR 在三任务上不能在固定预算同时降低错误图编辑和反馈成本，方向应降级为机制观察论文。不能用 retrospective gold-optimum 或已暴露开发题目挽救算法门。
