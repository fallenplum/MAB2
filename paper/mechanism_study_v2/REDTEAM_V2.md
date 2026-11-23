# V2 ACL Main 红队审查

## 判断

当前稿件已经形成一个可检验的机制论文主线，但还不能标记为 ACL Main ready。风险门控的离线 worker 证据较强，Qwen-0.5B 的单条服务成本结果和 Qwen-14B 的服务复算诊断也已经纳入。稿件仍缺少服务动作口径闭合、正式的证书命题、完整公平基线表和足够的文献覆盖。

## 已对齐的主张

- 风险门控保留最终答案动作，未宣称提高语言模型知识准确率。
- full edge-audit TCR 的 token 节约与 fair answer-action mask 的比较已经区分。
- edge consistency 被写成 worker reference 诊断，未直接当作低风险掩码的运行时 edge certificate。
- Qwen-14B service forward mismatch 已写入限制和服务主张边界。

## 必须修正

### P0 证书与算法口径

稿件把“风险门控选择路线”和“答案区间证书”放在同一叙述中，但低风险 mask 分支只刷新 relay 行，未购买所有下游 conditional cell。正文需要明确风险门控的正式停止目标是 answer-action。edge consistency 只能放到诊断表，不能进入方法的 exact certificate 结论。

经验半径 `delta_i` 是开发集分位数，不具备有限样本覆盖保证。正文需要把安全命题写成条件命题，显式列出 TV 半径假设和接口指纹完备假设。`q=.5` 选择在 Qwen-14B 三个密度上有效，不能写成通用最优量化值。

### P0 公平基线

当前稿件还没有完整展示 fair answer-only mask、risk gate、answer-region、full TCR 和 cold restart 的同一张表。至少要报告 action exact、answer accuracy、relay forwards、input tokens、logical relay tokens 和 wall time。full TCR 的 downstream cell 成本不能与 answer-only route 的 relay 成本直接汇总成一个“总成本”。

### P0 服务闭合

Qwen-0.5B 的单条 forward 测量显示 action exact 保持为 1.0，但 answer-region 没有降低 relay forwards。Qwen-14B 的 service JSON 记录了成本，却与 worker reference 存在动作复算差异。论文需要把 Qwen-14B service action 从主结果表移到诊断表，并在方法主表中使用 matched worker replay 口径。若要声称端到端服务收益，必须修复 worker 与 service 的完整 128 题一致性后再补表。

### P1 独立性描述

21 个 cell 由重复的 128 题在模型和密度条件下展开，不能称为 21 个独立数据集。正文应报告 unique question keys、task、model、density 四个重复因素，并避免把 cell 数直接当成样本量。

### P1 文献与新颖性

当前 Related Work 只有三条引用，无法支撑与 decision-region determination、adaptive testing、message routing、agent pruning、conditional distribution shift 和 certificate reuse 的边界。需要补充 Literature Tree 与 Novelty Matrix，并在正文明确风险门控的差异是 graph-conditioned certificate validity 和 interface-fingerprint invalidation。

### P1 论文完整性

当前 draft 没有结果表、消融图、风险阈值 frontier、接口漂移图或成本图。ACL Main 稿件至少需要一张主结果表、一张 risk-versus-margin 或 density 图、一张 interface drift failure/recovery 图，以及附录中的 per-cell 表。

## 数值核对

- Matched Qwen-14B worker rows1/2/4 risk relay tokens 为 19.65、112.65、337.94。
- Fair answer-only mask tokens 为 112.65、112.65、337.94，只有 rows1 有 82.6% relay-token reduction。
- Full TCR tokens 为 151.80、150.48、415.53，较大的 87.1%、25.1%、18.7% 差距必须标明 TCR 还承担 edge-audit target。
- Qwen-0.5B single-forward action exact 为 1.0，TCR forwards 为 1、1、3，answer-region q=1 forwards 为 1.31、1.31、3.02。
- Qwen-14B service action values必须按 diagnostic 处理，不能和 matched worker exact action 混排。

## 结论

风险门控的机制结果足以支撑继续冲击 ACL Main。当前稿件还不能提交，主要原因是 fair baseline 表、条件证书表述和 Qwen-14B service 口径尚未闭合。优先补齐这些内容，再决定是否把工作写成正向算法论文或机制边界论文。

