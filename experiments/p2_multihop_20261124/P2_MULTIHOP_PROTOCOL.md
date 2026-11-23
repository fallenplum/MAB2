# P2 小规模真实多跳服务实验协议

## 目的

P2 只在 P1 受控 DAG 证明拓扑信息有额外收益后启动。目标是把 CPU 上的拓扑 VOI 结果映射到真实自然语言多跳服务，不把模拟 token 节约当成服务收益。

## 数据和任务

使用本地 QASC 两事实输入 `MAB2_recovered/qasc_rows2_INPUT.json`。准备脚本按 `sha256(key)` 固定选择 32 条问题，前 16 条用于开发选择，后 16 条作为小规模独立留出。每条问题保留原始题干、8 个选项、source fact、old relay fact、new relay facts 和 control fact。答案键只从私有 `SEALED.json` 读取，不写入公开输入。

QASC 的两个事实构成可审计的自然语言两跳证据链。实验把它显式表示为

```text
S0 source reviewer
  -> R0 relay aggregator
  -> R1/R2/R3 hypothesis-specific relays
  -> T terminal answer selector
```

`R1/R2/R3` 同时接收 `S0` 的结构化输出和对应原始 relay evidence。`T` 接收四个 relay 输出以及原始选项。每条记录同时保存 `raw_evidence` 和 `structured_interface`，后者只规定节点依赖、假设槽位和返回格式，不替代自然语言证据。

## 方法

- `full_new_graph` 重新执行全部 source、relay 和 terminal 节点。
- `old_graph_reuse` 完全沿用旧图输出，不购买编辑后反馈。
- `single_model` 直接将题干和证据交给一个模型，不使用图协同。
- `retrieval_augmented_generation` 使用固定检索器选取两条证据，再由单模型回答。
- `exact_cache` 复用未改变节点，精确重算改变节点。
- `graph_mask_recompute` 重算图清单中改变 relay 的所有后代。
- `topology_voi_online` 根据决策区域未定程度、节点后代集合和实际 token 费用选择下一次调用。它只能使用旧 transcript、公开图清单和当前新图输出。
- `full_information_voi_upper_bound` 使用封存答案键做离线收益上界分析，只用于界定可达到的收益空间，不作为在线基线。

## 公平性和信息权限

所有在线方法使用相同问题、模型 checkpoint、旧 transcript、source edit、预算、停止目标和 tokenizer。封存答案键只用于最终评分。在线方法不能读取封存标签来决定查询动作。检索器、图掩码和 VOI 选择器的运行时间分别记账。

每个方法记录 `forward_calls`、`input_tokens`、`output_tokens`、`wall_seconds`、`cache_hits` 和 `retrieval_calls`。模型加载时间单独记录，不混入方法调用时间。最终质量记录 `answer_accuracy`、`answer_action_exact`、`decision_exact`、`false_drop`、`false_keep` 和 `recovery_rate`。

## 运行条件

首个真实服务 checkpoint 使用服务器已有的 Qwen-0.5B，历史入口路径为 `/home/chen/KDBPT/KD/models/Qwen-0.5B`。执行脚本必须支持 `MAB2_MICROBATCH=1`，避免批位置差异混入方法比较。开发阶段只允许使用前 16 条，后 16 条只用于一次独立报告。

## P1 到 P2 的门槛

如果 P1 中 `topology_voi_online` 相对 exact cache、graph-mask recompute 和 Pairwise 在相同风险目标下没有额外成本收益，P2 只保留协议和输入准备，不启动真实模型调用。如果 P1 通过，则先在 16 条开发题上固定所有超参数，再在 16 条留出题上运行一次全方法矩阵。

P2 的成功标准不是某个模型的绝对准确率，而是在相同任务质量或相同 action-risk 约束下，真实 forward 调用、输入 token、wall time 或总服务成本有可重复下降。若只节省逻辑 relay token，不能宣称端到端服务收益。

## 产物

- `P2_MULTIHOP_INPUT.json` 保存脱敏题目和完整原始证据。
- `P2_MULTIHOP_MANIFEST.json` 保存固定选择、DAG、方法、信息权限和成本字段。
- 私有 scorer 输出逐题质量、调用账本和失败记录。
- 最终报告区分开发结果、独立留出结果和离线 VOI 上界。

当前状态为 `prepared_only_pending_P1_gain_gate`。
