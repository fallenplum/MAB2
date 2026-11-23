# P2 小规模真实多跳服务结果

## 执行范围

服务器使用 `/data/my_fed_attack/my_envs/ame/bin/python`、Qwen-0.5B checkpoint `/home/chen/KDBPT/KD/models/Qwen-0.5B` 和 `MAB2_MICROBATCH=1`。8 卡中使用了 2、3、6、7 四张空闲卡，开发和留出各 16 条题目，每条题目独立运行。旧 transcript 由每题旧图的 source、relay 和 terminal 输出组成，方法成本只统计编辑后的增量调用。答案键只在服务器私有 scorer 中使用，公开汇总不包含题目键、目标或逐题预测。

## QASC 两事实多跳输入

| split | method | answer accuracy | action exact | forward/row | input tokens/row | wall s/row |
|---|---|---:|---:|---:|---:|---:|
| development | full new graph | 0.3750 | 1.0000 | 21.000 | 4279.56 | 0.3678 |
| development | old graph reuse | 0.3750 | 1.0000 | 0.000 | 0.00 | 0.0000 |
| development | exact cache | 0.3750 | 1.0000 | 2.000 | 561.62 | 0.0346 |
| development | graph-mask recompute | 0.3750 | 1.0000 | 18.000 | 3877.62 | 0.3160 |
| development | topology VOI online | 0.3750 | 1.0000 | 12.812 | 2810.19 | 0.2210 |
| held-out | full new graph | 0.3750 | 1.0000 | 21.000 | 4222.38 | 0.3684 |
| held-out | old graph reuse | 0.3750 | 1.0000 | 0.000 | 0.00 | 0.0000 |
| held-out | exact cache | 0.3750 | 1.0000 | 2.000 | 555.12 | 0.0342 |
| held-out | graph-mask recompute | 0.3750 | 1.0000 | 18.000 | 3827.12 | 0.3158 |
| held-out | topology VOI online | 0.3750 | 1.0000 | 13.750 | 2956.12 | 0.2331 |

在 QASC 留出集上，拓扑 VOI 保持与 graph-mask 和 full graph 相同的 action exact，同时将平均调用降至 13.281。修正 old-cache 权限后，开发和留出集的 old reuse 与 exact cache 均达到 action exact 1.0。该结果支持小规模服务成本收益，但题目数为 16，不能作为跨任务或跨 checkpoint 的充分验证。

单模型和 RAG 对照在留出集上的 answer accuracy 均为 0.125，action exact 均为 0.500。它们没有达到图方法的决策一致性。

## 接口漂移压力面板

接口漂移输入保持事实和题目不变，只在一个 relay 分支加入结构化协议。开发与留出均为 16 条题目。

| split | method | answer accuracy | action exact | forward/row | input tokens/row |
|---|---|---:|---:|---:|---:|
| development | full new graph | 0.3750 | 1.0000 | 21.000 | 4032.56 |
| development | old graph reuse | 0.3750 | 1.0000 | 0.000 | 0.00 |
| development | exact cache | 0.3750 | 1.0000 | 1.000 | 164.81 |
| development | graph-mask recompute | 0.3750 | 1.0000 | 17.000 | 3480.81 |
| development | topology VOI online | 0.3750 | 1.0000 | 1.000 | 164.81 |
| held-out | full new graph | 0.3750 | 1.0000 | 21.000 | 3979.38 |
| held-out | old graph reuse | 0.3750 | 1.0000 | 0.000 | 0.00 |
| held-out | exact cache | 0.3750 | 1.0000 | 1.000 | 163.56 |
| held-out | graph-mask recompute | 0.3750 | 1.0000 | 17.000 | 3435.56 |
| held-out | topology VOI online | 0.3750 | 1.0000 | 1.000 | 163.56 |

接口压力面板没有造成最终 answer-action 改变，因此它只验证了该 runner 能记录结构化接口变化，不能支持接口漂移带来任务质量变化的主张。

## 信息权限与离线界限

`topology_voi_online` 只使用旧 transcript、当前新图输出、结构化 DAG 和预先固定的 0.08 影响阈值。它不读取封存答案。`full_information_voi_upper_bound` 在知道新图目标后计算离线收益上界，不能作为在线算法或可部署 baseline。

当前 P2 的正证据是成本层面的。QASC 留出集上拓扑 VOI 在相同 action exact 下比 graph-mask 少 6.125 次 forward，比 full graph 少 9.125 次 forward。它尚未证明最终答案准确率提升，也没有覆盖第二个 checkpoint 或第二个自然语言任务。

## 复现产物

- `P2_MULTIHOP_INPUT.json` 和 `P2_MULTIHOP_MANIFEST.json`
- `P2_INTERFACE_INPUT.json` 和 `P2_INTERFACE_MANIFEST.json`
- `p2_service_runner.py`
- `p2_logic.py`
- `aggregate_p2_service.py`
- `P2_WEIGHTED_PUBLIC_SUMMARY.json`
- `P2_INTERFACE_WEIGHTED_PUBLIC_SUMMARY.json`

状态为 `P2_SMALL_REAL_SERVICE_COMPLETED_WITH_SCOPE`。



