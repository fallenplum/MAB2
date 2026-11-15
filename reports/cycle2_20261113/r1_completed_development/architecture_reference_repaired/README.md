# 架构参照 A/B 语义修复入口

这是已曝光描述性结果的诊断修复，不是新科学实验。原 R1 协议、门槛、主结果、购买和动作均不变。原 source/protocol/test/aggregate result 已按字节保存在 original_private，原文件和 run_001 均不修改。

旧错误：原代码从 R1 S 数组轴 0 取得 B，却标为 A_stay；轴 1 取得 A，却标为 B_stay。新实现明确绑定 `change→B→axis0`、`no_change→A→axis1`，不得按 A/B 字母顺序猜轴。

修复入口要求 root 先将 REPAIR_FREEZE_TEMPLATE.json 审核冻结为单独的 REPAIR_FREEZE.json，状态精确为 `FROZEN_DIAGNOSTIC_REPAIR_AFTER_EXPOSURE`，保留 source SHA。模板本身不能启动真实重评分。本包仅完成 CPU 合成测试，尚未读真实标签或原始模型文本进行修复。

```text
python -B repair_reference.py --freeze REPAIR_FREEZE.json --repo <frozen-repo> --plan <frozen-plan> --descriptor <owner-phase-descriptor> --run-root <original-raw-run> --analysis <completed-R1-analysis> --independent-review <PASS-replay-json> --old-run <architecture-run_001> --out <new-Cstage-path>/run_002
```

输入全部只读。新输出必须叫 run_002 且尚不存在。输出是私有诊断目录：PRIVATE_ARRAYS.npz、SOURCE_GRAPH_PINS.json 不发布；只有脱敏汇总可另行列白名单。禁止把整个运行目录直接公开。

程序对全部 192 题分别读取原 A/B stay graph，共 384 图，按原完整 output freeze 的 SHA 校验原始字节，校验 request/background/mask/seed/question 绑定。原 frozen scorer 重算每个 aggregate，与各自显式 S_quality 轴逐格相等；重新加总每图全部 9 节点 input/output/calls，与 S_cost 对应轴逐 episode 相等。只有全部通过才输出修复后的 A/B 参照。

五个首轮/投票参照仍使用原辅助函数及 scorer，不变更投票规则；同时核对其全部汇总与原结果相同。固定 4 个对比仍用原 8-episode bootstrap 10000 次、seed95173。预计 CPU 评分调用为 960+384=1344，新增模型调用/GPU/正式轮次均为 0；已支付生成费不重复记录为新 GPU 消耗。

CPU 回归为原 5 项投票测试及新增 6 项 A/B 哨兵测试。后者令 A/B 质量和 token 均不同，旧轴序解释会明确失败，也覆盖只交换标签却没有对应费用、单图错误及单 episode 费用错误。原测试报告只证明这些合成路径，不代表 384 真实图已核验。

失败保留 FAILED_RECEIPT 和所有已有输出，不覆盖原证据；不得通过忽略 mismatch 或扩大 tolerance 使修复通过。
