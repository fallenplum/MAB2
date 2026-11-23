# Qwen-0.5B 实际服务层 Forward Benchmark

## 实验定义

本实验在服务器 `/home/chen/KDBPT/KD/models/Qwen-0.5B` 上使用独立 ARCC 输入的 128 个问题，分别运行 rows1、rows2、rows4 三种图变化密度。每个方法都实际调用模型 forward，并记录调用数、输入 token、单次输出 token 和 wall-clock 时间。模型加载时间未计入方法时间。

- `stale` 复用历史 relay 和终端输出，不发起编辑后 forward。
- `tcr_guard` 只对 worker 提供的 changed relay rows 重新 forward。Qwen-0.5B 的已有完整 scorer 显示 rows1/2 平均 1 个 action、rows4 平均 3 个 action，和本实验的调用数一致，没有额外 cell action。
- `answer_region_q1` 使用 all-row answer-region policy，半径在校准子集上取 q=1。未改变的行半径为 0，属于接口不变先验。
- `cold_restart` 重新 forward 3 个 source、4 个 new relay 和 16 个 terminal，共 23 次调用/题。

## 主要结果

### 正式 q=1 对照

| 图密度 | 方法 | forward/题 | 输入 token/题 | wall s/题 | action exact | answer accuracy |
|---|---|---:|---:|---:|---:|---:|
| rows1 | stale | 0.00 | 0.0 | 0.0053 | 0.8594 | 0.2500 |
| rows1 | TCR guard | 1.00 | 132.0 | 0.0232 | **0.8828** | 0.2422 |
| rows1 | answer-region q=1 | 2.89 | 371.4 | 0.0511 | 0.8359 | 0.2422 |
| rows1 | cold restart | 23.00 | 2621.8 | 0.3657 | 0.6406 | 0.2344 |
| rows2 | stale | 0.00 | 0.0 | 0.0052 | 0.8594 | 0.2500 |
| rows2 | TCR guard | 1.00 | 132.0 | 0.0282 | **0.8828** | 0.2422 |
| rows2 | answer-region q=1 | 2.89 | 371.4 | 0.0667 | 0.8359 | 0.2422 |
| rows2 | cold restart | 23.00 | 2621.8 | 0.4965 | 0.6406 | 0.2344 |
| rows4 | stale | 0.00 | 0.0 | 0.0054 | **0.7891** | 0.2500 |
| rows4 | TCR guard | 3.00 | 396.0 | 0.0559 | 0.7813 | 0.2813 |
| rows4 | answer-region q=1 | 3.91 | 508.7 | 0.0689 | 0.7734 | 0.2734 |
| rows4 | cold restart | 23.00 | 2627.8 | 0.3765 | 0.6406 | **0.3047** |

## q-frontier

降低 q 只减少调用和延迟，没有恢复 action exact。rows1/rows2 的 action exact 从 q=.5/.75 的 0.8438 降到 q=.9 的 0.8359。rows4 在 q=.5、.75、.9 均为 0.7734。

| 密度 | q | forward/题 | 输入 token/题 | wall s/题 | action exact | answer accuracy |
|---|---:|---:|---:|---:|---:|---:|
| rows1/2 | .50 | 2.39 | 306.2 | 0.044–0.046 | 0.8438 | 0.2422 |
| rows1/2 | .75 | 2.49 | 319.8 | 0.046–0.050 | 0.8438 | 0.2422 |
| rows1/2 | .90 | 2.67 | 343.1 | 0.050 | 0.8359 | 0.2422 |
| rows4 | .50 | 3.55 | 460.6 | 0.0615 | 0.7734 | 0.2734 |
| rows4 | .75 | 3.68 | 478.4 | 0.0646 | 0.7734 | 0.2734 |
| rows4 | .90 | 3.72 | 483.3 | 0.0640 | 0.7734 | 0.2734 |

## 解释

这组结果是当前候选方法的**候选负证据**。在 Qwen-0.5B 上，图条件 answer-region policy 的实际调用数高于 TCR guard，action exact 低于 TCR guard。它只相对 cold restart 显著减少了调用和延迟，不能证明相对强 warm baseline 的独立收益。q-frontier 也没有出现质量–成本拐点。

因此，论文不能把 q=1 answer-region 作为跨模型的普适服务算法。后续应优先检验更大模型上的 calibrated route 是否能降低调用数并保持 action safety，或者将该结果作为模型规模依赖的边界机制。

## 单条 forward 口径修正

随后发现，前述正式表使用的 worker 输出是在微批量执行下生成的，而服务 benchmark 按单条 forward 重新计算。为避免批位置差异混入方法比较，独立输入已按 `MAB2_MICROBATCH=1` 重跑。单条口径的 Qwen-0.5B 结果如下。

| 图密度 | 方法 | forward/题 | 输入 token/题 | wall s/题 | action exact | answer accuracy |
|---|---|---:|---:|---:|---:|---:|
| rows1 | stale | 0.00 | 0.0 | 0.0053 | 0.9062 | 0.2266 |
| rows1 | TCR guard | 1.00 | 132.0 | 0.0280 | 1.0000 | 0.2344 |
| rows1 | answer-region q=1 | 1.31 | 175.8 | 0.0331 | 1.0000 | 0.2344 |
| rows1 | cold restart | 23.00 | 2621.8 | 0.4921 | 1.0000 | 0.2344 |
| rows2 | stale | 0.00 | 0.0 | 0.0054 | 0.9062 | 0.2266 |
| rows2 | TCR guard | 1.00 | 132.0 | 0.0230 | 1.0000 | 0.2344 |
| rows2 | answer-region q=1 | 1.31 | 175.8 | 0.0262 | 1.0000 | 0.2344 |
| rows2 | cold restart | 23.00 | 2621.8 | 0.3602 | 1.0000 | 0.2344 |
| rows4 | stale | 0.00 | 0.0 | 0.0054 | 0.7578 | 0.2266 |
| rows4 | TCR guard | 3.00 | 396.0 | 0.0562 | 1.0000 | 0.3047 |
| rows4 | answer-region q=1 | 3.02 | 398.6 | 0.0551 | 1.0000 | 0.3047 |
| rows4 | cold restart | 23.00 | 2627.8 | 0.3810 | 1.0000 | 0.3047 |

The corrected single-call benchmark removes the earlier apparent action gap. It also shows that answer-region q=1 does not reduce actual relay forward calls below TCR on this model. The earlier q-frontier table remains a batch-mismatch diagnostic and is excluded from the main service claim.

原始 JSON 文件

- `SERVICE_FORWARD_QWEN05_R1.json`
- `SERVICE_FORWARD_QWEN05_R2.json`
- `SERVICE_FORWARD_QWEN05_R4.json`
- `SERVICE_FORWARD_QWEN05_R{1,2,4}_Q{0.5,0.75,0.9}.json`

