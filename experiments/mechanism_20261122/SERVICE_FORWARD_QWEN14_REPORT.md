# Qwen-14B 实际服务层 Forward 结果

服务器已有的 Qwen-14B 独立 ARCC rows1/2/4 forward 结果已同步。每个 JSON 记录了真实 model forward 次数、输入 token、输出 token、wall-clock 时间和动作指标。rows1 和 rows4 的当前 JSON 使用了与服务 forward 不同的微批量 worker 输出，动作指标存在批位置不一致风险，因此不进入主服务结论。

| 图密度 | forward/题 | 输入 token/题 | wall s/题 | decision exact | action exact | answer accuracy |
|---|---:|---:|---:|---:|---:|---:|
| rows1 | 1.000 | 102.99 | 0.0598 | 1.0000 | 0.7891 | 0.2344 |
| rows2 | 0.0078 | 1.26 | 0.0042 | 1.0000 | **1.0000** | 0.2500 |
| rows4 | 2.836 | 292.70 | 0.1637 | 0.8047 | 0.7031 | 0.2188 |

rows2 的图变化在匹配的独立 route 结果中只触发 1 次 relay forward，动作完全恢复。rows1 和 rows4 的表格只作为批位置差异诊断，不能用来推断服务动作质量。需要使用同一单条 worker 输出重新生成后，才能判断高密度迁移是否成立。

该面板当前只有 rows2 的 calibrated-route 结果满足同一输入口径。完整 TCR、cold restart 和 rows1/rows4 的匹配单条 worker 对照仍是候选路线的未闭合证据。

原始文件

- `service_q14_r1.json`
- `service_q14_r2.json`
- `service_q14_r4.json`
- `SERVICE_FORWARD_QWEN14_R2_CALIBRATED_ROUTE.json`

