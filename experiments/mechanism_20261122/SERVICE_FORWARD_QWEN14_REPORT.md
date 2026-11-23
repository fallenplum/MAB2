# Qwen-14B 实际服务层 Forward 结果

服务器已有的 Qwen-14B 独立 ARCC rows1/2/4 单条 forward worker 结果已同步。每个 JSON 记录了真实 model forward 次数、输入 token、输出 token、wall-clock 时间和动作指标。旧的 `service_q14_r1/r2/r4.json` 使用了微批量 worker，存在 batch-position mismatch，排除出正式口径。

| 图密度 | forward/题 | 输入 token/题 | wall s/题 | decision exact | action exact | answer accuracy |
|---|---:|---:|---:|---:|---:|---:|
| rows1 | 0.055 | 5.1 | 0.0050 | 0.9844 | **0.9922** | 0.2344 |
| rows2 | 0.992 | 102.0 | 0.0578 | 0.8281 | 0.7578 | 0.2344 |
| rows4 | 2.914 | 300.5 | 0.1675 | 0.9375 | 0.7188 | 0.2578 |

单条 worker 下，rows1 的 q=1 answer-region 只触发 7 次 forward，action exact 为 0.9922。rows2 触发 127 次，action exact 降至 0.7578。rows4 触发 373 次，action exact 为 0.7188。稀疏 rows1 的低成本现象不能外推到 rows2/rows4。

同一 worker 的 TCR 和 cold restart 对照也在 JSON 中。TCR action exact 分别为 0.9922、0.7578、0.7266。cold restart 使用 23 次 forward/题，输入 token 约为 1955–1961/题，action exact 分别为 0.6172、0.4453、0.4844。

## 口径说明

`service_q14_mb1_r1/r2/r4.json` 是当前服务层正式口径。早先的 `service_q14_r1/r2/r4.json` 保留作批位置差异诊断，不用于主结论。

原始文件

- `service_q14_mb1_r1.json`
- `service_q14_mb1_r2.json`
- `service_q14_mb1_r4.json`
- `SERVICE_FORWARD_QWEN14_R2_CALIBRATED_ROUTE.json`

Decision exactness and action exactness here compare a single-forward replay
with the worker's recorded full new-graph output. They do not replace an
independent gold-answer quality metric. Qwen-14B still shows forward replay
differences under the custom model path, so its call count, input-token count
and wall-clock values are retained as service-cost diagnostics. The action
metrics are excluded from the primary method claim until this model-side
replay discrepancy is resolved.

