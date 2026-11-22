# Independent deduplicated ARCC evidence

This table uses the corrected worker that evaluates identical prompts once before restoring the original row positions. Earlier non-deduplicated outputs are excluded because batch-position effects changed repeated prompts. The radius is fixed at (q=0.9) from the development split.

## Aggregate across ten independent cells

| controller | decision exact | answer-action exact | answer accuracy | relay tokens | queries |
|---|---:|---:|---:|---:|---:|
| stale | 0.9203 | 0.9164 | 0.2539 | 0.00 | 0.000 |
| TCR guard | 1.0000 | 1.0000 | 0.2719 | 276.75 | 2.460 |
| cold restart | 1.0000 | 1.0000 | 0.2719 | 1396.93 | 12.974 |
| answer-region, changed-row mask | 0.9968 | 1.0000 | 0.3161 | 142.60 | 1.068 |
| answer-region, all rows | 0.9968 | 1.0000 | 0.3161 | 162.29 | 1.184 |

The all-row controller reduces relay tokens by 41.4% relative to TCR guard while preserving the final answer action in every independent cell. It does not improve model knowledge accuracy. One Qwen-14B four-row cell has decision exactness 0.9677, so the claim is limited to action safety and cost reduction.

## Cell boundary

| cell | stale action | TCR tokens | all-row action | all-row decision | all-row tokens |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 rows1 | 0.9453 | 186.14 | 1.0000 | 1.0000 | 163.55 |
| Qwen2.5 rows2 | 0.9453 | 186.14 | 1.0000 | 1.0000 | 163.55 |
| Qwen2.5 rows4 | 0.7969 | 489.23 | 1.0000 | 1.0000 | 428.26 |
| Qwen2.5 interface | 0.9453 | 186.14 | 1.0000 | 1.0000 | 163.55 |
| Qwen-14B rows1 | 0.7813 | 151.38 | 1.0000 | 1.0000 | 112.65 |
| Qwen-14B rows2 | 1.0000 | 166.27 | 1.0000 | 1.0000 | 5.19 |
| Qwen-14B rows4 | 0.8203 | 453.40 | 1.0000 | 0.9677 | 313.29 |
| Qwen-14B interface | 1.0000 | 146.20 | 1.0000 | 1.0000 | 0.00 |
| Llama-3.2-1B rows1 | 0.9766 | 378.89 | 1.0000 | 1.0000 | 135.03 |
| Llama-3.2-1B interface | 0.9531 | 423.73 | 1.0000 | 1.0000 | 137.87 |

The interface cells are included to test the boundary of graph-local masks. The all-row fallback remains exact for answer actions under these cells, whereas the Qwen-14B four-row cell shows why edge-decision exactness must be reported separately.

