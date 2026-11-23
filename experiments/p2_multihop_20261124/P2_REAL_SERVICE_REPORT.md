# P2 真实多跳服务实验报告

## Protocol

- Model: server-local Qwen-0.5B, `MAB2_MICROBATCH=1`.
- Primary panel: 32 fixed SHA256-sorted QASC rows, split into 16 development and 16 held-out rows.
- Online methods share the old transcript, source edit, graph manifest, stopping rule and model checkpoint. Sealed answers are used only for scoring.
- Old-cache replay disables all post-edit structured-interface suffixes. Terminal decisions aggregate all sixteen terminal cells with weight `P(S0=i) * P(R_i=j)`.
- Full-information VOI is an offline diagnostic.

## Corrected primary result

| split | method | answer accuracy | action exact | forward calls | input tokens | wall seconds |
|---|---|---:|---:|---:|---:|---:|
| development | old graph reuse | 0.3750 | 1.0000 | 0.00 | 0.0 | 0.000 |
| development | exact cache | 0.3750 | 1.0000 | 2.00 | 561.6 | 0.035 |
| development | graph-mask recompute | 0.3750 | 1.0000 | 18.00 | 3877.6 | 0.310 |
| development | topology VOI | 0.3750 | 1.0000 | 12.81 | 2810.2 | 0.221 |
| development | full new graph | 0.3750 | 1.0000 | 21.00 | 4279.6 | 0.362 |
| held-out | old graph reuse | 0.3750 | 1.0000 | 0.00 | 0.0 | 0.000 |
| held-out | exact cache | 0.3750 | 1.0000 | 2.00 | 555.1 | 0.034 |
| held-out | graph-mask recompute | 0.3750 | 1.0000 | 18.00 | 3827.1 | 0.304 |
| held-out | topology VOI | 0.3750 | 1.0000 | 13.75 | 2956.1 | 0.233 |
| held-out | full new graph | 0.3750 | 1.0000 | 21.00 | 4222.4 | 0.355 |

The corrected 32-row primary panel shows a service-cost reduction at unchanged action quality. Topology VOI uses 13.28 calls on average across the two splits, compared with 18 for graph-mask recomputation and 21 for full graph execution. It does not improve answer accuracy.

Single-model and RAG controls remain weaker on held-out action exactness at 0.500 and answer accuracy at 0.125.

## Corrected exploratory 128-row extension

The fixed hash-sorted 128-row QASC extension was rerun after the old-cache permission fix. Development and held-out summaries are identical because the extension is reported as one repeated 128-row panel.

| method | answer accuracy | action exact | forward calls |
|---|---:|---:|---:|
| old graph reuse | 0.3828 | 0.9688 | 0.00 |
| exact cache | 0.3828 | 0.9688 | 2.00 |
| graph-mask recompute | 0.3828 | 1.0000 | 18.00 |
| topology VOI | 0.3828 | 1.0000 | 13.17 |
| full new graph | 0.3828 | 1.0000 | 21.00 |

The 128-row extension reproduces the cost–action pattern. It is exploratory because it reuses the same QASC cohort and one checkpoint.

## Scope

The primary result supports a small-panel service-cost reduction under unchanged action quality. It does not establish a general task-accuracy gain, a universal topology advantage, or a production latency claim. Raw inputs, sealed keys and per-row outputs remain local/private.
