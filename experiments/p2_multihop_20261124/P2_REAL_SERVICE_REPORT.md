# P2 真实多跳服务实验报告

## Protocol

- Model: server-local Qwen-0.5B, `MAB2_MICROBATCH=1`.
- Primary panel: 32 fixed SHA256-sorted QASC rows, split into 16 development and 16 held-out rows.
- Online methods share the old transcript, source edit, graph manifest, stopping rule and model checkpoint. Sealed answers are used only for scoring.
- Terminal decisions aggregate all sixteen terminal cells with weight `P(S0=i) * P(R_i=j)`. The earlier single-argmax implementation is quarantined and excluded from this report.
- Full-information VOI is an offline diagnostic.

## Primary result

| split | method | answer accuracy | action exact | forward calls | input tokens | wall seconds |
|---|---|---:|---:|---:|---:|---:|
| development | old graph reuse | 0.3750 | 0.9375 | 0.00 | 0.0 | 0.000 |
| development | exact cache | 0.3750 | 0.9375 | 2.00 | 561.6 | 0.035 |
| development | graph-mask recompute | 0.3750 | 1.0000 | 18.00 | 3877.6 | 0.316 |
| development | topology VOI | 0.3750 | 1.0000 | 11.63 | 2566.3 | 0.204 |
| development | full new graph | 0.3750 | 1.0000 | 21.00 | 4279.6 | 0.368 |
| held-out | old graph reuse | 0.3750 | 1.0000 | 0.00 | 0.0 | 0.000 |
| held-out | exact cache | 0.3750 | 1.0000 | 2.00 | 555.1 | 0.035 |
| held-out | graph-mask recompute | 0.3750 | 1.0000 | 18.00 | 3827.1 | 0.316 |
| held-out | topology VOI | 0.3750 | 1.0000 | 11.88 | 2571.1 | 0.208 |
| held-out | full new graph | 0.3750 | 1.0000 | 21.00 | 4222.4 | 0.368 |

Across the primary 32 rows, topology VOI keeps exact action agreement at 1.0000, while reducing forward calls from 18.00 for graph-mask recompute to 11.75. It does not improve answer accuracy because every graph method has the same underlying Qwen-0.5B answer accuracy on this panel.

## Additional exploratory extension

A fixed hash-sorted 128-row extension gives action exactness 0.9844 for topology VOI and 1.0000 for graph-mask recompute. It is retained as exploratory evidence and does not replace the locked 32-row primary result.

## Scope

The primary result supports a service-cost reduction under unchanged action quality on this small panel. It does not establish a general task-accuracy gain, a universal topology advantage, or a production latency claim. Raw inputs, sealed keys and per-row outputs remain local/private.
