# Qwen2.5-3B three-task conditional-drift pilot

This artifact records one-model replay on three disjoint 128-question panels (QASC, SciQ and Commonsense, 384 questions total). It is exploratory and is not independent multi-model confirmation.

All methods share the same model outputs, old transcript and post-edit replay scorer. Gold labels are used only for reporting.

| method | exact-set | action-error | queries | token cost |
|---|---:|---:|---:|---:|
| stale | 92.45% | 2.60% | 0.000 | 0.000 |
| warm_reweighted | 97.92% | 0.78% | 0.688 | 67.599 |
| source_only_ctdr | 97.92% | 0.78% | 0.688 | 67.599 |
| tcr_guard | 100.00% | 0.00% | 1.651 | 156.734 |
| tcr_token | 100.00% | 0.00% | 1.651 | 156.612 |
| cold_restart | 100.00% | 0.00% | 2.523 | 260.500 |
| cold_token | 100.00% | 0.00% | 2.516 | 259.573 |

The single-model pilot keeps the conditional relay revalidation direction viable. TCR guard/token recovered the finite decision set on all 384 questions and used fewer replay tokens than cold restart. This result cannot support an ACL Main claim without independent model families or checkpoints, graph-delta-only localization stress tests, and real-drift component ablations.
