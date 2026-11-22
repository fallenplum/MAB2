# Qwen2.5-3B conditional-drift pilot

This artifact records a single-model exploratory replay on two disjoint 128-question panels (QASC and SciQ, 256 questions total). It is not a formal multi-team confirmation.

All methods use the same model outputs and the same post-edit replay scorer. Gold labels are used only for reporting.

| method | exact-set | action-error | queries | token cost |
|---|---:|---:|---:|---:|
| stale | 89.45% | 3.52% | 0.000 | 0.000 |
| warm_reweighted | 97.66% | 0.78% | 1.031 | 101.398 |
| source_only_ctdr | 97.66% | 0.78% | 1.031 | 101.398 |
| tcr_guard | 100.00% | 0.00% | 1.949 | 187.883 |
| tcr_token | 100.00% | 0.00% | 1.949 | 187.699 |
| cold_restart | 100.00% | 0.00% | 2.902 | 311.348 |
| cold_token | 100.00% | 0.00% | 2.898 | 310.598 |

The pilot supports continued testing of conditional relay revalidation because tcr_guard and tcr_token recovered the full finite decision set on this sample while using fewer replay tokens than cold restart. The result is exploratory and cannot be used as an ACL Main claim without independent models, tasks, and frozen evaluation.
