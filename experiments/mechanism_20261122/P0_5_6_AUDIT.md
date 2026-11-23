# P0-5 manifest-matched Pareto audit

All methods share the same graph-manifest affected-row set, old transcript, post-edit source, token prices, full-restart budget and exact-set stopping target.

| method | exact set | action error | tokens | Pareto |
| --- | ---: | ---: | ---: | --- |
| stale | 0.9039 | 0.0428 | 0.00 | yes |
| exact_warm_cache | 0.8877 | 0.0443 | 0.00 | no |
| reweighted_old_history | 0.9167 | 0.0391 | 22.86 | yes |
| ctdr_transport_only | 0.9167 | 0.0391 | 22.86 | yes |
| ctdr_transport_selective_ranking | 0.9167 | 0.0391 | 24.61 | no |
| full_ctdr | 1.0000 | 0.0000 | 230.91 | yes |
| max_uncertainty | 0.9167 | 0.0391 | 22.86 | yes |
| periodic | 0.9167 | 0.0391 | 39.81 | no |
| random | 0.9167 | 0.0391 | 37.34 | no |
| full_restart | 1.0000 | 0.0000 | 878.78 | no |
| oracle_row_refresh | 0.9068 | 0.0391 | 42.71 | no |

Budget violations: 0.

# P0-6 decision and task-quality transition audit

The transition counts below refer to sealed decision-target agreement. They are not task-answer accuracy.

| baseline | corrected | damaged | unchanged correct | unchanged wrong | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact_warm_cache | 153 | 0 | 3303 | 0 | 3456 |
| reweighted_old_history | 135 | 0 | 3321 | 0 | 3456 |
| max_uncertainty | 135 | 0 | 3321 | 0 | 3456 |
| periodic | 135 | 0 | 3321 | 0 | 3456 |
| random | 135 | 0 | 3321 | 0 | 3456 |

Per-question answer correctness transitions are computed separately from private WORKER*_score_auto.json rows and published in P0_6_TASK_QUALITY.md. They are not inferred from decision-set transitions.

