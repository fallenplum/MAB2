# M60 strong baseline matrix

All controllers receive the same old certificate transcript, post-edit source, relay outputs, token prices and stopping target. Costs are post-edit incremental query tokens only.

| method | exact set | action error | false drop | false keep | queries | tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stale | 0.9039 | 0.0428 | 0.0637 | 0.0576 | 0.0000 | 0.00 |
| exact_warm_cache | 0.8877 | 0.0443 | 0.0894 | 0.0408 | 0.0000 | 0.00 |
| reweighted_old_history | 0.9167 | 0.0391 | 0.0521 | 0.0547 | 0.2801 | 22.86 |
| ctdr_transport_only | 0.9167 | 0.0391 | 0.0521 | 0.0547 | 0.2801 | 22.86 |
| ctdr_transport_selective_ranking | 0.9167 | 0.0391 | 0.0521 | 0.0547 | 0.3084 | 24.61 |
| full_ctdr | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0987 | 230.91 |
| max_uncertainty | 0.9167 | 0.0391 | 0.0521 | 0.0547 | 0.2801 | 22.86 |
| periodic | 0.9167 | 0.0391 | 0.0521 | 0.0547 | 0.4907 | 39.81 |
| random | 0.9167 | 0.0391 | 0.0521 | 0.0547 | 0.4716 | 37.34 |
| full_restart | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 11.3698 | 878.78 |
| oracle_row_refresh | 0.9068 | 0.0391 | 0.0932 | 0.0000 | 0.5486 | 42.71 |

## Oracle interpretation

`oracle_row_refresh` is an offline row-refresh optimum. It is evaluated only over changed-row actions, so it is a reproducible restricted oracle rather than a global optimum over every cell-level schedule. `oracle_lower_bound_tokens` is a valid necessary-cost lower bound when the exact warm cache is wrong, and zero when the warm cache already matches the target.

## Attribution

The matrix distinguishes exact warm-cache reuse from reweighted old-history replay. `ctdr_transport_only` isolates the transport gate, `ctdr_transport_selective_ranking` adds token-aware ordering, and `full_ctdr` uses graph-conditioned changed-row revalidation plus selective ranking.
