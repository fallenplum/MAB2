# M60 strong baseline public summary

All methods share the same old transcript, post-edit source, legal action prices, graph-manifest affected-row set, predeclared budget and exact-decision stopping target.

| method | exact set | action error | incomplete | tokens | budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| stale | 0.9039 | 0.0428 | 0 | 0.00 | 1924.96 |
| exact_warm_cache | 0.8877 | 0.0443 | 0 | 0.00 | 1924.96 |
| reweighted_old_history | 0.9167 | 0.0391 | 0 | 22.86 | 1924.96 |
| ctdr_transport_only | 0.9167 | 0.0391 | 0 | 22.86 | 1924.96 |
| ctdr_transport_selective_ranking | 0.9167 | 0.0391 | 0 | 24.61 | 1924.96 |
| full_ctdr | 1.0000 | 0.0000 | 0 | 230.91 | 1924.96 |
| max_uncertainty | 0.9167 | 0.0391 | 0 | 22.86 | 1924.96 |
| periodic | 0.9167 | 0.0391 | 0 | 39.81 | 1924.96 |
| random | 0.9167 | 0.0391 | 0 | 37.34 | 1924.96 |
| full_restart | 1.0000 | 0.0000 | 0 | 878.78 | 1924.96 |
| oracle_row_refresh | 0.9068 | 0.0391 | 0 | 42.71 | 1924.96 |

The restricted oracle is an offline changed-row action diagnostic. It uses sealed target labels only to select a plan, then independently replays that plan. It is not a deployable or global optimum.

Per-panel and per-density values are in the JSON summary.

