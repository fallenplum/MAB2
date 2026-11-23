# M60 strong-baseline interpretation

The 3,456-unit CPU replay gives a single matched ledger for all controllers.
Every method receives the same old transcript, post-edit source, token prices,
and stopping target. Costs are post-edit incremental tokens.

| controller | exact set | action error | tokens |
| --- | ---: | ---: | ---: |
| exact warm cache | 0.8877 | 0.0443 | 0.00 |
| reweighted old history | 0.9167 | 0.0391 | 22.86 |
| CTDR transport only | 0.9167 | 0.0391 | 22.86 |
| CTDR + selective ranking | 0.9167 | 0.0391 | 24.61 |
| full CTDR | 1.0000 | 0.0000 | 230.91 |
| max uncertainty | 0.9167 | 0.0391 | 22.86 |
| periodic | 0.9167 | 0.0391 | 39.81 |
| random | 0.9167 | 0.0391 | 37.34 |
| full restart | 1.0000 | 0.0000 | 878.78 |
| restricted oracle row refresh | 0.9068 | 0.0391 | 42.71 |

The full CTDR route saves 73.7% of full-restart tokens while recovering the
exact decision set. It does not reduce cost relative to reweighted replay or
the restricted row-refresh oracle. Selective ranking adds 1.75 tokens to the
transport-only route without improving its exactness. The matrix therefore
supports a conservative certificate-transport trade-off and an action-error
reduction relative to warm/reweighted controls. It does not support a claim
that CTDR is the cost-optimal or independently superior feedback scheduler.

The restricted oracle is an offline hindsight diagnostic over changed-row actions. It reaches 0.9068 exact-set recovery after the permission and unfinished-plan fixes, so it is not a deployable method or an exact global lower bound. The exact warm cache and stale routes are
kept separate because they make different transcript assumptions. These
results should replace any earlier comparison that used only cold restart as
the baseline.

