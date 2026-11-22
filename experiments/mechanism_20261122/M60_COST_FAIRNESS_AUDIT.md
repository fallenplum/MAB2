# M60 cost fairness audit

The baseline replay uses one token function for all post-edit queries. `tokens[3+i]` counts a relay-row query and `tokens[7+4*i+j]` counts a terminal-cell query. The old transcript is constructed once under the old source distribution and replayed into `Audit(new_source, ...)` for CTDR.

The published `stale.queries` and `stale.tokens` fields are historical pre-edit audit costs. Stale post-edit incremental cost is zero. The fields should not be compared directly with CTDR post-edit incremental cost without separating the phases.

Offline replay over the four formal panels and three teams per panel gives the following equivalence.

| panel | n | warm q | CTDR q | warm tokens | CTDR tokens | same decision | same query count |
|---|---:|---:|---:|---:|---:|---:|---:|
| QASC-A | 576 | 1.3299 | 1.3299 | 127.71 | 127.71 | 100% | 100% |
| QASC-B | 576 | 1.1233 | 1.1233 | 107.73 | 107.73 | 100% | 100% |
| SciQ-A | 576 | 0.6910 | 0.6910 | 85.20 | 85.20 | 100% | 100% |
| SciQ-B | 576 | 0.6372 | 0.6372 | 72.18 | 72.18 | 100% | 100% |
| pooled mean | 2304 | 0.9453 | 0.9453 | 98.21 | 98.21 | 100% | 100% |

Warm restart replays every legal old observation into the new-source audit and then calls the same `Audit.next()` scheduler. Full CTDR calls `safe_transport` before the same scheduler. In every unit, `safe_transport=True` coincides with `Audit.decision()` already being available. Therefore the current implementation produces no independent radius or decision-ranking reduction relative to warm/reweighted replay.

The fair post-edit cost comparison is stale 0, warm/reweighted/CTDR 98.21, full restart 493.16, and matched-budget no-history controls around 104--113 tokens depending on policy. The 80.09% pooled saving over full restart belongs to historical transcript reuse plus reweighting, while the current evidence does not identify a separate contribution from rho or decision-region ranking.

This audit is diagnostic. It does not alter the formal M60 result files.

## Real token-price ranking diagnostic

Using each recorded relay and terminal generated-token length as the scheduler price, while keeping the same complete old history, exact decisions remain identical on all 2304 units. Pooled warm/reweighted cost is 98.2057 tokens and price-aware cost is 96.6341 tokens, a 1.6003% reduction. Panel reductions are -0.0734% for QASC-A, 0.1902% for QASC-B, 4.7459% for SciQ-A, and 2.9535% for SciQ-B. This is a separate small cost-aware scheduling effect. It does not explain the approximately 80% reduction versus cold full restart.
