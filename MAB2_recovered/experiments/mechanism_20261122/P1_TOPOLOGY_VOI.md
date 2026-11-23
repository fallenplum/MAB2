# P1 controlled DAG topology VOI audit

This CPU-only experiment uses a four-level DAG with explicit cross-node dependencies. One source node is edited in each case, and its reachable descendants are the graph-conditioned affected set. The action is the argmax of three weighted scores over node credits. Costs are queried node token prices.

| method | action error | exact action | mean tokens | mean queries |
| --- | ---: | ---: | ---: | ---: |
| exact_cache | 0.0462 | 0.9538 | 0.00 | 0.00 |
| descendant_recompute | 0.0000 | 1.0000 | 132.09 | 11.03 |
| pairwise | 0.0000 | 1.0000 | 66.56 | 5.86 |
| generic_adaptive | 0.0000 | 1.0000 | 124.44 | 11.06 |
| topology_voi | 0.0000 | 1.0000 | 66.00 | 5.80 |
| full_recompute | 0.0000 | 1.0000 | 179.63 | 15.00 |
| offline_full_info | 0.0000 | 1.0000 | 0.00 | 0.00 |

## Interpretation

The topology-aware policy receives the affected descendant mask from the DAG. Pairwise and generic adaptive controls do not use that mask and treat all nodes as potentially stale. The offline full-information line is a diagnostic upper bound only and is not an executable baseline.

This experiment is evidence about topology-dependent information value. It does not establish natural-language task quality, model-call latency, or a universal advantage outside the generated DAG family.

## Seed sweep

Three additional seeds with 1,000 cases each preserve the same ordering. Topology VOI has exact action rate 1.000 on all 3,000 cases, with mean cost between 66.20 and 68.05 tokens. Masked Pairwise costs range from 66.79 to 68.47 tokens and generic adaptive costs range from 123.94 to 125.67 tokens. The detailed table is in `P1_TOPOLOGY_VOI_SEEDS.md`.
