# MAB2 P0-P2 phase status

## P0

The M60 strong-baseline entrypoint was repaired and rerun on 3,456 units. The Oracle warm-cache return-value unpacking defect and the empty changed-row lower-bound case are fixed. All eleven controllers completed, with zero incomplete units. Pairwise and graph-mask costs are frozen in `MAB2_recovered/experiments/mechanism_20261122/P0_BASELINE_FREEZE.json`.

## P1

A controlled 15-node four-level DAG was evaluated on 5,000 cases and three additional 1,000-case seeds. Masked Pairwise and Topology VOI use the same affected descendant mask. Topology VOI retained exact action rate 1.0 and used 66.00 tokens on the main seed. Masked Pairwise used 66.56 tokens. The gain is stable but marginal. Offline full information is an upper-bound diagnostic.

## P2

A real Qwen-0.5B service panel used 32 fixed hash-sorted QASC rows, split 16/16. Terminal actions aggregate all sixteen cells with `P(S0=i) * P(R_i=j)`. On held-out rows, topology VOI achieved answer accuracy 0.375, action exact 1.0, 13.75 forward calls and 2956.1 input tokens per row. Graph-mask recompute also achieved action exact 1.0 with 18 calls and 3827.1 input tokens. Full new graph used 21 calls and 4222.4 input tokens. Single-model and RAG controls had answer accuracy 0.125 and action exact 0.5.

A fixed 128-row extension is exploratory. After the old-cache permission fix, topology VOI action exact is 1.0 versus old/exact-cache 0.96875 and graph-mask 1.0, using 13.17 calls versus 18 for graph-mask. It is retained as exploratory evidence because it reuses one QASC cohort and one checkpoint.

## Current claim boundary

The evidence supports a graph-conditioned service-cost reduction at unchanged action quality on the locked 32-row panel. It does not establish an answer-accuracy gain, cross-checkpoint robustness, or a universal topology advantage. Raw inputs, sealed keys and per-row outputs remain local/private.

The old-cache permission fix disables post-edit structured-interface suffixes during old-transcript replay. Early summaries produced before this fix are quarantined.
