# M60 mechanism-control audit

## Scope

This audit separates graph-manifest masking, old-transcript reuse, conditional-channel drift, and token-priced scheduling. It reuses frozen replay outputs and does not tune on sealed labels.

## Existing controls

| control | manipulation | result | inference |
|---|---|---|---|
| Warm/Reweighted/No-radius | Same old transcript and new source, radius removed | Four panels and 2,304 question-model units had identical replay decisions, with post-edit cost about 98.21 tokens and exact-set 1 | The earlier savings are explained by transcript reuse and source reweighting. The radius is not independently credited |
| Wrong graph manifest | True rows1 descendant is relay 1, declared relay 0 | TCR exact-set 96.875%, cold restart 100% | Correct graph-to-relay mapping is required |
| Missing/degraded manifest | Missing or malformed graph manifest | Fallback to all relay rows. Guard remains exact, while density gate falls back to cold | Fail-closed behavior trades cost for safety |
| Worst-case row interval | Replace conditional-row revalidation by an interval shortcut | Action error 2.60%, 2.86%, and 2.86% on one-, two-, and four-row panels | Shortcut is rejected |
| Equal-budget controls | Same transcript, graph, and token cap | Random and periodic fall below TCR exact-set recovery across edit densities | Recovery is not explained by buying the same number of queries |

## Formal boundary

For a reweighted interval [L', U'] and nonnegative radius rho, L' - rho > tau implies L' > tau, and U' + rho < tau implies U' < tau. Therefore the radius test cannot terminate before the ordinary audit decision in the current loop. It is retained as a conservative guard and is not claimed as an independent source of savings.

A zero source-TV bound is insufficient when an affected relay conditional channel changes. If a changed row has source weight w and its old and new credits cross the action threshold, reusing the old value can produce a wrong action even when the source vector is unchanged. The sufficient finite-interface condition is that every changed relay descendant is in the revalidation mask and every unmasked relay and terminal channel is invariant. Missing signatures or malformed graph manifests map to the all-row mask.

## New interface-drift stress panels

The new rows1, rows2, and rows4 panels keep source distributions and terminal prompts fixed and append a frozen protocol-v2 suffix only to graph-descendant relay rows. Across three tasks, three checkpoints, and 1,152 question-model units per density, TCR guard and token ranking recovered the cold-restart decision set exactly. Token costs were 171.05 versus 952.71 for one changed row, 307.79 versus 960.12 for two changed rows, and 586.94 versus 994.26 for four changed rows. Warm/source-only exact-set rates were 0.9618, 0.9097, and 0.8177. The panels reuse the existing 128-question cohort and are same-input stress controls, not independent task confirmation.

## Claim boundary

The defensible method boundary is frozen graph-manifest descendant union plus relay-interface fingerprint detection with fail-closed revalidation. Generic decision-region stopping, TV bounds, uncertainty scheduling, and certificate reuse are controls or prior art. Current evidence supports finite-audit exact-set recovery and post-edit logical-token savings. It does not support end-to-end answer accuracy, wall-clock latency, or independent-cohort generalization.

## Acceptance gates

- Fresh sealed question cohort with a predeclared credit-to-graph controller.
- Source-only, conditional-interface, format, and manifest-degradation panels.
- Mask-only, signature-only, and full attribution with identical history and budget.
- Per-task and per-model costs plus wall-clock and GPU ledger.
- Independent replay from the graph manifest after corrupting row-level metadata.

