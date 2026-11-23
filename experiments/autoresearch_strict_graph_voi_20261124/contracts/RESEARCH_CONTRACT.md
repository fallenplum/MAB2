# P3 Strict Graph VOI research contract

## Scope

This cycle tests graph-conditional sequential revalidation. P0-P2 artifacts remain frozen and are not overwritten. The primary claim is a cost reduction at the same new-graph action risk under identical history, edit, information access and cost ledger.

## Locked certificate

For non-negative terminal weights `w_j` and labels `l_j`, known new-graph labels form `K` and unknown cells form `U`. `L_y` sums only trusted labels in `K` and `W_U=sum_{j in U} w_j`. Candidate `a` is certified only when `L_a > max_{b != a}(L_b + W_U)`. Equality is not certified. Unknown labels are exhaustively enumerated for small tests.

## Candidate requirements

A candidate must update downstream fingerprints, dirty cone, weights or certificate bounds after every observation. A fixed pre-sorted list or static threshold is not a strict Graph VOI candidate. Any predictor uses development data only. Gold labels and hidden new-node outputs are scoring-only.

## Locked baselines

Stale reuse, exact dependency cache, graph descendant invalidation, masked Pairwise, maximum uncertainty, random, periodic and generic DRD-equivalent search share the same legal action space and stopping target. Full-information plans are offline diagnostics.

## Gates

G1 requires a distinct sequential algorithm, valid certificate or calibrated risk contract, exhaustive finite verification and stable CPU savings against the strongest legal baseline. G2 requires two independent natural-language task settings and two checkpoint settings with genuine parent-child input consumption. G3 requires clean-checkout replay, independent tests T01-T10 and raw ledger provenance.

## Stop conditions

A candidate is rejected for soundness failure, cache permission failure, target leakage, cost accounting failure or no headroom. A project-level stop requires a declared budget, permission or environment block. Early P2 results using single-path terminal argmax remain quarantined.

