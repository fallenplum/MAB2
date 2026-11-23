# P3 CPU strict graph VOI contract

The controlled generator creates real parent-to-child dependencies. A child
input fingerprint includes every queried parent output. A source query can
therefore either validate its cached descendants or dirty their entire cone.

The primary comparison uses the same graph, source edit, cost regime, query
budget and full-recompute action target. `topology_voi_exact_dp` uses a frozen
development prior and exact dynamic programming over the remaining binary
source states. It is an executable diagnostic of nonmyopic information value,
not an oracle that sees the realised new graph.

`strict_certificate_first` uses conservative weighted action bounds and stops
only when every unqueried node assignment preserves the current action.
`masked_pairwise` and `generic_adaptive` are static-order controls. The
offline full-information line is excluded from deployable comparisons.

The contract is frozen before result execution. Negative results and topology
families without headroom remain in the result ledger.

