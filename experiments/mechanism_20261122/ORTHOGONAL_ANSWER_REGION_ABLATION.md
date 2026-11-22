# Strict unknown-row sensitivity

This sensitivity holds the cached transcript, graph rows, calibration split and
stopping rule fixed. It changes only the source of the uncertainty radius. It
uses the observed graph-change rows for calibration, then expands the test mask
to all rows and assigns the conservative default radius to rows with no
calibration observations. The main four-row protocol is reported separately in
`GENERIC_ANSWER_REGION_ABLATION.md`.
`graph-radius` estimates a separate conditional-drift radius for each relay row.
`pooled-radius` uses one pooled radius for every row and therefore removes the
row-conditional information. The uncertainty-only selector removes the token
cost denominator from row ranking.

## Historical held-out cells

| mask | controller | decision exact | answer-action exact | relay tokens | queries |
|---|---|---:|---:|---:|---:|
| changed rows | graph-radius + cost | 0.9972 | 0.9905 | 120.00 | 1.490 |
| changed rows | pooled-radius + cost | 0.9972 | 0.9905 | 122.79 | 1.522 |
| changed rows | graph-radius, no cost term | 0.9972 | 0.9905 | 120.00 | 1.490 |
| all rows | graph-radius + cost | 0.9981 | 1.0000 | 141.07 | 1.696 |
| all rows | pooled-radius + cost | 0.9981 | 1.0000 | 253.24 | 2.764 |
| all rows | graph-radius, no cost term | 0.9981 | 1.0000 | 141.08 | 1.696 |

Under this strict unknown-row interpretation, the all-row controller spends
270.26 tokens and the pooled control 263.70 tokens. This panel is a boundary
sensitivity rather than the primary result. It shows that a row-conditioned
radius cannot be extrapolated to an unseen interface without an invariance
condition. The main protocol therefore gates row-conditioned transport on the
observed interface fingerprint and falls back to conservative all-row
revalidation when that fingerprint is missing or changed.

The result does not show an accuracy gain. It shows that the graph-conditioned
uncertainty model, rather than answer-region stopping alone, accounts for a
large part of the cost reduction when all relay rows are eligible for reuse.

