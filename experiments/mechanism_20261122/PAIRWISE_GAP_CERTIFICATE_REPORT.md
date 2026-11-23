# Pairwise answer-action certificate

## Candidate mechanism

The route targets the final answer action instead of the full edge-sign vector.
For the current cached graph, let `a` be the highest-scoring answer and let
`b` be a competitor.  For each unqueried changed row `i`, a conditional
total-variation radius `delta_i` bounds the change in the pairwise score by
`2 * s_i * delta_i`.  The route certifies the action when every lower bound

`V(a) - V(b) - sum_i 2 s_i delta_i`

is strictly positive.  Otherwise it queries the row with the largest
worst-case pairwise-gap reduction per observed token and repeats.  The claim is
conditional on the supplied radii.  Empirical radii are development estimates
and do not provide finite-sample coverage guarantees.

At runtime, tokenized relay-interface fingerprints take precedence over legacy
changed-row metadata. Missing or malformed metadata falls back to all four rows.
The metadata resolver has regression tests for fingerprint precedence, legacy
compatibility and fail-closed fallback.

## Independent worker replay

The 27 deduplicated Qwen-0.5B, Qwen-14B and Llama-3 panels use the existing
question-disjoint residue.  Quantile selection uses only the calibration and
selection residues.  No test residue is used to select a radius.

| model | panels | action exact | pairwise tokens | full changed-row tokens | token reduction |
|---|---:|---:|---:|---:|---:|
| Qwen-0.5B | 9 | 1.000 | 160.69 | 215.69 | 25.5% |
| Qwen-14B | 9 | 1.000 | 86.30 | 148.02 | 41.7% |
| Llama-3 | 9 | 1.000 | 89.80 | 171.32 | 47.6% |
| all panels | 27 | 1.000 | 112.27 | 178.34 | 37.1% |

The complete cell-level records are in the `dedup_*_pairwise.json` files. The
full changed-row comparator is the fair answer-action mask baseline. The route
has no claim about exact edge decisions when it does not acquire every changed
row.

As a fixed-radius robustness diagnostic, every tested quantile from 0.5 to
1.0 retained action exactness on all 27 held-out panels. Mean tokens increased
from 111.43 at q=0.5 to 134.77 at q=1.0. The diagnostic is stored in
`PAIRWISE_GAP_QUANTILE_STRESS.json` and was not used to alter the test protocol.

## Matched Qwen-14B worker panels

| edited rows | pairwise action | pairwise tokens | fair answer mask tokens | risk-gated tokens |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 3.39 | 112.65 | 19.65 |
| 2 | 1.000 | 110.23 | 112.65 | 112.65 |
| 4 | 1.000 | 320.00 | 337.94 | 337.94 |

The pairwise route reduces the rows1 cost further and gives a smaller but
positive reduction at rows4.  These are logical worker replay tokens.  They do
not establish deterministic service-forward savings for the unresolved Qwen-14B
entry point.

## Conditional-drift mechanism test

The synthetic panel supplies the true conditional TV radius and varies the
number of edited rows and drift.  It contains 5,000 random instances per cell.

| edited rows | drift | action exact | pairwise queries | full-mask queries |
|---:|---:|---:|---:|---:|
| 1 | 0.1 | 1.000 | 0.253 | 1 |
| 1 | 0.4 | 1.000 | 0.629 | 1 |
| 1 | 1.0 | 1.000 | 0.811 | 1 |
| 2 | 0.1 | 1.000 | 0.614 | 2 |
| 2 | 0.4 | 1.000 | 1.357 | 2 |
| 2 | 1.0 | 1.000 | 1.654 | 2 |
| 4 | 0.1 | 1.000 | 1.569 | 4 |
| 4 | 0.4 | 1.000 | 2.944 | 4 |
| 4 | 1.0 | 1.000 | 3.341 | 4 |

The synthetic result supports soundness under the stated radius assumption and
shows that action certification can require fewer rows than an all-row mask.
It is not a task-quality evaluation.

## Scope decision

This candidate is retained as a mechanism upgrade over a scalar risk gate. Its
independent evidence supports answer-action cost reduction across 27 replay
panels. The manuscript must state that empirical-radius coverage and
single-forward service equivalence remain open limitations until a matching
service run is available.

