# Development-gated route selection

The candidate route selector chooses between the graph-descendant mask and an
all-row answer-region controller using only the cached old transcript, the
edited source distribution, and development radii.  A second variant chooses
the route with fewer relay rows.

## Result

On the deduplicated historical QASC, SciQ and Commonsense panels, the nested
development selector preserved exact answer actions and edge decisions on the
held-out residue for the Qwen-0.5B and Llama-1B worker families.  It selected
the descendant mask at rows1/2 and the answer-region route for several rows4
cells.  One QASC rows4 cell reached decision exactness 0.952 while retaining
answer-action exactness 1.0.

The same selector did not transfer to the Qwen-14B server panels.  The
answer-region route was selected on development data, but the held-out
answer-action exactness was 0.875 for Commonsense, 0.857 for QASC, and 0.783
for SciQ under the low-call gate.  The corresponding TCR traces are the
stronger safety reference for these cells.

The corrected single-forward Qwen-0.5B service benchmark gives TCR and
answer-region q=1 action exactness of 1.0 at rows1, rows2 and rows4.  The
answer-region route uses 1.31, 1.31 and 3.02 relay forwards per question,
compared with 1.00, 1.00 and 3.00 for TCR.  It therefore does not reduce
actual relay forward cost on this model.

## Decision

Development route selection remains a diagnostic and is excluded from the
primary method claim.  The transfer failure is retained as evidence that
conditional credit revalidation is model- and density-dependent.  The robust
mechanism claim remains interface-aware invalidation of historical relay
credit with a conservative revalidation fallback.

