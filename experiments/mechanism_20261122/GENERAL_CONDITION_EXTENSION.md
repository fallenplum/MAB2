# General conditional-channel extension audit

This audit tries both extension routes while keeping claim boundaries explicit.

## Route A: narrow source-edge certificate transport

The existing same-input interface panel provides two conditional-channel drift
variants, a short protocol suffix and a semantic reviewer suffix, across QASC,
SciQ and CommonsenseQA, three fixed checkpoints, and one-, two- and four-row
edits. Each density contains 1,152 question-model units. Full CTDR guard has
exact-set recovery 1.000 and action error 0 on all six variant-density cells.
Warm/max-uncertainty exact-set recovery ranges from 0.773 to 0.986 and its
action error reaches 0.107 on semantic four-row edits. The corresponding CTDR
incremental token means are 145.27, 262.38 and 502.10 for short edits, and
177.86, 325.69 and 628.22 for semantic edits. The source is the existing
same-input cohort, so these are mechanism stress controls rather than an
independent task confirmation.

The route-A result supports the narrow claim that graph and interface metadata
can determine when an old certificate must be revalidated. It does not support
a general multi-task service-quality claim.

## Route B: broader conditional-channel sanity check

`general_channel_synthetic.py` evaluates a binary service action under source-only,
conditional, interface and mixed drift. It uses the same old channel, post-edit
source, row prices, stopping rule and token ledger for stale, warm reweighted,
CTDR transport, fixed reverse-order recheck, full restart and an explicitly non-executable
oracle lower bound. The generated row radii equal the true binary-channel TV
drift and are therefore synthetic oracle diagnostics.

Across 500 cases per drift type, CTDR has zero action error in all four cells.
It uses zero new tokens for source-only drift, 9.51 tokens for conditional drift,
9.36 for interface drift, and 11.54 for mixed drift. Full restart uses 41.81,
41.76 and 42.22 tokens in the latter three cells. Fixed reverse-order recheck uses 14.20
tokens in the mixed cell. Stale and warm reuse have action error 0.430, 0.486
and 0.404 on conditional, interface and mixed drift.

This route demonstrates representation-level transfer of the certificate logic,
including the no-drift degeneration to warm reuse. It is synthetic and uses an
oracle radius. A real BoolQ, GSM8K or other non-MCQA worker run with a new
checkpoint was not completed in this cycle, so Route B is not used to widen the
paper's empirical claim.

## Decision

Keep the paper's main claim on graph-conditional certificate transport for the
finite MCQA interface. Preserve Route B as a falsifiable extension artifact and
do not present it as real-task generalization.

