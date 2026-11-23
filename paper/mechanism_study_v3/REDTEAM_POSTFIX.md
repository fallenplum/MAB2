# ACL Main gate: post-fix review

## Scope

This review checks the current Pairwise Gap Certificate source, the pooled ARC
transfer replay, the service-utility audit, the ARC provenance protocol, and the
latest ACL-style PDF. The review treats answer-action preservation as the
primary target and keeps edge recovery, gold-answer accuracy, and relay-only
cost as separate diagnostics.

## Findings

The three former P0 protocol issues are fixed in the current source tree.

1. ARC worker and ARC service now use the same frozen pooled $q=.9$. The
   independent 24-question held-out replay has action exactness 1.000 at
   146.33, 297.71 and 574.25 logical relay tokens, versus 151.38, 302.75 and
   599.50 for the fair descendant-mask baseline.
2. ARC provenance records Hugging Face revision
   `210d026faf9955653af8916fad021475a3f00453`, dataset fingerprint
   `5336412f86bd5bc2`, Arrow size 98,080 bytes, Arrow SHA-256
   `4cf59b321066b627168d8a7b4c0ddeb5c29b5db6a8083056871baa943d5b0c90`, and
   128 unique question keys.
3. The abstract distinguishes the original Qwen-0.5B service cohort and uses
   the corrected ARC pooled-quantile reductions of 3.3%, 1.7% and 4.2%.

The evidence chain is sufficient for a bounded mechanism paper. The 27-panel
pooled replay, matched row-order controls, split-conformal diagnostic,
hidden-interface replay, graph-conditioned radius control, negative Full CTDR
ablation, 51-cell service utility table and independent ARC transfer all point
to the same claim. The claim is limited to graph-conditional certificate
maintenance and answer-action preservation on the tested finite MCQA interface.
The manuscript does not claim improved model knowledge accuracy, unrestricted
finite-sample safety, or complete end-to-end deployment latency.

## Remaining release condition

The PDF must be rebuilt once from the corrected `main.tex` and pushed together
with that exact source. The previously compiled PDF was produced before the
final abstract wording and the corrected ARC reductions. After this rebuild,
run the existing compile, page-count and visual checks again. No scientific
experiment is required for this release condition.

## Gate decision

**CONDITIONAL GO.** After the single source/PDF rebuild and branch update, the
current evidence supports an ACL Main-level mechanism submission within the
stated scope. A broader claim about general multi-model task accuracy or
full-service latency remains unsupported and must stay out of the paper.

## Release-condition verification

The corrected source was rebuilt with pdfTeX on the server and the resulting
five-page PDF was rendered locally. The source and PDF now correspond to the
same release pair. The release commit is `326181fee3b84fe94941c4f9560b2155a53a4b1b`.

