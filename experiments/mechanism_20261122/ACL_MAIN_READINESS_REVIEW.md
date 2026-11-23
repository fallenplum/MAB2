# ACL Main readiness review

## Current judgment

The evidence supports a mechanism paper candidate about graph-conditioned
communication-credit revalidation. It does not yet close the ACL Main bar.
The strongest result is a quality-preserving cost mechanism. Across 51 frozen
cells, row-specific drift radii with source-aware answer-region stopping use
141.07 relay tokens, compared with 253.24 for a pooled-radius control and
264.65 for pooled radii with source-agnostic priority. Across ten corrected
independent cells, all-row (q=0.9) preserves the final answer action and uses
162.29 tokens versus 276.75 for TCR guard. The Qwen-14B four-row cell has
decision exactness 0.9677 and remains an explicit boundary failure.

## Remaining rejection risks

1. The pooled-radius and source-priority ablations now isolate the method's two
   graph-conditioned components. Under the same trusted changed-row mask, the
   row-specific radius saves 2.3% and source-aware priority saves another 3.0%.
   The larger all-row gap relies on a zero-radius interface-invariance assumption
   for rows without observed calibration drift. The same ablation should be run
   on the independent cohort before claiming generalization.
2. The (q=0.9) radius is an empirical quantile. It is a useful operating
   point, but it does not provide a finite-sample coverage guarantee. Report
   the (q)-frontier and the (q=1.0) safety control explicitly.
3. Existing service results measure logical relay tokens and answer-action
   agreement. A main-track paper should add actual generated tokens, model-call
   counts, and elapsed latency on a held-out service stream.
   The corrected single-forward Qwen-0.5B stream now closes this accounting
   for three edit densities. It shows equal action safety and no relay-call
   reduction for answer-region q=1 relative to TCR. Qwen-14B cost accounting is
   now available for rows1/2/4 under single-forward execution. Its action
   comparison remains diagnostic because the custom model path does not exactly
   reproduce the worker reference.
   A risk-gated route reaches exact answer-action recovery on the 21 local
   historical cells. Edge matches there are consistency diagnostics because
   the low-risk branch does not acquire every conditional cell. On the three
   Qwen-14B historical panels it improves
   action exactness over TCR, but those panels still require a matching
   single-forward worker output before they can support a service claim.
   The matched Qwen-14B single-forward worker replay now covers rows1/2/4 and
   reaches exact action and edge recovery with 87.1%, 25.1% and 18.7% fewer
   logical relay tokens than TCR. The separate service entry point still has a
   model-forward mismatch, so this closes the algorithmic replay gate while
   leaving the end-to-end service gate open.
   Against the fair answer-only mask baseline, the matched risk gate saves
   82.6% at rows1 and ties the baseline at rows2/rows4. The larger TCR savings
   therefore compare different certification targets and must be labeled that
   way.
4. The theoretical soundness proof is a standard interval argument. The paper
   should present the graph-density and answer-margin predictions as the
   mechanism contribution, with the proof supporting the certificate rather
   than claiming a new decision-region theorem.
5. Descendant masks can fail under interface drift. The primary method should
   use the all-row fallback when interface invariance is unknown and report mask
   failures as a boundary result.

## Highest-value next experiments

- Recompute the pooled-radius and source-priority ablation on the ten-cell
  independent cohort with the same fixed calibration split.
- Run a held-out service stream for QASC, SciQ, and Commonsense with stale,
  TCR guard, answer-region (q=0.9), and cold restart. Record final answers,
  actual input/output tokens, model calls, and wall-clock latency.
- Re-run the risk-gated route with a matched Qwen-14B worker and service
  reference before treating it as a primary service method.
- Report the (q=0.5) through (q=1.0) frontier with question-level failure
  counts and density/margin strata.

Until these gates are complete, the appropriate claim is action-preserving
feedback-cost reduction under graph evolution. An unconditional exact
decision-certificate claim is unsupported.

