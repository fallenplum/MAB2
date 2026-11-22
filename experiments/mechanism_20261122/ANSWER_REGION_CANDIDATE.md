# Answer-region credit revalidation candidate

## Hypothesis

After a graph edit, a relay row can be revalidated only when the final answer
region is not certified. A graph-edge certificate is stronger than needed for
answer serving and can therefore spend feedback on labels that cannot change
the final answer.

For a changed row `i`, let `rho_i` be a calibrated total-variation radius,
`s_i` the current source mass, and `C_i(a)` the relay probability mass assigned
to answer label `a`. The contribution interval is

`[s_i(C_i(a)-rho_i), s_i(C_i(a)+rho_i)]`.

After combining unchanged cached rows and uncertain changed rows, stop when
`L_a > max_{b != a} U_b` for the current answer `a`. This certifies the final
answer action while allowing individual edge decisions to remain unresolved.

## Fixed evaluation protocol

- Calibration uses questions with `hash(key) mod 5 < 3`.
- The selection split is used only for development diagnostics.
- The reported primary candidate uses a fixed calibrated TV radius (`q=0.9`)
  selected before the independent test and revalidates all four relay rows. It
  does not rely on an oracle changed-row mask. The maximum-radius `q=1.0`
  policy is retained as a safety control, while the graph-edit descendant mask
  is a secondary diagnostic.
- The all-row policy is compared with the descendant-mask policy to expose
  whether graph-localized metadata is actually safe for the model interface.
- Metrics are decision exactness, final-answer action exactness, answer
  accuracy, incremental relay tokens, and query count.
- No confidence intervals or bootstrap procedures are run under the project
  experiment convention.

## Evidence

The corrected independent ARCC cohort has ten cells spanning Qwen2.5-0.5B,
Qwen-14B, and Llama-3.2-1B, with one-, two-, four-row drift and interface drift.
The all-row q=0.9 policy reaches mean decision exactness 0.9968, answer-action
exactness 1.0, answer accuracy 0.3161, and 162.29 relay tokens. TCR guard reaches
1.0, 1.0, 0.2719, and 276.75 tokens. Cold restart costs 1396.93 tokens. The
all-row controller therefore saves 41.4% relative to TCR guard on this independent
cohort while preserving the final answer action. One Qwen-14B four-row cell has
decision exactness 0.9677, which remains an explicit boundary failure.

The independent outputs were regenerated with a deduplicating worker after a
batch-position artifact was found in the earlier worker. Earlier non-deduplicated
figures are excluded from the claim. The all-row fallback is the primary policy.
The descendant-mask policy remains a secondary diagnostic because interface drift
can change rows outside the declared graph edit mask.

The historical 51-cell all-row q=0.9 evaluation gives 0.9981 decision exactness,
1.0 answer-action exactness, 0.5521 answer accuracy, and 141.1 tokens. At q=1.0
it gives exact decision recovery at 150.9 tokens. The corresponding full
graph-plus-answer certificate averages 0.999, 0.990, 0.551, and 221.8 tokens.

Answer accuracy does not improve because the policy preserves the model's
selected action. The evidence supports a cost reduction for certified action
maintenance, not a claim of better language-model knowledge or task accuracy.

## Open failure conditions

The independent cohort remains a single new MCQA task family. The historical
QASC, SciQ, and Commonsense panels provide three additional task families and
the service-level readout, but the main claim remains limited to action
preservation and feedback cost. A publication-level claim still requires a
fixed finite-sample coverage rule for `rho_i`, explicit interface-invariance
conditions, and a complete end-to-end latency accounting. Until those are
closed, this remains a candidate mechanism rather than a final ACL Main result.

