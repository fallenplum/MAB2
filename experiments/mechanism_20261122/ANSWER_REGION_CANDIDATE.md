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
- The reported candidate uses a fixed `q=0.9` calibration quantile and the
  graph-edit descendant mask. It does not select `q` on the held-out split.
- The all-row mask is a control for the value of graph-localized metadata.
- Metrics are decision exactness, final-answer action exactness, answer
  accuracy, incremental relay tokens, and query count.
- No confidence intervals or bootstrap procedures are run under the project
  experiment convention.

## Evidence

The independent ARCC cohort has ten cells spanning Qwen2.5-0.5B, Qwen-14B,
and Llama-3.2-1B, with one-, two-, four-row drift and interface drift. The
changed-mask q=0.9 mean is decision exactness 0.983871, answer-action exactness
0.958065, answer accuracy 0.367742, and 132.803226 relay tokens. The all-row
q=0.9 control reaches decision exactness 0.995037 and action exactness 1.0 at
252.977667 tokens. The historical 51-cell evaluation gives q=0.9 means of
0.997 decision exactness, 0.990 answer-action exactness, 0.550 answer
accuracy, and 120.0 tokens. The corresponding full graph-plus-answer
certificate averages 0.999, 0.990, 0.551, and 221.8 tokens.

Answer accuracy does not improve because the policy preserves the model's
selected action. The evidence supports a cost reduction for certified action
maintenance, not a claim of better language-model knowledge or task accuracy.

## Open failure conditions

The independent cohort is still one MCQA task family. A publication-level
claim requires additional task families, a public implementation of the
descendant mask, a fixed finite-sample coverage rule for `rho_i`, and an
end-to-end service evaluation. Until those are available, this remains a
candidate mechanism rather than a final ACL Main result.

