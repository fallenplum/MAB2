# Graph-conditioned answer-region ablation

This zero-additional-model-cost ablation reuses the 51 frozen worker cells and
keeps the same all-row answer-region stopping rule. The calibration and test
row sets both expose four possible relay rows. The controls remove the row-
conditional drift radii or the source-aware cost ranking.

| policy | decision exact | answer-action exact | answer accuracy | relay tokens | queries |
|---|---:|---:|---:|---:|---:|
| graph-conditioned radii + source-aware priority | 0.9981 | 1.0000 | 0.5521 | 141.07 | 1.696 |
| pooled radius + source-aware priority | 0.9981 | 1.0000 | 0.5521 | 253.24 | 2.764 |
| pooled radius + source-agnostic priority | 1.0000 | 1.0000 | 0.5521 | 264.65 | 2.896 |
| graph-conditioned radii, trusted changed-row mask | 0.9972 | 0.9905 | 0.5499 | 120.00 | 1.490 |
| pooled radius, trusted changed-row mask | 0.9972 | 0.9905 | 0.5499 | 122.79 | 1.522 |
| pooled radius + source-agnostic priority, trusted mask | 0.9972 | 0.9905 | 0.5499 | 126.64 | 1.567 |

With the same trusted changed-row mask, the row-conditional radius removes 2.3%
of relay tokens relative to the pooled-radius control. Source-aware token
priority removes a further 3.0% relative to the pooled source-agnostic control.
The larger all-row gap is driven by rows with no observed calibration drift
receiving zero radius. That operating point assumes interface invariance for
those rows. It must be reported as a mask-prior result, while nonzero pooled
radii provide the conservative interface-drift control.

The result isolates two graph-conditioned effects inside the answer-region
mechanism. It does not claim improved answer accuracy.

