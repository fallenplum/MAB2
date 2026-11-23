# Qwen-14B matched single-forward worker risk-gated replay

The Qwen-14B workers were regenerated with `MAB2_MICROBATCH=1`. The risk gate
was selected on the worker development residue and evaluated on the independent
residue.

| density | q | risk threshold | risk action | edge consistency | risk relay tokens | risk queries | fair-mask action | fair-mask tokens | TCR relay tokens | cold relay tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rows1 | 0.50 | 0.00 | 1.000 | 1.000 | 19.65 | 0.226 | 1.000 | 112.65 | 151.80 | 795.76 |
| rows2 | 0.50 | inf | 1.000 | 1.000 | 112.65 | 1.000 | 1.000 | 112.65 | 150.48 | 896.54 |
| rows4 | 0.50 | inf | 1.000 | 1.000 | 337.94 | 3.000 | 1.000 | 337.94 | 415.53 | 859.95 |

The matched worker replay shows an 82.6% relay-token reduction at rows1
relative to the fair answer-action mask. Rows2 and rows4 select the same
mask route and therefore have equal fair-baseline cost. The larger 87.1%,
25.1% and 18.7% reductions are relative to the full edge-audit TCR. The route
has exact answer-action recovery on the independent residue. The edge-consistency column compares the
selected relay values with the full worker reference. It is a diagnostic
because the low-risk mask branch does not acquire every downstream conditional
cell required for a runtime edge certificate.

The separate Qwen-14B service forward process records lower agreement between
single calls and the worker reference. Its call and input-token measurements
are retained as diagnostics. The matched worker table is the current evidence
for the route algorithm, while service action claims remain open until the
custom model forward path is deterministic across worker and service entry
points.

