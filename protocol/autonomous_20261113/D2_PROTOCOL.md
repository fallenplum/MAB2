# D2: bounded root-cause diagnosis

This is exploratory reuse of all historical G2 data. The120 formerly held-out
questions have already been exposed. The machine-readable D2_PROTOCOL.json is
the exact metric and intervention contract. No model calls or new test labels.

R0→R1→R1b distinguishes the single global intercept penalty from the entire
question-only baseline penalty. R2 removes arbitrary training-question offsets
by centering BOTH features and targets. R3 weights the24 actually observed
directions; shared graph labels remain correlated within questions. It changes
the contrast geometry and possibly loses between-anchor component directions,
so it is not a pure isolated objective correction. Record D rank/components.

All original features, rows and seed means remain fixed. Alpha selection uses
old tune data only. Old R0 coefficients, selected alpha and heldout predictions
must reproduce before trusting any intervention. Constant-target decomposition
is a linear algebra control, not an estimate of a unique causal contribution.

Benefit: tests plausible mechanisms cheaply without expanding network capacity.
Risk: all evaluation is development evidence; 5seed proxies are noisy, alternate
objectives change information geometry, and an improved offline action policy
does not validate sequential credit maintenance. Independent evidence is required.
