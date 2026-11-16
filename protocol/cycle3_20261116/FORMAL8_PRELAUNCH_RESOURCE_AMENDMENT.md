# Formal8 resource-only pre-first-call amendment

The initial GPU7 v4 plan was deployed but never launched or reserved: a fresh resource probe showed less than38000MiB free. Its plan and deployment receipts are retained. No Formal8 inference or quality scoring occurred.

Use the already-qualified symmetric CPU-offload runtimev3 on two memory-qualified shared cards (initial placement0,2), minimum28000MiB each. Each worker receives48 distinct inputs, keeps the same two-unit blind prefix and13950second residency cap; total7.75GPUh plus0.25h shutdown margin remains8GPUh. The prefix projection remains1.5*elapsed_at_second_unit*(assigned_units/2) and must pass on both workers. All loads, transfers, failures and waiting count.

The v2 symmetric worker was qualified on both artificial fixtures, exactly replayed, independently audited, and matched every completed condition from the initial variant. Runtimev3 adds only prelaunch zero-cost failure closure and supports one/two/four worker placement. Its scientific worker is unchanged. The later v4 artificial fixture comparison matched44 completed conditions/1248 fields. This is existing engineering evidence, not a new quality experiment or proof of general stochastic reproducibility.

The operator, models, source identities, exact96 selected inputs, condition order, seeds, prompts, outputs, alias scorer and all five scientific gates remain unchanged. No post-inference filtering or threshold change is authorized. The initial one-card placement contract is superseded only for residency/placement; prior files remain retained. This amendment is committed before first call and before Formal7 quality release.

Advantage: it uses currently memory-sufficient resources without touching other tasks. Risk: transfer/contention can slow the prefix enough to reject the run; preserve and charge any partial result rather than loosening the prefix or reducing the sample after observing outcomes.
