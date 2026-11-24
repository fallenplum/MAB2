# P4-0 source and historical audit

The audit read the P3 CPU, service runner, contracts, tests, theory, ledgers,
reports, paper draft, historical literature matrix, and the research-cycle
negative-result references named by the P4 prompt.

The P3 numbers remain historical artifacts. Cycle 006 is recorded as
`EXPLORATORY_GAIN_WITH_CONFOUNDS` because the CPU simulator uses an ideal
deterministic source-to-label generator, the baseline action space is not
identical, and the H2 implementation differs from the real-service
`certificate_voi` implementation.

The following source issues remain open and prevent formal P4 candidate work.

- `strict_graph_voi_cpu.py` and H2 code can access `new_labels` and the
  simulator generator. They are ideal-model experiments, not deployable
  broker execution.
- Several simulator paths return `action=target` when no legal candidate or
  certificate exists. P4 forbids target backfill and requires unresolved or
  budget-exhausted status in the denominator.
- `p2_service_runner.py` marks dirty terminal cells using only the first parent
  index, although terminal prompts consume two relay outputs.
- The CPU H2 planner and service `certificate_voi` are different algorithm
  identities. Their results cannot be combined.
- The service ledger is created inside the candidate runner. It is not an
  independent broker ledger and cannot establish real-call authenticity.
- P3 QASC/SciQ panels are finite interface fixtures and do not satisfy the P4
  complex-NLP task qualification or 200 independent groups per task-model
  combination.

The public gate kit tests and example verifier pass. They are development-only
because the private fixture and isolated broker are unavailable. No P4 GPU,
paid API, formal candidate confirmation, or G1/G2 claim has been started.

