# Executable-feedback outer-loop decision

The long-document direct-deletion direction failed even an optimistic sunk-cost oracle. Freeze it. Executable feedback is a possible new observable channel, not a novelty claim or evidence of effectiveness.

| Candidate question | Nearest overlap | Decision / falsifier |
|---|---|---|
| Generate a changing code-agent graph from runtime errors | AgentConductor, ICML2026 | Reject as standalone novelty |
| Learn local edge credit with adaptive counterfactual mutations | LEMONv1; C3 trace-conditioned continuations | Reject as standalone novelty |
| Optimize agent prompts from visible tests and joint traces | CANTANTEv1 | Reject as standalone novelty |
| Inference-time revalidation using partial execution signatures after upstream graph edits, with frozen models/predictor | Combines established components; missing evidence is whether a cheap signature forecasts stale credit and improves which feedback is purchased under a complete fixed budget | Candidate only. Compare random,periodic,uncertainty,graph-change,simple visible-test failure priority, and cost-normalized selection. Must beat unchanged topology, single coder, best-of-k+test selection and self-repair. No claim until independent confirmation |
| Causal exact-context cache validity | LEMON AppendixC and C3 | Engineering correctness, not innovation |

Three perspectives agree on constraints: causal analysis requires the same legal information/actions for every strategy; software evaluation requires genuinely held-out executions and full failed-test costs; statistical analysis requires task-group partitions and no treating seeds as independent tasks. No confidence intervals or CI-bootstrap under user instruction.

Critical counterexample: on an unchanged coding task, previously generated programs remain valid candidate outputs even after an internal graph edit. A fair reuse baseline may simply return the best previously observed program. Do not prohibit such reuse to create headroom. An online cross-task experiment can instead study historical population credit on fresh tasks, but must explicitly distinguish this from same-question credit and preregister adaptation/feedback accounting.

Advantages of execution signatures: concrete observable failure evidence, no new predictor architecture, separation between feedback and final correctness. Risks: sparse visible tests, cached-program selection dominating revalidation, generation cost dominating cheap test cost, and prior work already covering broad graph-feedback ideas. Next action is input/isolation feasibility only; no GPU admission or effective method follows from that gate.
