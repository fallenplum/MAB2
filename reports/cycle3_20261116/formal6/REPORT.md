# Formal6: evidence-indexed communication applicability

Fixed gates passed: False. This is an applicability pilot, not a demonstrated active-feedback method or ACL-ready result.

All96 fresh training questions and2 execution seeds completed;2112 terminal outputs,6912 logical calls and3490 physical calls. Official validation/test remained unopened. The96 questions are now exposed development data.

Actual GPU occupancy: 2.495882985h. Global closed use: 79.260637723/160h. Model transfers are included in occupancy; model-call latency is not end-to-end latency.

| Frozen gate | Passed |
|---|---|
| selector_valid | True |
| fixed_G1_quality | False |
| credit_shift_support | True |
| certificate_coverage | True |

| Condition | EM | F1 | Logical service tokens | Service utility |
|---|---:|---:|---:|---:|
| G0 | 0.458333 | 0.544091 | 3173.562 | 0.442466 |
| G0_MINUS_EU | 0.463542 | 0.546070 | 2995.276 | 0.448565 |
| G0_MINUS_EV | 0.442708 | 0.516147 | 2996.536 | 0.427726 |
| G1 | 0.453125 | 0.536260 | 2933.672 | 0.438457 |
| G1_MINUS_EU | 0.348958 | 0.433990 | 2452.318 | 0.336697 |
| G1_MINUS_EV | 0.390625 | 0.458628 | 2471.193 | 0.378269 |
| FLAT_EVIDENCE | 0.453125 | 0.538604 | 2754.635 | 0.439352 |
| SINGLE_Q | 0.505208 | 0.596796 | 1567.198 | 0.497372 |
| SINGLE_M | 0.505208 | 0.630385 | 1763.068 | 0.496393 |
| INDEPENDENT3_Q | 0.619792 | 0.727016 | 6613.479 | 0.586724 |
| VOTE_Q3 | 0.500000 | 0.619452 | 4491.370 | 0.477543 |

Credit shifts: 0.096354 of question-seed-edge observations, 22 distinct questions. Exact terminal evidence equality coverage: 0.218750; violations: 0. Selector validity: 0.989583.

D8 fixed model gates: {"ridge": false, "tree": false}.
D9 is descriptive reliability/headroom only. Opposite-seed oracles consume unavailable labels and are not deployable policies. D8 excludes training feedback costs; neither diagnostic overrides the formal gates.

Advantages: complete request/cache/cost reconstruction, fixed nominee, strong single/flat/independent controls and independent raw scoring. Risks: small pilot, only two seeds, sparse credits, protocol-specific evidence partition, service-only utility and no tested active mechanism. Do not select another matrix condition retrospectively as a confirmed winner.

Project remains ACTIVE. Retain this result unchanged, analyze the recorded failure/signal, then freeze the next justified candidate or fresh policy confirmation within the inherited160GPUh cap. No confidence intervals or bootstrap were run.
