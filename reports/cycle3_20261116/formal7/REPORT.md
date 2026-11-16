# Formal7: evidence-preserving annotations

All fixed foundation gates passed: False. This result does not establish an active acquisition method, independent replication or ACL-level novelty.

The fixed96 fresh training questions, two generation seeds and11 conditions completed. All prior8 prefix questions were rerun and compared before label release; the resource-stopped attempt remains unscored on its own and is fully charged. CPU trace reconstruction and independently implemented raw scoring both passed. The96 questions are now exposed development data; official validation/test remain sealed.

| Frozen gate | Passed |
|---|---|
| complete_matrix | True |
| fixed_G1_quality | False |
| credit_shift_support | True |
| cross_seed_reliability | False |
| service_economic_signal | False |

| Condition | EM | F1 | Mean service tokens | Service utility | EM by seed |
|---|---:|---:|---:|---:|---|
| G0 | 0.479167 | 0.566551 | 4908.714 | 0.454623 | [0.4270833333333333, 0.53125] |
| G0_MINUS_EU | 0.489583 | 0.576000 | 4844.057 | 0.465363 | [0.4895833333333333, 0.4895833333333333] |
| G0_MINUS_EV | 0.494792 | 0.585640 | 4843.458 | 0.470574 | [0.4479166666666667, 0.5416666666666666] |
| G1 | 0.505208 | 0.600670 | 4331.047 | 0.483553 | [0.4791666666666667, 0.53125] |
| G1_MINUS_EU | 0.489583 | 0.573413 | 3737.995 | 0.470893 | [0.46875, 0.5104166666666666] |
| G1_MINUS_EV | 0.500000 | 0.587475 | 3751.016 | 0.481245 | [0.5, 0.5] |
| FLAT_ANNOTATION | 0.520833 | 0.601514 | 3792.026 | 0.501873 | [0.5104166666666666, 0.53125] |
| SINGLE_Q | 0.562500 | 0.629886 | 1540.828 | 0.554796 | [0.5833333333333334, 0.5416666666666666] |
| SINGLE_M | 0.489583 | 0.612117 | 1736.458 | 0.480901 | [0.4895833333333333, 0.4895833333333333] |
| INDEPENDENT3_Q | 0.604167 | 0.702658 | 6522.208 | 0.571556 | [0.625, 0.5833333333333334] |
| VOTE_Q3 | 0.604167 | 0.667163 | 4412.219 | 0.582106 | [0.5833333333333334, 0.625] |

G1 versus SINGLE_Q: EM difference -0.057292; 20 corrections and 31 destructions among192 question-seed observations. These are96 independent question groups, not192 independent questions.

Credit-shift fraction: 0.138021, across39 question groups. Current-credit cross-seed correlation: 0.25204394316941936; shift correlation: 0.19263274217821538.

Actual occupancy: formal attempts 4.687113970GPUh plus engineering 0.306720203GPUh; candidate total 4.993834173GPUh. Failed calls, loading, transfers and prefix waiting are included. Global closed use is88.458577745/160GPUh. Service utility uses the frozen token penalty and excludes feedback/init; no full-budget active-method benefit is inferred.

D11 is an exposed-data decision diagnostic. Current/alternate-seed and old-graph gold-dependent choices are unavailable oracles; prerequisite observation tokens/calls are reported separately. Oracle headroom and repeatability cannot override any failed Formal7 gate, and cannot justify claiming a practical policy wins.

Advantages: preserved legal raw evidence, exact request reconstruction, fixed strong single/flat/independent/vote controls, all-attempt costs and independently implemented scoring. Risks: one task, a small two-seed development pilot, protocol-specific input allocation, restricted edit actions and potentially dominant free structural policies. Full task/seed decompositions and all fixed baselines remain in SUMMARY.json. No confidence interval or bootstrap was computed.

Project remains ACTIVE. A failed gate ends this nominee; the next outer-loop decision must use a distinct justified mechanism. A passing foundation still requires a fixed budgeted policy, strong free baselines, fresh independent questions, multiple seeds, ablations and complete-cost confirmation before a positive paper claim.
