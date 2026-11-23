# P0 baseline freeze

冻结日期：2026-11-24。后续实验不得把下面的成本口径与旧实验混用。

## Frozen artifacts

| block | artifact | SHA-256 | scope |
| --- | --- | --- | --- |
| M60 strong matrix | `P0_M60_STRONG_BASELINE_MATRIX_FROZEN_PUBLIC.json` | `705e4409c7c9650e354bf6d6d0c4a690624a4c8399cf68e18c77d460fd2635b1` | 3456 units, common post-edit budget 1924.961516 tokens |
| Pairwise | `PAIRWISE_GAP_CERTIFICATE_AGGREGATE.json` | `5bab00b6a57da8c3e1147379a62c0fb6def8b3a4d603572af8476565b0dfc929` | 27 matched cells, action exact 1.0000, pairwise tokens 112.2671, fair tokens 178.3439 |
| Graph mask | `M60_MASK_PROVENANCE_ATTRIBUTION.json` | `e7367921554243962246abd89e5ea0a4fb716111d5cc69468dd722e282ab6543` | 1152 question-model units per density, graph-manifest mode exact set 1.0 at rows1/2/4 |

## M60 corrected aggregate

| method | exact set | action error | tokens |
| --- | ---: | ---: | ---: |
| stale | 0.9039 | 0.0428 | 0.00 |
| exact_warm_cache | 0.8877 | 0.0443 | 0.00 |
| reweighted_old_history | 0.9167 | 0.0391 | 22.86 |
| ctdr_transport_only | 0.9167 | 0.0391 | 22.86 |
| ctdr_transport_selective_ranking | 0.9167 | 0.0391 | 24.61 |
| full_ctdr | 1.0000 | 0.0000 | 230.91 |
| max_uncertainty | 0.9167 | 0.0391 | 22.86 |
| periodic | 0.9167 | 0.0391 | 39.81 |
| random | 0.9167 | 0.0391 | 37.34 |
| full_restart | 1.0000 | 0.0000 | 878.78 |
| oracle_row_refresh | 0.9068 | 0.0391 | 42.71 |

## Pairwise aggregate

All 27 matched cells have action exact 1.0000. Mean pairwise tokens are 112.2671, fair baseline tokens are 178.3439, and unweighted token reduction is 0.4019. The weighted reduction is 0.3705.

## Graph-mask aggregate

The graph-manifest, fingerprint and row-metadata masks are decision/cost equivalent on the finite stress panel. Full CTDR exact set is 1.0 at rows1, rows2 and rows4. The graph-manifest token totals are 170.2092, 307.5851 and 587.8151 for one, two and four edited rows. The all-row negative control costs 487.0234, 517.0365 and 587.8151 tokens.

## Scoring entrypoint hashes

- `m60_strong_baseline_matrix.py` SHA-256 `91bc3e6a54a7ceb81fa1a51d2f8370e7245655e0ab4d46eed5db1d721f2eec0e`
- `conditional_real_score.py` SHA-256 `4d606a9a6be7865044348d7acd3483df8594a06695560b2ecca9a9888c93b9f1`
- `ctdr.py` SHA-256 `625749390376d5610af494974481caa9dd4ace827db590e445779dff11147704`
- `blind_scheduler.py` SHA-256 `13e457a1dfd746ed16e48b1864f7078211c9cecea3ddad548db21cdc1dcefe84`
- `test_p0_permissions_oracle.py` SHA-256 `4f771b65073c8012fe675fe32fbabbaede50f1260a611976ec7366ddeb0cf641`
- `pairwise_gap_baselines.py` SHA-256 `c8cabbdf118ef11698b4dd7e81dac44223f3957dffae66b5a9bd7fa8fd732911`
- `pairwise_gap_certificate.py` SHA-256 `279e5fd7fc7cd66baa3284f0ca81b7eced55e2667b50e2b4bd11c6296ea726a9`
- `pairwise_gap_global_selection.py` SHA-256 `de389c0590fbc85801127b7a37742e0668ee123bda944effbcb9a5c4b8aeb639`
- `score_interface_semantic.py` SHA-256 `9344e8c91564acc3845c408cb640b60db5c8c70e94a0e16251423dd2c25e29c8`
- `p0_5_6_audit.py` SHA-256 `0d1c33e949148c3891cd2829c18c36aafc67cc81099d5d56ba3a45715aeadd7c`

## Freeze boundaries

- The corrected M60 strong matrix is the formal P0 baseline.
- Pairwise and graph-mask summaries are frozen references, not new online inputs.
- The Oracle is a restricted hindsight diagnostic and is excluded from deployable Pareto claims.
- Logical token cost does not equal full service latency or model loading cost.
