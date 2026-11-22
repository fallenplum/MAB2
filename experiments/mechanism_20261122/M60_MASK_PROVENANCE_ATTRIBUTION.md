# Mask provenance and attribution replay

The scorer now supports four changed-row provenance modes. The graph-manifest mode is the default Full CTDR route. Fingerprint mode derives the mask from old/new relay token-sequence hashes. Row-metadata mode uses the worker-declared `changed_rows`. The all-row mode is a conservative negative control.

## Results

| edit density | graph manifest guard tokens | fingerprint guard tokens | row metadata guard tokens | all-row guard tokens | exact set for all modes |
|---:|---:|---:|---:|---:|---:|
| 1 | 170.2092 | 170.2092 | 170.2092 | 487.0234 | 1.0 |
| 2 | 307.5851 | 307.5851 | 307.5851 | 517.0365 | 1.0 |
| 4 | 587.8151 | 587.8151 | 587.8151 | 587.8151 | 1.0 |

The token-aware guard gives the same decision recovery and the same provenance equivalence. The one-row and two-row graph masks reduce mean guard tokens by 65.1% and 40.5% versus the all-row control. For four changed rows the masks coincide by construction.

## Interpretation

The replay rules out an implementation-level explanation in which Full CTDR's result depends on an unverified row-mask source. It also quantifies the finite-interface benefit of refreshing only graph descendants. The panel reuses the existing stress cohort, so it remains a mechanism attribution control rather than an independent task-quality result.

