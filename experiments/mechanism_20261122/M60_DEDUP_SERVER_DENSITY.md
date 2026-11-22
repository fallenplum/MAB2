# M60 deduplicated density audit

This report is a development-stage mechanism audit on three disjoint task panels (QASC, SciQ, Commonsense), 128 questions per panel, and two local checkpoints (Qwen-0.5B and Qwen-14B). It uses the same old transcript for stale, warm reweighted, random, periodic, TCR, and cold restart. The worker now evaluates each distinct prompt once and reuses identical old/new relay logits.

## Results

| edit density | warm exact set | TCR token exact set | TCR token cost | cold token cost | TCR saving vs cold |
|---|---:|---:|---:|---:|---:|
| one row | 96.875% | 100.000% | 132.95 | 1296.10 | 89.7% |
| two rows | 88.932% | 100.000% | 224.80 | 1255.30 | 82.1% |
| four rows | 77.083% | 100.000% | 419.10 | 1188.98 | 64.7% |

Values are weighted over 768 question–model units per density. No confidence intervals or bootstrap were run, per project contract.

## Interpretation boundary

The exact-match result supports a conditional claim that interface-fingerprint-guided relay revalidation can recover the finite audit decision set on these panels while using fewer incremental logical tokens than a cold restart. Rows2 and rows4 use a conservative density gate that falls back to cold restart, so they do not establish an adaptive multi-row saving beyond the safe fallback.

The old M60 radius and generic decision ranking are not credited as independent innovations. The worker fix and fingerprint-derived affected-row mask are required for the result to be interpretable. Fingerprint masks matched the worker's edit metadata on all 2,304 units. Gold labels are used only by the scorer for reporting and are not used to form the mask or choose actions.

This is not yet an ACL Main claim. The remaining evidence gap is an independent task family with graph manifests and conditional-channel drift controls, plus a replay that derives the mask from those manifests rather than post-hoc row metadata.

