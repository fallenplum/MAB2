# Interface-drift stress panel (rows2)

Source distributions and terminal prompts are fixed. New relay prompts append a frozen protocol-v2 suffix on the graph-descendant rows. The worker derives changed rows from old/new token fingerprints, and the scorer uses the frozen graph manifest. This is a same-input stress control over 1,152 question-model units from QASC, SciQ, and Commonsense with three model checkpoints.

| method | exact-set | post-edit tokens |
|---|---:|---:|
| warm_reweighted | 0.9097 | 23.30 |`n| tcr_guard | 1.0000 | 307.59 |`n| tcr_token | 1.0000 | 307.79 |`n| cold_token | 1.0000 | 960.12 |`n| random_recheck | 0.9097 | 38.52 |`n| periodic_recheck | 0.9097 | 40.33 |`n| tcr_bound | 0.8837 | 62.67 |`n
TCR revalidation recovers the full cold-restart decision set. Warm/reweighted and source-only replay retain stale relay certificates. The interval shortcut is retained as a negative control. The result supports a finite-interface mask rule and does not establish independent-cohort generalization or end-to-end answer quality.


