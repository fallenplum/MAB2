# Interface-drift stress panel (rows4)

Source distributions and terminal prompts are fixed. New relay prompts append a frozen protocol-v2 suffix on the graph-descendant rows. The worker derives changed rows from old/new token fingerprints, and the scorer uses the frozen graph manifest. This is a same-input stress control over 1,152 question-model units from QASC, SciQ, and Commonsense with three model checkpoints.

| method | exact-set | post-edit tokens |
|---|---:|---:|
| warm_reweighted | 0.8177 | 23.31 |`n| tcr_guard | 1.0000 | 587.82 |`n| tcr_token | 1.0000 | 586.94 |`n| cold_token | 1.0000 | 994.26 |`n| random_recheck | 0.8177 | 38.15 |`n| periodic_recheck | 0.8177 | 40.94 |`n| tcr_bound | 0.8229 | 144.72 |`n
TCR revalidation recovers the full cold-restart decision set. Warm/reweighted and source-only replay retain stale relay certificates. The interval shortcut is retained as a negative control. The result supports a finite-interface mask rule and does not establish independent-cohort generalization or end-to-end answer quality.


