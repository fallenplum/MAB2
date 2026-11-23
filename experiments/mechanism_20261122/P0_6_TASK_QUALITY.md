# P0-6 per-question task-quality audit

Scored question rows: 6528 across 51 score files.

Corrected means baseline answer incorrect and candidate answer correct. Damaged means baseline answer correct and candidate answer incorrect.

| baseline -> TCR guard | corrected | damaged | unchanged correct | unchanged wrong | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm_reweighted -> tcr_guard | 348 | 219 | 3071 | 2890 | 6528 |
| max_uncertainty -> tcr_guard | 348 | 219 | 3071 | 2890 | 6528 |
| source_only_ctdr -> tcr_guard | 348 | 219 | 3071 | 2890 | 6528 |

The answer-accuracy transitions are computed from per-question sealed score artifacts. They are not inferred from decision-set transitions.
	cr_guard is the graph-manifest guard available in the per-question scorer. A separate per-question full-CTDR service trace is not present, so these transitions do not imply full-CTDR task quality.\n
