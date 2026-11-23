# Independent ARC-Challenge pooled-quantile worker replay

This replay uses the frozen pooled $q=.9$ radius selected before the ARC
cohort was scored. The 128 ARC-Challenge validation records are split into 73
calibration, 31 selection and 24 held-out questions for each edit density.
The held-out questions are evaluated once and are disjoint from the 27-panel
development residues.

| edited rows | Pairwise action exact | Pairwise tokens | fair descendant-mask tokens | reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 146.33 | 151.38 | 3.3% |
| 2 | 1.000 | 297.71 | 302.75 | 1.7% |
| 4 | 1.000 | 574.25 | 599.50 | 4.2% |

The ARC worker transfer run uses the same pooled $q=.9$ value as the service
benchmark. The radius is calibrated on the 73-question split and the 31-question
selection split is used only to freeze the pooled quantile before the 24-question
test residue. This file reports logical relay tokens and answer-action agreement
only. It does not claim improved gold-answer accuracy.

