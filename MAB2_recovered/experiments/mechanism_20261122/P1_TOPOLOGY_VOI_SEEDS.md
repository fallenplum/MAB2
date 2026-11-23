# P1 controlled DAG topology VOI seed sweep

Three independent seeds were run with 1,000 DAG cases per seed. The policy and graph generator are unchanged across seeds. Existing Pairwise receives the same affected descendant mask as topology VOI.

| method | exact action mean | exact action range | mean tokens mean | mean tokens range |
| --- | ---: | ---: | ---: | ---: |
| exact_cache | 0.9567 | 0.9530--0.9590 | 0.00 | 0.00--0.00 |
| descendant_recompute | 1.0000 | 1.0000--1.0000 | 132.47 | 131.09--133.40 |
| pairwise | 1.0000 | 1.0000--1.0000 | 67.54 | 66.79--68.47 |
| generic_adaptive | 1.0000 | 1.0000--1.0000 | 124.62 | 123.94--125.67 |
| topology_voi | 1.0000 | 1.0000--1.0000 | 67.09 | 66.20--68.05 |
| full_recompute | 1.0000 | 1.0000--1.0000 | 179.61 | 179.41--179.94 |
| offline_full_info | 1.0000 | 1.0000--1.0000 | 0.00 | 0.00--0.00 |

The topology VOI policy is exact on all 3,000 cases and remains lower cost than the masked Pairwise and generic adaptive controls in every seed. The offline full-information line remains a diagnostic upper bound and is not an online method.
