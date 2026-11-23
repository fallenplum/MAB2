# Independent ARC-Challenge validation cohort

The Qwen-0.5B worker was run on 128 ARC-Challenge validation questions from
the server's cached dataset. This cohort was not used in the pooled quantile
selection or the 27-panel replay. The held-out residue contains 24 questions
per edit density.

| density | Pairwise action | Pairwise tokens | Fair mask action | Fair mask tokens | reduction |
|---:|---:|---:|---:|---:|---:|
| rows1 | 1.000 | 113.29 | 1.000 | 151.38 | 25.2% |
| rows2 | 1.000 | 253.75 | 1.000 | 302.75 | 16.2% |
| rows4 | 1.000 | 530.54 | 1.000 | 599.50 | 11.5% |

The route uses per-panel development residues to select its quantile and reads
the ARC answer map only after replay scoring. This is an independent question
cohort for the worker-level answer-action claim. It does not add a new model
family or a separate service-forward run.

