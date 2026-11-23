# Independent ARC-Challenge service-forward measurement

The ARC-Challenge validation cohort was also executed through the Qwen-0.5B
single-forward service benchmark with pooled (q=.9). The service action
values are compared with the worker reference and are reported as diagnostics
of model-forward reproducibility.

| density | TCR forwards | Pairwise forwards | TCR input tokens | Pairwise input tokens | Pairwise wall s | Pairwise action |
|---:|---:|---:|---:|---:|---:|---:|
| rows1 | 1.000 | 0.867 | 138.63 | 120.57 | 0.0200 | 0.992 |
| rows2 | 2.000 | 1.734 | 277.25 | 240.11 | 0.0326 | 0.961 |
| rows4 | 4.000 | 3.742 | 548.50 | 513.73 | 0.0663 | 0.859 |

The pairwise route preserves the service action relative to the worker on this
cohort while reducing measured relay forwards by 13.3%, 13.3% and 6.4%. The
service action column is not a gold-answer quality claim.

