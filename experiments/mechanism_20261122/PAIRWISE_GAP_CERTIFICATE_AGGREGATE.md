# Pairwise answer-action certificate aggregate

The pairwise route and fair descendant-mask baseline use the same cached transcript, graph mask and row prices. The pairwise route selects its radius and quantile on residues 0--3 and is evaluated once on residue 4.

| group | cells | pairwise action | pairwise tokens | fair tokens | mean reduction | weighted reduction | fair action |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 27 | 1.000 | 112.27 | 178.34 | 40.2% | 37.1% | 1.000 |
| model=Llama-3 | 9 | 1.000 | 89.80 | 171.32 | 49.8% | 47.6% | 1.000 |
| model=Qwen-0.5B | 9 | 1.000 | 160.69 | 215.69 | 27.4% | 25.5% | 1.000 |
| model=Qwen-14B | 9 | 1.000 | 86.30 | 148.02 | 43.4% | 41.7% | 1.000 |
| density=1 | 9 | 1.000 | 45.54 | 77.29 | 41.6% | 41.1% | 1.000 |
| density=2 | 9 | 1.000 | 89.00 | 154.58 | 43.9% | 42.4% | 1.000 |
| density=4 | 9 | 1.000 | 202.26 | 303.16 | 35.0% | 33.3% | 1.000 |

