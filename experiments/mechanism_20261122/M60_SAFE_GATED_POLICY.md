# M60 Safe-Gated Budget Policy

This candidate combines a row-impact classifier with a graph-drift and decision-boundary gate. The gate uses only old observations and source-distribution shift. Hyperparameters are selected on non-target development tasks. Evaluation uses question-disjoint target rows with sealed answer labels.

| target | warm answer | safe-gated answer | warm exact | safe exact | safe tokens |
|---|---:|---:|---:|---:|---:|
| QASC | 0.2698 | 0.2698 | 0.9841 | 0.9841 | 26.27 |
| SciQ | 0.6522 | 0.5942 | 0.9565 | 0.9710 | 18.57 |
| Commonsense | 0.4375 | 0.5625 | 0.9583 | 0.9583 | 31.01 |

The method matches the cold answer accuracy on Commonsense while querying about one third of a row on average. The result does not transfer to SciQ or QASC. The candidate is retained as a mechanism probe and is not promoted as a final ACL method.

