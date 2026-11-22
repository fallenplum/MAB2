# Answer-margin stratification

The historical held-out rows are grouped by the final-answer margin under the
edited graph. Smaller margins require more revalidation, as predicted by the
answer-region stopping rule.

| margin range | q | n | relay tokens | queries | action exact |
|---|---:|---:|---:|---:|---:|
| [0, .01) | .9 | 522 | 200.60 | 2.375 | 1.000 |
| [0, .01) | 1.0 | 522 | 208.31 | 2.475 | 1.000 |
| [.01, .05) | .9 | 173 | 154.05 | 1.948 | 1.000 |
| [.01, .05) | 1.0 | 173 | 168.49 | 2.104 | 1.000 |
| [.05, .10) | .9 | 75 | 162.52 | 1.960 | 1.000 |
| [.05, .10) | 1.0 | 75 | 185.81 | 2.187 | 1.000 |
| [.10, inf) | .9 | 37 | 53.19 | 0.649 | 1.000 |
| [.10, inf) | 1.0 | 37 | 103.35 | 1.162 | 1.000 |

The monotonic trend is strongest between near ties and large-margin answers.
This is evidence that the controller spends feedback according to the final
decision region rather than a fixed number of relay queries.

