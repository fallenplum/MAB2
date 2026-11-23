# Service utility from matched controller traces

This report exposes the existing controller traces at the question level.  It
uses the held-out rows already selected by the frozen `key % 5 == 4` split and
keeps answer accuracy, answer-action agreement, certificate agreement, and
incremental relay tokens paired by question.  No new labels or bootstrap
estimates are introduced.

## Aggregate across 51 cells

| controller | answer accuracy | answer-action exact | decision exact | relay tokens |
| --- | ---: | ---: | ---: | ---: |
| stale | 0.5513 | 0.8498 | 0.8863 | 0.0 |
| max uncertainty | 0.5513 | 0.8498 | 0.9061 | 18.1 |
| tcr bound | 0.5725 | 0.9106 | 0.9006 | 52.2 |
| tcr guard | 0.5509 | 0.9896 | 0.9991 | 221.8 |
| answer-region q=.9, all rows | 0.5521 | 1.0000 | 0.9981 | 141.1 |
| answer-region q=1.0, all rows | 0.5521 | 1.0000 | 1.0000 | 150.9 |
| cold restart | 0.5521 | 0.9951 | 1.0000 | 1065.7 |

The answer-region controller matches cold-restart answer accuracy in aggregate and reaches worker-reference action exactness 1.0000 while reducing relay tokens by 86.8% at q=.9 and 85.8% at q=1.0.
The result is a quality-preserving cost result.  It does not show that the
controller improves the underlying model accuracy.

## Task breakdown

| task | controller | answer accuracy | answer-action exact | decision exact | relay tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| QASC | stale | 0.3950 | 0.8711 | 0.8347 | 0.0 |
| QASC | answer-region q=.9 | 0.4454 | 1.0000 | 0.9944 | 155.2 |
| QASC | cold restart | 0.4454 | 1.0000 | 1.0000 | 1175.3 |
| SciQ | stale | 0.6266 | 0.8696 | 0.8977 | 0.0 |
| SciQ | answer-region q=.9 | 0.5934 | 1.0000 | 1.0000 | 144.7 |
| SciQ | cold restart | 0.5934 | 1.0000 | 1.0000 | 1036.4 |
| Commonsense | stale | 0.6324 | 0.8088 | 0.9265 | 0.0 |
| Commonsense | answer-region q=.9 | 0.6176 | 1.0000 | 1.0000 | 123.4 |
| Commonsense | cold restart | 0.6176 | 0.9853 | 1.0000 | 985.3 |

Task accuracy varies with the selected model and dataset.  The mechanism's
stable effect is answer-action preservation with a large cost reduction.

## Scope

The table is an end-to-end service readout of model-generated answer
probabilities.  It still evaluates the current MCQA interface and does not
establish user-facing latency or deployment throughput.  The independent
ARC-Challenge worker replay is reported separately with the frozen pooled
$q=.9$ protocol and the input provenance manifest.

