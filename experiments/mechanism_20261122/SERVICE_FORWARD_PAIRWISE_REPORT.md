# Pairwise Gap Certificate service-forward measurements

The service benchmark was run on the server with Qwen-0.5B and Qwen-14B using
the matched single-forward worker inputs. The pooled worker selection fixed the
Pairwise Gap radius quantile at (q=0.9). The service process measured model
forwards, input tokens and wall time independently of the worker replay.

## Qwen-0.5B

| density | method | forwards/question | input tokens/question | wall s/question | action exact | decision exact | answer accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | TCR guard | 1.000 | 131.99 | 0.0311 | 1.000 | 1.000 | 0.234 |
| 1 | Pairwise Gap (q=.9) | 0.867 | 115.27 | 0.0262 | 1.000 | 0.992 | 0.234 |
| 2 | TCR guard | 1.000 | 131.99 | 0.0263 | 1.000 | 1.000 | 0.234 |
| 2 | Pairwise Gap (q=.9) | 0.867 | 115.27 | 0.0228 | 1.000 | 0.992 | 0.234 |
| 4 | TCR guard | 3.000 | 395.98 | 0.0778 | 1.000 | 1.000 | 0.305 |
| 4 | Pairwise Gap (q=.9) | 2.852 | 377.56 | 0.0686 | 1.000 | 0.992 | 0.305 |

The answer-action target is preserved while service forwards fall by 13.3% at
one- and two-row density and 4.9% at four-row density. The edge decision-set
diagnostic loses one residue unit at each density. The method is evaluated at
the answer-action target, so this trade-off is recorded rather than hidden.

## Qwen-14B

The Qwen-14B service process was also run at (q=.9) and (q=1.0). Pairwise
Gap and the answer-region route selected the same rows on the service trace,
so their forward counts and action values coincide. At (q=.9), Pairwise Gap
uses 0.039, 0.977 and 2.891 forwards per question at rows1, rows2 and rows4.
The corresponding input-token counts are 4.00, 100.59 and 298.45. The
matched worker replay still shows logical-token reductions against the fair
descendant mask. The service action values are retained as diagnostics because
the custom model forward path does not reproduce the worker reference exactly.

The raw JSON files are `service_q05_mb1_r*_pairwise_q09.json`,
`service_q05_mb1_r*_pairwise.json` and `service_q14_mb1_r*_pairwise.json`.

