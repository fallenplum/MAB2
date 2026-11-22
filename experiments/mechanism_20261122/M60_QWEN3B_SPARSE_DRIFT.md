# Qwen3B sparse conditional-drift pilot

This is a single-model development pilot on three task panels, 384 questions per edit-density variant. It is not independent multi-model confirmation.

| variant | warm tokens | TCR token | density-gate token | cold token | TCR exact | gate exact |
|---|---:|---:|---:|---:|---:|---:|
| rows1 | 67.60 | 156.61 | 156.61 | 259.57 | 100.00% | 100.00% |
| rows2 | 66.85 | 252.63 | 256.12 | 256.12 | 100.00% | 100.00% |
| rows4 | 64.66 | 428.15 | 240.55 | 240.55 | 100.00% | 100.00% |

Singleton edits retain a token advantage for TCR. At two changed rows, unconditional revalidation is approximately equal to cold restart and the density-gated fallback correctly chooses cold restart. At four changed rows, unconditional revalidation is substantially more expensive. The worst-case interval shortcut is rejected because it introduced action errors.
