# Qwen3B sparse conditional-drift pilot

This is a single-model development pilot on three task panels, 384 questions per edit-density variant. It is not independent multi-model confirmation. All methods share the same old transcript and post-edit scorer.

| variant | warm | random | periodic | TCR | density gate | cold | TCR exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| rows1 | 67.6 | 194.2 | 172.9 | 156.6 | 156.6 | 259.6 | 100.00% |
| rows2 | 66.8 | 187.9 | 164.4 | 252.6 | 256.1 | 256.1 | 100.00% |
| rows4 | 64.7 | 177.8 | 154.7 | 428.2 | 240.5 | 240.5 | 100.00% |

Token values are mean post-edit token costs over 384 questions. TCR is exact on all three variants in this pilot. Singleton edits retain a cost advantage. At two changed rows, unconditional TCR is approximately equal to cold restart, so the density gate falls back to cold. At four changed rows, unconditional TCR is more expensive and the gate again falls back to cold. The worst-case interval shortcut is rejected because it introduced action errors.
