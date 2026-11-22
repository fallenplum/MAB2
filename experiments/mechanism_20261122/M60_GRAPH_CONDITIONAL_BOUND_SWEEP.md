# Source-only transport bound stress sweep

This synthetic finite-row sweep tests whether a source-distribution-only transport certificate remains safe when conditional relay rows drift. The source-only bound is intentionally evaluated as a negative control.

| changed rows | conditional drift | warm exact | source-only policy exact | source-only certified |
|---:|---:|---:|---:|---:|
| 1 | 0.0 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 1.0 | 0.7139 | 0.7139 | 1.0000 |
| 2 | 0.0 | 1.0000 | 1.0000 | 0.8099 |
| 2 | 1.0 | 0.5556 | 0.6736 | 0.8121 |
| 4 | 0.0 | 1.0000 | 1.0000 | 0.7613 |
| 4 | 1.0 | 0.3810 | 0.5601 | 0.7588 |

The source-only certificate continues to certify rows while conditional drift causes decision flips. The result supports a narrow mechanism claim that source-TV transport is insufficient under conditional relay drift. It is a synthetic mechanism stress test and does not establish task-level answer quality.
