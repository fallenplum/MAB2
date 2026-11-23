# P0 M60 runtime and baseline-freeze audit

Date: 2026-11-24

## Runtime defect fixed

`m60_strong_baseline_matrix.py` called `exact_warm_cache` as if it returned
three values. The controller returns `(prediction, actions)`, so the Oracle
entrypoint failed before selecting a row-refresh plan. The fix derives the
completion state from `prediction is not None`. The no-edit case also now has a
zero lower bound and zero fallback cost instead of applying `min` to an empty
changed-row set.

## Tests

- `python -m compileall -q MAB2_recovered/experiments/mechanism_20261122`
- `python MAB2_recovered/experiments/mechanism_20261122/test_p0_permissions_oracle.py` — 6/6
- Full M60 entrypoint on the three frozen manifests — 3456 units completed,
  all eleven controllers completed, zero incomplete units.

The exact command was

```text
python MAB2_recovered/experiments/mechanism_20261122/m60_strong_baseline_matrix.py \
  MAB2_recovered/server_rows1_three_model_manifest.json \
  MAB2_recovered/server_rows2_three_model_manifest.json \
  MAB2_recovered/server_rows4_three_model_manifest.json \
  MAB2_recovered/experiments/mechanism_20261122/P0_M60_STRONG_BASELINE_MATRIX_FROZEN.json
```

## Frozen M60 results

| method | exact set | action error | tokens |
| --- | ---: | ---: | ---: |
| stale | 0.9039 | 0.0428 | 0.00 |
| exact warm cache | 0.8877 | 0.0443 | 0.00 |
| reweighted old history | 0.9167 | 0.0391 | 22.86 |
| CTDR transport only | 0.9167 | 0.0391 | 22.86 |
| CTDR + selective ranking | 0.9167 | 0.0391 | 24.61 |
| full CTDR | 1.0000 | 0.0000 | 230.91 |
| max uncertainty | 0.9167 | 0.0391 | 22.86 |
| periodic | 0.9167 | 0.0391 | 39.81 |
| random | 0.9167 | 0.0391 | 37.34 |
| full restart | 1.0000 | 0.0000 | 878.78 |
| restricted row-refresh oracle | 0.9068 | 0.0391 | 42.71 |

Common post-edit budget is 1924.9615 tokens. Costs are incremental logical
query tokens. The restricted Oracle is an offline hindsight diagnostic over
changed rows, not a global or deployable optimum.

## Freeze files

- `P0_M60_STRONG_BASELINE_MATRIX_FROZEN_PUBLIC.json`
- `P0_BASELINE_FREEZE.json`
- `P0_BASELINE_FREEZE.md`

The freeze ledger separately records Pairwise matched baselines and graph-mask
provenance results with artifact and entrypoint hashes. Future experiments must
create new artifacts when changing the cost or information-access contract.
