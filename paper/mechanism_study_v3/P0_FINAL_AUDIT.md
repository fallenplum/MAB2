# P0 evaluation-integrity final audit

## Status

P0-1 through P0-6 are complete for the released evaluation scope. The branch
contains the repaired code, tests, redacted summaries and source-matched PDF.

## Gate results

| gate | status | evidence |
| --- | --- | --- |
| P0-1 placeholder cleanup and source/PDF match | PASS | `P0_1_AUDIT.md`, six-page PDF, source SHA `1a1a8420...79623`, PDF SHA `d93703d...f4cb5` |
| P0-2 `None` versus proved empty set | PASS | `test_none_semantics.py`, completion and incomplete fields in the strong matrix |
| P0-3 baseline information permissions | PASS | old transcript is the only reusable history, post-edit queries read `new_relay`, graph mask comes from the manifest |
| P0-4 Oracle replay and cost | PASS WITH SCOPE | independent row-refresh replay and common cost function. The Oracle is hindsight-selected and restricted to changed-row actions, not a global optimum |
| P0-5 manifest-matched Pareto | PASS | budget violations 0, same graph mask, transcript, source, prices, budget and stopping target |
| P0-6 task-quality transitions | PASS WITH SCOPE | 6,528 per-question score rows. TCR-guard transitions report 348 corrected and 219 damaged answers relative to warm, max-uncertainty and source-only controls. A separate full-CTDR per-question service trace is unavailable |

## Corrected strong-matrix interpretation

After the permission and empty-set fixes, the aggregate strong matrix is

- reweighted old history and CTDR transport only 0.9167 exact set, 22.86 tokens
- CTDR selective ranking 0.9167 exact set, 24.61 tokens
- full CTDR 1.0000 exact set, 230.91 tokens
- full restart 1.0000 exact set, 878.78 tokens
- restricted hindsight row-refresh diagnostic 0.9068 exact set, 42.71 tokens

The previous Oracle exactness and lower cost numbers are superseded. The
restricted Oracle is retained as a replayable diagnostic and is excluded from
the deployable Pareto claim.

## Remaining scope boundary

The legacy four-panel module ablation uses a 23-field schema without separate
old/new relay channels. It remains historical negative evidence and is not
pooled with the corrected 11-row strong matrix. Full-CTDR task-answer quality
requires a new per-question service trace with the corrected scorer.
