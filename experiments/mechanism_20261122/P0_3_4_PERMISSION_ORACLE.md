# P0-3/P0-4 permission and oracle audit

## Baseline information access

Every online controller receives the same old paid transcript and post-edit
source. Stale now replays the old transcript directly. It does not reconstruct
an old decision from the full old relay table or sealed labels. Warm and
reweighted controllers reuse the same history and may only acquire legal
post-edit actions through the common audit scheduler. Max-uncertainty,
periodic and random policies use the same legal action set and new relay value
only when the selected action is charged. Full CTDR receives the graph-manifest
affected-row set as an explicit method input. The manifest is not derived from
sealed labels or a fresh target decision.

The audit also found and fixed a replay permission bug. Reweighted history and
source-only CTDR were passing `old_relay` into the post-edit replay function,
which allowed a new query to read an old-graph value. They now pass `new_relay`.
The same correction was applied to the independent ARC and interface-semantic
scorers. The existing aggregate matrix must be regenerated after this
correction.

## Oracle semantics

`oracle_row_refresh` is explicitly a hindsight diagnostic. Target labels select
the cheapest subset among changed-row refresh actions. The selected actions are
then replayed independently through `_row_refresh_prediction` before scoring.
The target set is never returned directly as the oracle prediction. The oracle
is restricted to changed-row row actions and is not a global optimum over all
cell-level schedules. Its cost uses the same `cost(actions, tokens)` function as
every other controller.

The public summary records these fields explicitly:

- `oracle_selection = hindsight_target_selection_only`
- `oracle_prediction = independent_row_refresh_replay`
- `oracle_scope = restricted_changed_row_actions_not_global`

This prevents the diagnostic from being interpreted as a deployable baseline or
as a globally optimal cost curve.

Unfinished warm-cache or oracle plans are recorded with `completed=0` and an
`incomplete` count. They are not converted into a valid empty decision set.

## Remaining boundary

The matrix still uses sealed targets for post-hoc exact-set and action-error
scoring. They are outside every controller's information path. The oracle's
hindsight selection is retained only as a lower-bound diagnostic. Any future
online comparison must exclude it from the deployable Pareto frontier.

