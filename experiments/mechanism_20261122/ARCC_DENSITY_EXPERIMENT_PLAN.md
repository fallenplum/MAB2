# ARCC independent graph-density experiment

## Hypothesis

After a graph edit, revalidating only the affected relay rows should preserve the final answer decision with lower incremental feedback cost than revalidating every row. The benefit should be largest for one-row edits and should degrade monotonically as the number of affected rows grows.

## Cohorts

- `arcc_independent_rows1_INPUT.json` changes relay row 1.
- `arcc_independent_rows2_INPUT.json` changes relay rows 1 and 2.
- `arcc_independent_rows4_INPUT.json` changes all four relay rows.

All cohorts contain the same 128-question ARCC holdout and share the same gold answers. Only the graph edit density changes. The worker derives `changed_rows` from `second_facts_new`.

## Methods

Evaluate stale reuse, warm reweighted replay, maximum uncertainty, and sequential conditional answer-safe revalidation. Every method receives the same old transcript and new source distribution. New feedback cost is counted from the new relay rows and downstream terminal cells selected after the edit. A scorer must not use unqueried terminal labels while charging only relay-row cost. If terminal labels are already cached, that cache is shared by every method and its cost is reported separately.

## Primary measurements

- final answer accuracy against the private ARCC gold map
- graph decision exactness against the full new-graph certificate
- answer-action exactness
- mean incremental tokens and number of queried relay rows
- stale-error correction and new-error destruction counts

The central mechanism claim requires the sequential policy to retain full-mask answer accuracy while reducing incremental tokens as edit density falls. A failure at rows2 or rows4 is retained as a boundary result.

