# Answer-region revalidation as a graph-density mechanism

This note states the mechanism that can be tested with the existing graph-edit
experiments. It is intentionally limited to action certification and feedback
cost. It does not claim an increase in the underlying model accuracy.

## Setup

Let the final answer labels be \(a\in\{1,\ldots,K\}\). After a graph edit,
some relay rows are reused from the old transcript and a set \(U\) of rows is
uncertain. Let \(\widehat z_a\) be the cached contribution to answer \(a\),
including all rows already queried on the edited graph. For each unqueried row
\(i\in U\), let \(r_{i,a}\) be a valid upper bound on the change in its weighted
contribution to answer \(a\). A label-agnostic TV bound gives

\[
r_{i,a}\le s_i\delta_i,
\]

where \(s_i\) is the current source mass and \(\delta_i\) is the calibrated
conditional drift radius. The resulting answer interval is

\[
L_a=\widehat z_a-\sum_{i\in U}r_{i,a},\qquad
U_a=\widehat z_a+\sum_{i\in U}r_{i,a}.
\]

The policy stops with action \(a\) when

\[
L_a>\max_{b\ne a}U_b.
\]

This condition certifies the final answer action while leaving individual
edge decisions unresolved.

## Proposition 1: answer-region soundness

Assume each unqueried row satisfies the stated drift bound. If

\[
L_a>\max_{b\ne a}U_b,
\]

then every conditional relay distribution inside the uncertainty set selects
the same answer \(a\). The proof is direct because every admissible post-edit
answer score lies in its interval. This is a decision-region certificate,
whereas a graph-edge certificate additionally requires every edge score to be
separated from its edit threshold.

## Proposition 2: local information lower bound

If the interval for an answer pair \(a,b\) overlaps,

\[
L_a\le U_b,
\]

the uncertainty set contains admissible row distributions that can reverse the
ordering of \(a\) and \(b\). A policy that receives no new information about
the rows responsible for this overlap cannot be exact in both worlds. Querying
is therefore necessary only for unresolved answer pairs, rather than for every
edge whose individual sign is unknown.

## Why graph density changes stale-credit risk

Let \(m=|U|\) be the number of edited or interface-affected rows. Suppose each
row independently has probability \(p\) of changing the action-relevant
ordering enough to invalidate stale credit. The stale action failure rate is

\[
1-(1-p)^m,
\]

which increases with edit density \(m\). The same monotonicity holds whenever
the row events are positively associated. This gives a falsifiable explanation
for the observed rows1-to-rows4 degradation of stale action agreement.

The answer-region policy does not remove this uncertainty. It aggregates it
into the final answer margin. Its query cost rises with \(m\) and with the
drift radius \(\delta_i\), but it can stop before certifying all edge signs when
the surviving answer margin is already larger than the aggregate uncertainty.

## Testable predictions

| Prediction | Required observation | Falsification |
|---|---|---|
| Density increases stale failure | stale action error rises from rows1 to rows4 at fixed task and model | no monotonic increase across independent tasks |
| Drift increases stale failure | synthetic or real conditional drift increases stale action error | stale remains unchanged as drift grows |
| Answer-region saves cost at fixed action safety | lower relay tokens than edge-wide TCR while action agreement stays equal | no cost gap or action errors at the fixed radius |
| Savings depend on answer margin | large answer margins stop early, near ties query almost all rows | query count is independent of margin |
| Interface drift defeats oracle masks | changing an undeclared relay interface causes mask-local safety failures | mask-local policy remains exact under interface changes |
| All-row fallback restores safety | revalidating all rows removes mask-local false-safe cases | all-row policy still misses changed actions |

## Evidence currently available

The three-task, three-density cohort shows stale action errors increasing with
graph-change density. With a fixed \(q=0.9\) radius and all-row revalidation,
answer-region certification keeps answer-action exactness at 1.0 in the listed
cells while saving roughly 39%--57% relay tokens against TCR guard. The QASC
rows2 cell has decision exactness 0.976, so the mechanism is reported as action
safety with a residual decision-boundary failure rather than unconditional
exactness. Historical held-out service traces show the same quality-preserving
cost pattern.

The 51-cell orthogonal ablation isolates the graph-conditioned components. With
all four rows exposed to the certificate, a row-specific radius with
source-aware token priority uses 141.07 relay tokens on average. Replacing the
row-specific radii with one pooled radius raises the cost to 253.24 tokens at
the same 1.0 answer-action exactness. This larger gap includes a zero-radius
interface-invariance assumption for rows with no observed calibration drift.
Under the same trusted changed-row mask, the row-specific radius uses 120.00
tokens versus 122.79 for a pooled radius, and source-aware priority lowers the
pooled control from 126.64 to 122.79 tokens. The component effects are therefore
separated from the graph-mask prior. This cost pattern is an empirical
mechanism result, not an accuracy improvement.

The historical radius frontier gives 127.72, 134.05, 141.07, 143.67, and
150.90 relay tokens at (q=0.5,0.75,0.9,0.95,1.0), respectively. Final answer
action exactness stays at 1.0 across the frontier, while decision exactness
reaches 1.0 at (q=1.0). This provides a fixed cost-safety operating curve
without tuning on the held-out rows.

The evidence does not establish improved answer accuracy. It supports a
mechanism claim about graph-density-induced stale-credit failure and a
decision-region reduction in feedback cost. Independent interface-drift cells
are required before treating a descendant mask as safe without the all-row
fallback.

The hidden-interface panel makes this boundary causal across Qwen2.5-0.5B,
Qwen-14B, and Llama-3.2-1B. A fact-only mask under an undeclared row-3
interface change loses answer-action exactness on Qwen2.5 and Llama. Including
the interface fingerprint restores exact answer actions for all three models
at q=1.0, with 15.8%--66.4% lower relay cost than TCR guard. Qwen-14B retains
one decision-boundary miss, so the claim is stated at the answer-action level.
The fingerprint is part of the safety condition, rather than metadata used only
for logging.

