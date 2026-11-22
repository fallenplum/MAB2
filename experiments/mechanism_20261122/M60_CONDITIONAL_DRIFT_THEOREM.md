# Conditional-drift certificate theorem

## Setup

For relay row `i`, let `q_i` be its categorical output distribution, `s_i >= 0` the current source weight, and `b` the base distribution. Let `J_y = {j : g(i,j)=y}` denote the cells contributing to decision label `y`. The conditional credit is

`F_y = sum_i s_i sum_{j in J_y} (q_i(j) - b(j))`.

After a graph edit, rows in a manifest descendant mask `U` may change. For an unqueried row, assume only

`TV(q_i_new, q_i_old) <= delta_i`.

## Sound interval

For each unqueried row and label, the grouped probability difference obeys

`|sum_{j in J_y}(q_i_new(j)-q_i_old(j))| <= delta_i`.

Therefore the post-edit credit lies in

`L_y = sum_i s_i sum_{j in J_y}(q_i_old(j)-b(j)) - sum_{i in U} s_i delta_i`

and

`U_y = sum_i s_i sum_{j in J_y}(q_i_old(j)-b(j)) + sum_{i in U} s_i delta_i`.

If every label interval is strictly above or below the action threshold `tau`, the edit decision is unchanged for every conditional distribution satisfying the TV assumptions. This certificate does not depend on a learned predictor.

## Query lower bound

Suppose an unqueried row has an interval that crosses `tau` for label `y`. Two conditional distributions can be constructed inside the allowed TV ball with grouped mass on opposite sides of `tau`. They produce the same observed transcript on all queried rows and induce opposite keep/remove decisions. Any exact policy that sees the same history must take the same action in both worlds, so it cannot be correct in both. A row whose interval crosses the threshold is therefore information-theoretically unresolved under the stated uncertainty set.

The lower bound is local to the unqueried row and does not claim that every implementation must query all edited rows. A graph descendant mask can shrink `U`, and a smaller certified `delta_i` can shrink the interval.

## Empirical falsification

The synthetic sweep in `M60_CONDITIONAL_DRIFT_SAFE_POLICY.json` used 5,000 instances per cell. Source-only transport retained a false-safe rate that grew with conditional drift. The TV interval policy had exact decision recovery of 1.0 in all 12 cells and queried more rows as drift or edit density increased. A question-disjoint real-output transfer calibrated empirical radii from training rows. It recovered exactness on SciQ and CommonsenseQA at roughly twice the maximum-uncertainty token cost, while QASC exposed a false-safe case under an empirical radius. This separates a soundness theorem with known radii from the practical problem of estimating those radii.

## Claim boundary

The result is a mechanism and lower-bound contribution. It does not establish an improved end-to-end answer-quality method or a universal cost reduction. Those claims require an independent task cohort with real graph metadata and matched service accounting.

