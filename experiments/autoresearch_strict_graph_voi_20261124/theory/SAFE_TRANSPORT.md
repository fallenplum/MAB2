# Strict certificate transport proof

Let each terminal cell `j` have non-negative weight `w_j` and label `y_j` in
a finite action set. For a queried set `K`, define `S_a` as the weighted score
of action `a` over `K`, and define `W_U` as the total weight of unqueried
cells. If

```text
S_a* > max_{b != a*} S_b + W_U,
```

then every completion of unqueried labels selects `a*` under weighted argmax.
The score of `a*` in any completion is at least `S_a*`, while another action's
score is at most its known score plus `W_U`.

The bound is tight. With two actions, known scores `S_a*=1`, `S_b=0`, and one
unknown cell of weight `1`, equality allows a completion that selects the other
action. The strict inequality cannot be replaced by equality without a tie
rule.

When no graph source or parent output changes, every descendant remains a
legal cache entry. The dirty set is empty and the controller reuses the old
transcript without a post-edit query. This is the no-drift degeneration used by
the implementation tests.

The proof establishes certificate soundness. It does not establish global
optimality of the mass ranking, which is evaluated separately against random,
periodic and relay-distance orderings.

## Row-level transport radius

For a product of independent row distributions, couple old and new rows so
that row `i` disagrees with probability `delta_i`, its total variation
distance. The probability that any row differs is at most

```text
rho_joint = 1 - product_i (1 - delta_i) <= sum_i delta_i.
```

For an action score in `[0,1]`, the expectation changes by at most
`rho_joint`. An action-gap score can therefore change by at most
`2 * rho_joint`. If an old certificate has known gap `g` and unknown mass
`W_U`, a conservative transport condition is

```text
g > 2 * rho_joint + W_U.
```

The row-level terms cannot be dropped when an edited row changes a descendant
channel. A one-row counterexample with two actions and a complete mass flip
changes the preferred action while all unchanged-row terms remain zero. The
radius is sufficient under the stated coupling assumptions and is not claimed
necessary for every model.

