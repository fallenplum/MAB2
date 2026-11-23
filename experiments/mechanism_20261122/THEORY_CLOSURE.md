# Theory closure for graph-conditional certificate transport

This note closes the scope of the transport claim. It distinguishes a
sufficient certificate from a worst-case necessary condition, and separates
the safety stopping rule from the row-selection heuristic.

## Setup

For answer labels (a,b), let (A_a(i)) and (A_b(i)) be the relay
condition groups mapped to the two labels in row (i). With post-edit source
weights (s_i), base distribution (r), and row conditional (q_i), define

\[
F_a(s,q)=\sum_i s_i\{q_i(A_a(i))-r(A_a(i))\},\qquad
D_{ab}=F_a-F_b.
\]

After the edit, some rows have been observed with (q_i^{+}). For an
unobserved row (i\in U), the old transcript supplies (q_i^{-}), and the
transport assumption is

\[
\operatorname{TV}(q_i^{+},q_i^{-})\leq\delta_i.
\]

The cached gap \(\widehat D_{ab}\) uses observed post-edit rows and the old row
for every (i\in U).

## Theorem 1: safe transport for an answer action

If every unresolved row satisfies the stated TV bound and

\[
\underline D_{ab}
 =\widehat D_{ab}-2\sum_{i\in U}s_i\delta_i>0
 \quad\text{for every }b\neq a,
\]

then (a) is the unique answer action for every admissible post-edit relay
table. Total variation bounds the change in each answer-group probability by
\(\delta_i\). Therefore an unresolved row can reduce the pairwise gap by at
most \(2s_i\delta_i\), and summing the row-wise bounds gives the displayed
lower gap. Strict positivity separates the incumbent from every competitor.

The condition is sufficient for the supplied uncertainty set. A development
quantile used to estimate \(\delta_i\) is not a finite-sample guarantee without
an exchangeability or coverage assumption.

## Source-distribution transport

When the source distribution changes from (s^{-}) to (s^{+}), define

\[
h_{i,ab}(q_i,r)=q_i(A_a(i))-q_i(A_b(i))
                 -r(A_a(i))+r(A_b(i)).
\]

For a fixed conditional table, the exact source contribution is
\(\sum_i(s_i^{+}-s_i^{-})h_{i,ab}\). A conservative source radius is

\[
\rho_{ab}^{\mathrm{src}}
 =\sum_i|s_i^{+}-s_i^{-}|\,|h_{i,ab}|.
\]

Combining source drift and unresolved conditional drift gives

\[
\underline D_{ab}
=\widehat D_{ab}
-\rho_{ab}^{\mathrm{src}}
-2\sum_{i\in U}s_i^{+}\delta_i.
\]

The (h_{i,ab}) form is the tight row-wise source bound when the conditional
table is fixed. Replacing it by (2\sum_i|s_i^{+}-s_i^{-}|) is valid but looser.
A source-only radius cannot certify conditional relay drift. Conditional drift
requires row revalidation or a valid row-level radius.

## Tightness and necessity boundary

The coefficient (2s_i\delta_i) is tight under only a TV constraint. Consider
one row, two answer groups, source mass (s_i>0), and no base effect. Let

\[
q_i^{-}(A_a)=\delta,\quad q_i^{-}(A_b)=1-\delta,
\qquad q_i^{+}(A_a)=0,\quad q_i^{+}(A_b)=1.
\]

Then \(\operatorname{TV}(q_i^{+},q_i^{-})=\delta\), while the pairwise gap
decreases by exactly (2s_i\delta). A smaller universal coefficient would
certify this admissible counterexample incorrectly. The row-level TV term is
therefore necessary for a distribution-free certificate.

Strict positivity of every \(\underline D_{ab}\) is minimax-necessary when the
only knowledge about unresolved rows is their TV balls and the balls are
otherwise unconstrained. It is not necessary for one fixed realized table. If
a lower gap is nonpositive, additional structure or a new observation can
still prove the realized action.

## No-drift degeneration

If the graph, relay interfaces and source distribution do not change, then

\[
U=\varnothing,\qquad s^{+}=s^{-},\qquad q_i^{+}=q_i^{-}.
\]

Both transport radii are zero and \(\underline D_{ab}=\widehat D_{ab}). The
controller performs no feedback query and returns the stale transcript action.
This is the formal degeneration of CTDR to stale reuse. If only source weights
change, the source term remains and stale reuse is not certified without it.

## Selective ranking versus uncertainty sampling

The stopping theorem is invariant to query order. The one-step uncertainty
removal for row (i) is (m_i=2s_i\delta_i). With token price (c_i), the
cost-aware score is (m_i/c_i). Maximum uncertainty is equivalent only when
it uses the same (m_i) and all prices are equal. It has no separate safety
guarantee. Selective ranking can reduce cost under unequal prices or
pair-specific unresolved gaps, but it cannot strengthen the certificate.

## Falsifiable predictions

1. Removing the row-level TV term produces an admissible action reversal in the
   tight counterexample.
2. At a fixed stopping rule, changing only row order changes cost but not the
   logical certificate after the required rows are observed.
3. With no graph, interface or source drift, CTDR makes zero queries and
   matches stale reuse exactly.
4. Under unequal row prices, (m_i/c_i) can be cheaper than uncertainty-only
   ordering while preserving the same action certificate.

