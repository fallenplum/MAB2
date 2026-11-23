# Pairwise Gap Certificate theory scope

For answer label \(y\), define

\[
F_y(s,q,b)=\sum_i s_i\sum_{j:g(i,j)=y}(q_i(j)-b(j)).
\]

For an incumbent action \(a\) and competitor \(b\), let \(D_{ab}=F_a-F_b\). If
the conditional relay row \(q_i\) changes to \(q_i'\) with

\[
\operatorname{TV}(q_i,q_i')\leq\delta_i,
\]

then the mass of either answer group changes by at most \(\delta_i\). The pairwise
gap therefore changes by at most \(2s_i\delta_i\). For an unresolved row set
\(U\), the certificate is

\[
\underline D_{ab}=D_{ab}^{\mathrm{cached}}-
\sum_{i\in U}2s_i\delta_i.
\]

If \(\underline D_{ab}>0\) for every \(b\neq a\), every admissible post-edit
conditional table has the same unique action \(a\). Querying row \(i\) removes
its term from the uncertainty sum and replaces its cached row with the new
observation. The controller stops only after all pairwise lower gaps are
positive.

The proof is conditional on the supplied TV radii and a complete changed-row
mask. A missing graph manifest or relay-interface fingerprint causes a
fail-closed all-row mask. Empirical quantiles are development estimates and do
not themselves provide finite-sample coverage. The gap-per-token ordering is a
cost heuristic and does not claim global optimality.

The certificate targets the final answer action. It does not certify every edge
sign when the route stops before all downstream conditional cells are acquired.

