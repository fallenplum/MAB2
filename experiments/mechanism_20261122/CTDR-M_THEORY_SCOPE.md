# CTDR-M mechanism and theorem scope

Let $q_i$ be the post-edit source weight, $b_j$ the terminal prior, and $R_i(j)$ the conditional relay distribution for relay row $i$. The finite audit credit for answer label $y$ is

$$C_y(q,R)=\sum_i q_i\sum_j b_j\bigl(R_i(j)-b_j\bigr)\mathbf{1}[T_{ij}=y].$$

## Certificate transport theorem

Assume a graph manifest identifies a set $S$ containing every relay row whose interface can change. Assume every row outside $S$ has an unchanged interface fingerprint, and every terminal input signature is unchanged. Revalidate the relay rows in $S$, reuse only the old observations outside $S$, and continue the exact finite decision-region audit. If the audit stops with a decision, the returned set equals the full post-edit audit decision set.

The proof is an information-preservation argument. For $i\notin S$, the interface and terminal signatures make the old relay and terminal observations valid under the post-edit graph. For $i\in S$, the new relay observation replaces the old conditional row before any credit is accumulated. The remaining unseen cells are bounded by the scheduler's lower and upper pending-mass bounds. Therefore each returned keep or drop action satisfies the same threshold certificate as a full post-edit audit. The theorem permits unnecessary revalidation and forbids an omitted changed row.

## Counterexample scope

If one changed row is omitted from $S$, cached weights can cross the decision threshold. In the rows1 wrong-manifest control, replacing the true relay-1 edit with relay 0 reduced exact-set recovery to 96.875% over 768 question–model units while cold restart remained exact. If an interface fingerprint is missing, CTDR-M must mark all relay rows as affected and fall back to full relay revalidation.

## What the theorem does not establish

The theorem covers a fixed finite credit interface and exact conditional observations. It does not prove a better language-model answer accuracy, general graph optimization, lower wall-clock latency, or superiority of the generic decision-region scheduler. Those claims require separate task-level and deployment evaluations.

