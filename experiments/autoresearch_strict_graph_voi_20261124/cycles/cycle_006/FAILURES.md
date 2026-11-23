# Cycle 006 confirmatory record

The bit-mask implementation of the bounded horizon-2 policy was evaluated on five locked seeds, four DAG families, four drift levels, two cost regimes, two edit counts and 500 cases per family-condition. This gives 160000 cases per method. It achieved action exactness 1.0000 and mean cost 26.831233, versus 30.046724 for masked Pairwise. The aggregate reduction is 10.70 percent, above the ten-percent CPU gate.

Family reductions are chain 23.73 percent, fanout 10.54 percent, diamond 7.00 percent and mixed 6.41 percent. The aggregate gate passes, while the family breakdown shows weaker gains for convergent and mixed graphs.

The fast bit-mask implementation uses first-seen label-branch order, `float(np.mean(completion))` at the rollout leaf, exact floating-point comparison followed by lexical `(kind, index)` tie-breaking, and a guard that excludes cached terminals. It is verified against the cycle006 smoke aggregate. It is not asserted to be bit-for-bit trajectory-identical to the exploratory Python-dictionary implementation in cycle005. Both use the same information contract and strict certificate.

