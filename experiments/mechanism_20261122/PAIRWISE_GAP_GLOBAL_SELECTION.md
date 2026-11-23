# Pooled quantile selection for Pairwise Gap Certificate

Per-cell quantile selection is useful for mechanism diagnosis, but it can
overstate deployment stability. This panel chooses one quantile on the pooled
selection residues of the 27 model--task--density cells, then evaluates each
held-out residue once. The pooled choice is `q=0.9`. It preserves answer
actions on all 27 test residues with mean 124.74 logical relay tokens. The
q=0.5 and q=0.75 candidates fail the pooled selection action gate, while q=0.9,
q=0.95 and q=1.0 pass with increasing cost.

Leave-one-task-out and leave-one-model-out selections also preserve action
exactness on every held-out panel. The selected quantile ranges from 0.5 to
0.9, which indicates that the mechanism is not tied to the per-cell q=0.5
diagnostic. The complete records are in `PAIRWISE_GAP_GLOBAL_SELECTION.json`.

