# Boundary and negative findings

Exact cache is cheaper than certificate revalidation when its stale terminal
decision remains correct. The candidate is therefore not claimed to dominate
exact cache under an unconstrained cost objective. On QASC Qwen-0.5B rows2
held-out and SciQ Qwen-14B semantic rows4 held-out, exact cache loses action
exactness while the certificate route recovers the full-graph action.

Certificate ranking was compared with random, periodic and relay-distance
ordering. All routes used the same stopping certificate and private model
calls. Mass-aware ordering used fewer queries in the semantic ablation.

