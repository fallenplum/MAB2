# P3 Phase 2 real-service report

The confirmatory rows2 panel used QASC and SciQ with Qwen-0.5B and Qwen-14B.
It evaluated 32 rows per task and checkpoint, with the first 16 development
and the last 16 held out. The certificate route matched full new-graph action
exactness on every held-out panel. It reduced calls against graph-mask
recomputation from 18 to 2–3 and preserved the full-graph answer accuracy.

The QASC Qwen-0.5B held-out panel is the one case where exact cache had lower
cost but lost action exactness at 0.938. The certificate route recovered full
action exactness at 3 calls. Other rows2 panels already had exact-cache action
exactness 1.0, so the certificate route is not claimed to dominate exact cache
under an unconstrained cost objective.

The exploratory semantic-rows4 128-row extension used 64/64 development and
held-out partitions. Certificate VOI retained action exactness 1.0 on all four
held-out task/checkpoint combinations and used 7–9 calls versus 20 for graph
mask and 21 for full graph. A separate 32-row ranking ablation showed lower
calls for mass ranking than periodic, maximum relay-distance and deterministic
random ordering, with identical action exactness.

The online route never read sealed labels. Full-information rows and answer
accuracy were used only after route selection. The raw rows, private keys and
per-question model outputs remain on the server.

