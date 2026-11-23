# P3 Phase 2 real-service contract

This contract evaluates graph-conditional certificate revalidation on two
independent natural-language MCQA tasks and two local checkpoints. The input
panels use the server's sealed QASC and SciQ interface-drift fixtures. The
online methods share the old transcript, changed relay set, tokenizer, model,
and exact token ledger. Sealed answer keys are used only after route actions
are selected.

The primary algorithm is `certificate_voi`. It queries changed relay rows,
orders dirty terminal cells by graph-conditioned terminal mass, and stops only
when the strict weighted decision certificate is satisfied. `certificate_random`,
`certificate_periodic`, and `certificate_uncertainty` use the same dirty cone,
costs, and stopping rule. They differ only in terminal ordering. `exact_cache`,
`graph_mask_recompute`, `topology_voi_online`, `full_new_graph`, and stale reuse
remain fixed reference routes.

The confirmatory panel is locked to the first 32 rows of the server fixture,
with rows 1–16 as development and rows 17–32 as held-out. The full 128-row
semantic extension is reported separately as exploratory. The primary gates
are held-out action exactness equal to full new graph and lower token/forward
cost than graph-mask recomputation. No confidence intervals or bootstrap are
computed under the project contract.

This contract does not authorize target-label access during route selection or
claim improvement over exact cache when exact cache already satisfies the same
risk target.

