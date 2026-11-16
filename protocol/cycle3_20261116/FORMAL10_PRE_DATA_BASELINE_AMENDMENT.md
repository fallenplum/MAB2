# Formal10 fairness hardening before real-data inference

Only artificial GPU qualification is running. No pilot model output or hidden score has been collected. Preserve the original contract and strengthen its controls now, with no metric/threshold relaxation:

1. SINGLE_Q/SINGLE_C do not need execution feedback to choose an answer. Their observed visible tests remain diagnostic collection costs; nominal service cost excludes those unnecessary executions. All actual collection CPU/GPU charges remain in the ledger.
2. Add BOK3_EARLY: sample/test the same frozen BOK3sequence but stop at the first program passing both visible cases, or after three. This selects exactly the same program as BOK3's maximum-visible-score/first tie rule while charging only its necessary prefix. It is a derived exact-prefix replay, not a new generation request or fitted policy. Include it in EVERY strong-control utility/quality gate and seed-consistency check with the unchanged.01/-.02/0 thresholds. Keep original BOK3reported.
3. Report REUSE_HISTORY6 as an additional full-history fairness diagnostic, already described in the original contract. It is not the input to VS_GRAPH and does not license free historical programs in later methods.

The collected matrix remains10conditions*16tasks*3seeds. The scored table now has13conditions including VS_GRAPH,REUSE_HISTORY6,BOK3_EARLY. This amendment is frozen before real model inference and cannot be revised after pilot outcomes. Its purpose is to remove unnecessary baseline costs, not make a failed candidate pass.
