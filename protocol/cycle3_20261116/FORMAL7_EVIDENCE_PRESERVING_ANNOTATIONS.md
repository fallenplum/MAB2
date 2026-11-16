# Formal7: evidence-preserving annotations as a foundation candidate

## Evidence and scope

Formal6 remains FAILED: G1 EM45.3125% vs SINGLE_Q50.5208%; no threshold changes. D8 input-allocation Ridge/tree failed. D9 current-credit and credit-shift seed correlations are0.6824/0.5502, suggesting more reproducible graph effects than the prior free-form route, not a successful policy. Exploratory D10 finds18/23 G1 destructions versus SINGLE_Q omit a gold support title, with only93/192 observations retaining all support. Gold support is diagnostic only and is forbidden at selection/inference/policy time. The independent full-context reasoning baseline reached61.9792% but costs substantially more tokens; it motivates testing preserved evidence plus reasoning, not declaring that baseline a new method.

Hypothesis H7: lossless forwarding of graph-reachable raw documents plus bounded model-generated annotations avoids the lossy document-ranking bottleneck while retaining reproducible conditional credit. This changes the foundation, not the core claim to novelty. No policy effectiveness is claimed from the pilot. Discarding this route continues the outer loop.

## Fixed operator

Keep the same model pool, title-hash3/3/4 input partition, U/V(Qwen3B), R(Mistral7B), M(Qwen3B), graph masks, node-based common RNG, generation temperature0.7/top_p0.9,128 annotation tokens,256 final tokens,8192 input guard, bf16/eager and exact request cache. Nodes forward every raw document legally available from their own shard and active parents. U/V produce the original fixed source reasoning note; R uses the original relay instruction with its legal documents and incoming notes. M receives a canonical union of reachable documents plus exact incoming notes. No global evidence bypass, supporting facts, selective gold allocation, ranking repair, summary retry or hidden annotation call.

The raw-document payload is not covered by the128-token annotation cap; every raw document and annotation token is charged. Identical evidence IDs alone do NOT certify identical final requests now, since incoming annotations also matter. No Formal6 certificate-coverage claim is transferred.

Eleven conditions: G0,G0_MINUS_EU,G0_MINUS_EV,G1,G1_MINUS_EU,G1_MINUS_EV,FLAT_ANNOTATION,SINGLE_Q,SINGLE_M,INDEPENDENT3_Q,VOTE_Q3. Flat uses only U/V/R to M; SINGLE reads all10 raw documents. INDEPENDENT3 retains its historical Qwen/Qwen/Mistral workers and Qwen final, despite legacy name. VOTE uses3Qwen calls. All baselines share exact cache and legal dead-node elimination. Planned96*2*11 terminal outputs and96*2*36 logical calls.

## Fresh data and fixed gates

96 previously unused HotpotQA train question groups, exact10 documents, exclusions include all known previous MAB2 groups and the full Formal6 input selection. PCG64 input selection261116701, execution seeds26111671/26111672; balanced upstream U/V removal by input order. Official validation/test unopened. No post-inference capacity filtering or failed-unit replacement. Input-only capacity screening and artificial short/long qualification precede admission. Freeze code, manifests and baseline identities before starting formal workers; archive and actual cost close before scoring.

All required: (1) complete fixed matrix with independent execution and scoring checks; (2) G1 EM minus each of SINGLE_Q and FLAT_ANNOTATION >=-0.01 overall and per seed; (3) c1 != c0 fraction >=0.03 overall,>=0.02 per seed,>=20 distinct shifted questions; (4) current-credit cross-seed Pearson >=0.30 and shift cross-seed Pearson >=0.20, constant-vector undefined fails; (5) G1 service utility minus SINGLE_Q >=0.005 overall and strictly positive on both seeds. Utility remains EM-5e-6*(input+output tokens). These new prospective gates do not rewrite Formal6. Two-seed correlations are pilot admission checks, not precise population estimates. Report all baselines, EM/F1, paired corrections/destructions, task/seed decomposition and all costs; no CI/bootstrap.

If all pass, keep this dataset as exposed development only and assess whether there is useful predictor/acquisition signal against free protect-affected/coverage rules before spending on fresh five-seed policy confirmation. Do not use an oracle or another retrospective matrix winner to override any failure.

## Budget and risks

Inherited global160GPUh cap, no reset. Formal7 route ceiling8.5GPUh: <=0.5 engineering and <=8 formal. All attempts, loading, prefix waiting, placement transfers and failed calls are charged. Formal worker aggregate residency cap7.75h plus controlled stop margin; no paid resources and no other projects terminated. Admission reserves only verified remaining capacity. Use memory-qualified shared cards and inspect actual processes/logs, not utilization guesses.

Advantages: directly addresses observed evidence omission with preserved information and tests the original communication-credit question. Risks: higher input cost, annotations may introduce noise, strong single/independent baselines may dominate, a free structural rule may eliminate the need for paid feedback, and fixed small sample gates may reject a weak/noisy route. Positive applicability is insufficient for ACL claims; maintain negative outcomes and complete-budget mechanism comparisons.
