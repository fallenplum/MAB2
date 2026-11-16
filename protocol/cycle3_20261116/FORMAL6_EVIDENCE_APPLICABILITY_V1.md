# Formal candidate 6: evidence-indexed graph applicability pilot v1

Frozen before new-model outcomes. This is an applicability screen, not proof of a useful revalidation policy or ACL novelty. D6/D7 and earlier failures remain unchanged. User continuation permits new candidates; cumulative GPU budget remains 160 hours.

## Hypothesis and competing explanation
Free-form notes can discard useful evidence and destabilize credit. Passing selected original documents may improve graph quality and give exact, label-free certificates for some zero-credit deletions. Competing explanation: full-context single models or free graph coverage dominate; communication credit remains sparse or stochastic. Exact prompt reuse and evidence compression are known techniques, not independent novelty claims.

## Frozen implementation
Ten raw documents sorted by SHA256(raw title), original index tie: first three at U, next three V, last four R. U/V use Qwen2.5-3B; R Mistral-7B-v0.3. Each selects at most two supplied document IDs, strict unique JSON list. Invalid selections become empty, are counted as failures, and never retried. R sees its own documents plus incoming selected original documents. Qwen M sees only canonical union of incoming documents. Original strings retained. Source cap128, final256, input8192, temperature0.7, top_p0.9. Same node RNG per question/seed across graph conditions; exact request reuse and terminal ancestor elimination apply to every method.

G0 edges U-R,V-R,R-M,U-M,V-M. Exogenous removal U-R or V-R is input-only balanced. Collect G0/G1 and each minus U-M or V-M; FLAT_EVIDENCE has only U-M,V-M,R-M with same 3/3/4 access. SINGLE_Q and SINGLE_M see all ten original documents using same final prompt. Historical INDEPENDENT3_Q and VOTE_Q3 retained unchanged. Eleven terminal conditions, 36 logical node calls per seed. Actual physical calls and all logical costs recorded separately. Labels/supporting facts forbidden in inference and selection.

## Input and scope
Input-only deterministic sample of 96 HotpotQA train questions, seed261116601, excluding all previous formal/diagnostic IDs and the old 1024-question input pool. Two generation seeds26111661/26111662. Official validation/test remain sealed. Hash selected rows and source snapshot before inference. No input replacement after quality is visible; rejection/partial output fails completeness. Up to two shared existing A6000 GPUs after process and free-memory checks. No other projects stopped.

## Engineering then scientific gates
First two artificial fixtures x two engineering seeds26111601/26111602. Require complete trace, replay consistency, no invalid selections and no information leakage. Artificial correctness is not scientific evidence. Engineering allocation0.25 GPUh, worker hard cap840 seconds. Formal allocation7.75 GPUh including termination slack, worker residence aggregate cap7.5h; prefix2/worker throughput projection1.5x must fit. New whole route allocation8GPUh maximum, within existing cumulative160. Failed/partial/load time charged; never impute reservation as actual cost.

Formal admission gate (all required; fixed G1 is nominee, no post-hoc graph winner):
1. All96 x2 x11 terminal conditions complete and trace/cost replay passes; selector valid rate >=0.98.
2. G1 EM minus SINGLE_Q and FLAT_EVIDENCE each >= -0.01 overall and separately in each seed.
3. For each direct edge define c_t=EM(Gt)-EM(Gt-minus-edge). Fraction of q/seed/edge with c_1 != c_0 >=0.03 overall and >=0.02 each seed, with at least20 distinct questions showing any shift.
4. Equal final evidence-set certificate fraction >=0.10 across all q/seed/edge/time pairs, with exactly zero unequal-final-request or unequal-answer violations among certified pairs.

Report EM/F1, per-seed/edge/task/type counts, selection failures, fixed action utilities EM-5e-6*(input+output tokens), single/independent/flat comparisons, paired correction/destruction, physical GPU residence and logical tokens. No CI/bootstrap. These gates only justify a separate frozen active-policy experiment on new data. They do not require beating strong baselines yet and cannot be described as an effective final method. Failure freezes this architecture and triggers outer-loop analysis under remaining budget; it is not project stop.

## Advantages and risks
Advantages: exact document-level certificates, explicit evidence permissions, strong free baselines, no trained predictor requirement. Risks: coarse documents increase transmission cost; selector errors discard evidence; certificate coverage may be trivial; upstream changes may still have weak/stochastic effects; prior work already covers message pruning and exact trace reuse. Any future policy must beat free coverage/protect-changed, random, periodic, uncertainty and cost-aware VOI at equal complete cost, on independent data.
