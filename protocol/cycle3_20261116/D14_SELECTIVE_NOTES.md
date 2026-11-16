# D14: bounded development intervention on selective note removal

## Hypothesis and status

D13's all-or-none suppression can remove useful notes together with abstaining ones. Test the distinct hypothesis that removing only notes matching the SAME frozen abstention lexicon preserves useful evidence and improves full-cost service. This is exposed-development screening and a baseline, not a new scientific contribution, independent confirmation, or a change to failed Formal7/8/D13 gates. No lexical tuning, larger predictor or model-pool change.

Use all96 questions and both frozen generation seeds on EACH task. Four variants: G1_SELECTIVE, G1_RANDOM_MATCH, FLAT_SELECTIVE and INDEPENDENT_SELECTIVE. Selective removes exactly the notes matching D13's word-boundary lexicon. Random removes the same number of notes per question/seed from G1, ordered by SHA256 of a fixed domain tag, policy seed261116141, question ID, generation seed and sender name. Preserve retained note order and every other terminal-generation argument. All policies pay their entire original prefix, including removed notes. Flat and independent controls receive the same filter. Existing SINGLE_Q, VOTE_Q3, unfiltered INDEPENDENT3_Q and D13 always-ignore/ANY baselines remain controls.

## Cache, execution and audits

The input-only inventory contains1536 logical terminal entries and669 unique new generation requests. Exact reuse requires equality of question fingerprint, node, model, pinned model identity, complete messages, RNG seed, generation cap and configuration. Protocol metadata alone is not an HFBackend generation argument; retain original source request and archive provenance separately. Reuse is restricted to verified archived generations, never approximate text similarity. New requests contain inputs and generated notes only, no gold labels.

Use the same pinned Qwen2.5-3B backend, bf16/eager, generation settings and maximum input length as the parent runs. Independent process-isolated model calls; no retry within an attempt. Qualification uses artificial prompts only. Persist every request start, returned generation and native cost event. Stop on input/capacity, source, model identity, disk, or execution mismatch. All attempts and qualifications count against the inherited160GPUh.

The entire D14 ceiling is2GPUh: at most.2 qualification and1.8 collection. Reserve before launch, cap aggregate worker residence at1.75h with stop margin, and verify source/archive/cost closure before computing any new quality scores. A partial attempt is preserved and charged, never selectively scored. At most two shared GPUs with at least20000MiB free at prelaunch; examine actual processes/logs. No other jobs stopped, no paid resources.

## Fixed development decision

Report all variants and all existing controls, EM/F1, exact paired corrections/destructions, task/seed breakdown, logical prefix-plus-terminal tokens and actual added GPU cost. New-inference cached hits and physical requests remain separate. No CI/bootstrap. A single random physical control is only a cheap screen, not random-policy robustness evidence.

The SAME G1_SELECTIVE variant must show utility advantage >=.005 overall on each task and strictly positive on each generation seed against EACH of: direct SINGLE_Q, VOTE_Q3, unfiltered INDEPENDENT3_Q, filtered INDEPENDENT_SELECTIVE, filtered FLAT_SELECTIVE, matched G1_RANDOM_MATCH, and the D13 ANY/always-ignore baselines. Utility remains EM-5e-6*(input+output). Any failed comparison rejects D14 as a basis for formal credit-policy confirmation; no task pooling or threshold adjustment. A pass still needs fresh full-matrix credit reliability tests and independent multi-seed active-policy evaluation with full acquisition costs.

Advantages: directly tests an identified all-or-none information-loss mechanism, uses exact cache to reduce compute, and shares filtering with strong controls. Risks: lexical abstention may be a poor quality cue, retained unsupported facts can still mislead, costs can dominate, and this is already-exposed small development data. A failed D14 triggers another evidence-based outer-loop decision, not automatic project termination.
