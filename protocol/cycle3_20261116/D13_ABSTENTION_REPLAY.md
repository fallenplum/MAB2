# D13: fixed abstention baseline screen on exposed development traces

## Scope and hypothesis

Both annotation pilots remain failed. D12 identified exact terminal note-only interventions but did not prove that a deployable rule can recognize harmful notes. D13 tests a simple, non-novel baseline: explicit uncertainty in a generated note may identify occasions where retaining the notes hurts. This is a fixed exposed-development screen; not fresh confirmation, not a formal candidate and not an active revalidation result.

Use all96 questions and both generation seeds separately on each task. No question filtering, response inspection to tune the lexicon, hyperparameter search or new model calls. Freeze source and protocol in Git before benchmark execution. Original tests remain sealed.

## Fixed actions, information and controls

All replay policies first pay the original G1 prefix and can inspect only its existing notes. Actions are retaining all notes or ignoring all notes. D12 must verify the latter terminal request exactly matches recorded SINGLE_Q for all192 observations. Cost is original prefix plus the selected terminal call; prefix cost is never erased. Direct SINGLE_Q, VOTE_Q3 and INDEPENDENT3_Q remain separate strong system controls with their own complete service costs.

Rules: always retain, always ignore, ignore all if ANY note contains an abstention phrase, and ignore all if R's note contains an abstention phrase. Frozen phrases: uncertain; insufficient; cannot determine; can't determine; cannot be determined; not enough; unknown; not provided; not mentioned; does not provide; do not provide; unable to determine. Match case-insensitively on word boundaries after whitespace/apostrophe normalization. No semantic verifier or gold-dependent rule.

For each lexical rule, random controls ignore exactly the same count separately per generation seed, using a SHA256 ordering from question identity, generation seed and one of five policy seeds261116131..261116135. Counts use only observed prefix signals, not labels. Report every random run and the mean; these are correlated replays of the same questions, not five independent datasets.

## Prospective development screening rules

Prefix signal gate: utility advantage >=.005 over always-ignore and the matched-random mean, with strictly positive differences on each generation seed. System gate: utility advantage >=.005 over EACH of direct SINGLE_Q, VOTE_Q3 and INDEPENDENT3_Q, with strictly positive differences on each seed. Utility is unchanged EM-5e-6*(input+output tokens). For further admission the SAME lexical rule must pass both gates on BOTH tasks. Do not pool tasks, choose separate task winners or rescue Formal7/8. Even a pass only justifies fresh confirmation and does not establish novelty.

Failure preserves the result and returns to mechanism analysis. No retrospective phrase additions or threshold edits within D13. A distinct revised mechanism needs a new protocol and evidence. Advantages: zero extra GPU calls, real paired outputs, common available information and full prefix accounting. Risks: coarse lexical cues can discard useful partial facts, minority helpful notes can be lost, prefix overhead may erase benefits, and the screen uses exposed development data. No CI/bootstrap. Report actual accuracy, corrections/destructions, seed breakdowns, service tokens and all failures.
