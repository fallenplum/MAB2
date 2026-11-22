# CTDR-M autonomous research audit, 2026-11-22

## Evidence status

The current candidate is **development-supported** and remains below the project-level ACL Main acceptance boundary. It has exact finite decision-set recovery on 2,304 question–model units from two checkpoints across three task panels and three edit densities, plus a rows1 replication on a third checkpoint (1,152 units). The server used the user-provided address `59.174.162.173:10021` and GPUs 1–5 while other cards were occupied.

## Main result

| density | warm exact | TCR token exact | TCR tokens | cold tokens | saving |
|---|---:|---:|---:|---:|---:|
| one relay row | 96.875% | 100.000% | 132.95 | 1296.10 | 89.7% |
| two relay rows | 88.932% | 100.000% | 224.80 | 1255.30 | 82.1% |
| four relay rows | 77.083% | 100.000% | 419.10 | 1188.98 | 64.7% |

The values average 768 question–model units per density over Qwen-0.5B and Qwen-14B. The TCR token method uses a frozen graph-manifest descendant union to select rows, then a decision-region selector priced by recorded logical token lengths.

The rows1 three-checkpoint replication adds Llama-3-8B-Instruct. TCR token is 100.000% exact at 115.26 tokens over 1,152 units. Warm exact is 97.917% at 23.22 tokens and cold exact is 100.000% at 920.01 tokens.

At each TCR per-unit token budget, random and periodic exact rates are 95.833% and 94.792% for one-row edits, 88.281% and 87.370% for two-row edits, and 76.953% and 76.823% for four-row edits.

## Implementation correction

The first worker implementation evaluated duplicated old/new relay prompts in a shared batch and produced non-identical probabilities for identical text. A deduplicating worker now evaluates each distinct token sequence once and reuses its logits. The pre-fix result is retained as an implementation failure and excluded from the current candidate evidence.

## Falsification controls

A wrong graph manifest replacing the true rows1 relay-1 edit with relay 0 reduces TCR exact recovery to 96.875% while cold restart remains exact. Missing or malformed graph manifests fail closed to all four relay rows. The bounded conditional-row shortcut remains failed with exact rates below 90% on rows1–rows4 and is not part of CTDR-M.

## Open evidence gates

- independent task-level answer-quality or service-cost evaluation with sealed labels
- a graph-manifest conditional-drift panel where source distribution is held fixed and relay interface signatures change
- independent replay from a manifest-generated mask without post-hoc row metadata
- complete task and wall-clock cost ledger

No sealed test was opened. No confidence intervals or bootstrap were run, as required by the project contract.

## Three-checkpoint density replication

Adding Llama-3-8B-Instruct to every density yields 1,152 question–model units per density. Warm exact-set rates are 97.917%, 92.622%, and 84.462% for one, two, and four changed rows. TCR token remains 100.000% exact with 115.26, 200.60, and 376.86 tokens. Cold token costs are 920.01, 892.05, and 832.89 tokens, giving 87.5%, 77.5%, and 54.7% savings. These values are separate from the two-checkpoint table and are the current cross-model replication.

