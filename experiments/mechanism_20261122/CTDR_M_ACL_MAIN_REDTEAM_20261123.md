# CTDR-M ACL Main red-team review

Date: 2026-11-23
Branch under review: `ctdr-audit-20261122`

## Verdict

The current record supports a development mechanism result. It does not yet support an ACL Main claim. The strongest supported statement is that a manifest-driven conditional revalidation policy can recover a finite post-edit credit decision set with fewer logical tokens than a cold restart on the evaluated model-generated audit interfaces. The evidence does not establish end-to-end answer quality, wall-clock service benefit, or broad generalization to unseen question cohorts.

Confidence in this verdict is high because the limitations follow from the released scorer, input cohort lineage, and current cost fields rather than from an aggregate metric choice.

## What is supported

| Claim | Evidence | Review status |
|---|---|---|
| Exact finite audit recovery | TCR token exact-set rate is 100% on 1,152 units per density for three checkpoints, three tasks, and three edit densities. | Supported for the evaluated interface |
| Lower post-edit logical-token count than cold restart | 115.26, 200.60, and 376.86 tokens for TCR token versus 920.01, 892.05, and 832.89 for cold token in the three-checkpoint density replay. | Supported for logical-token accounting |
| Graph manifest is a necessary mechanism input | The rows1 wrong-manifest control falls to 96.875% exact recovery, while the correct manifest is exact. | Supported in one negative-control panel |
| Missing metadata is handled conservatively | Missing or malformed manifests fall back to all relay rows. | Supported as a software behavior |
| End-to-end task accuracy or deployment quality improves | No controller-to-answer service trace exists. | Unsupported |
| Wall-clock or GPU cost improves | No closed wall-clock ledger is attached to the public results. | Unsupported |
| CTDR radius or generic decision-region stopping is independently novel | The current radius condition does not stop before the ordinary audit decision in the released loop, and decision-region active testing is established prior work. | Unsupported |

## Main blockers for ACL Main

### 1. The effective sample count is overstated if treated as independent

The 2,304 two-checkpoint units and 1,152 three-checkpoint rows1 units reuse the same 128-question cohort within each task across edit densities. Model checkpoints provide cross-model replication, but the input cohort is not independent across panels. The paper should report question keys, checkpoint, task, and edit density as repeated factors and avoid describing these panels as independent datasets.

Required correction:

- Report per-task and per-checkpoint tables with the number of unique question keys.
- Add a disjoint question cohort frozen before scoring.
- Keep the current cohort as development evidence and do not tune a new policy from its sealed labels.

### 2. The current score is a finite audit diagnostic, not service quality

`exact_set` compares the replayed positive-credit set with a full-restart reference. `gold in predicted_set` is an action diagnostic. The worker records conditional probabilities and terminal labels, but no controller applies a selected graph to produce a unique deployed answer. Consequently, current results cannot support answer accuracy, user utility, latency, or quality-preserving savings.

Required correction:

- Freeze a new question cohort and graph manifest.
- Make each controller emit a graph/action and then run the same terminal service.
- Read new sealed answer labels only after all controller outputs are sealed.
- Report exact match and paired corrected/damaged answers together with incremental tokens and wall-clock time.

### 3. Baseline information and budget are not fully aligned

The current scorer maps `max_uncertainty` to the warm replay path, so it is not an independent uncertainty baseline in the current public implementation. Random and periodic controls are capped by the TCR token budget and may stop before spending the cap when the next legal action is too expensive. Historical maximum-uncertainty results use a different protocol and must not be pooled with the new panels.

Required correction:

- Implement maximum decision uncertainty as an independent policy over the same legal actions and old transcript.
- Use a predeclared budget rule that either permits fractional budget accounting or reports the unused remainder for every unit.
- Recompute all controllers from the same old transcript, graph manifest, token prices, and budget.
- Publish a per-unit table of budget, spent tokens, unused tokens, actions, and exact-set result.

### 4. The independent algorithm contribution is not isolated

Warm restart, reweighted audit, and source-only CTDR can coincide. The released radius condition is stricter than ordinary audit decision and cannot be credited with an earlier stop under the current loop. Generic decision-region stopping, total-variation bounds, and cost-aware test selection have established precedents.

The defensible contribution is the graph-edit descendant mask plus relay-interface fingerprint and fail-closed revalidation policy. The paper must show that this combination adds value over each component.

Required correction:

- Compare warm/reweighted, no-radius replay, graph mask only, interface-signature guard only, mask plus guard, and mask plus token-ranked revalidation.
- Include correct manifest, wrong manifest, missing manifest, stale signature, and source-distribution shift controls.
- State the theorem assumptions explicitly and restrict the claim to those assumptions.

### 5. Conditional-channel drift coverage is still narrow

The current rows1, rows2, and rows4 panels change relay evidence, but they do not by themselves establish that a production interface signature changed while source distribution stayed fixed. A single fixed protocol suffix is also an artificial stress case.

Required correction:

- Freeze at least three drift classes before reading labels. Use semantic evidence replacement, formatting/template change, and upstream-message-key or prompt-order change.
- Keep source probabilities and terminal prompt text fixed where the protocol requires it.
- Use an unseen question cohort and at least two checkpoints.
- Require zero false skips for the graph-mask and signature guard, then report the extra revalidation cost when the manifest is degraded.

### 6. Cost accounting is incomplete for a systems paper

Logical token counts omit model loading, batching, GPU time, queueing, and wall-clock latency. The current outputs also mix post-edit incremental costs with historical transcript cost unless the ledger explicitly separates them.

Required correction:

- Record edit-before cost, post-edit public-source cost, relay cost, terminal cost, cache hits, model loading, wall-clock time, GPU-hours, and peak memory.
- Use the same batching and concurrency policy for all controllers.
- Report cost per question and per completed service answer.
- Preserve failed jobs, retries, and environment versions.

## Required acceptance gates

A Main-level submission should not be drafted as a positive-method paper until all gates below are met on data frozen before scoring.

1. **Independent cohort gate**. At least one disjoint question cohort with two or more tasks and two or more checkpoints.
2. **Service gate**. The controller emits a unique graph/action and the resulting terminal answer is scored against newly sealed labels.
3. **Fair baseline gate**. Warm/reweighted, independent max-uncertainty, random, periodic, mask-only, signature-only, full CTDR, and cold restart share transcript access, prices, and budget.
4. **Drift gate**. Semantic, formatting, and metadata/signature drift panels with source distribution held fixed where specified. Zero false skips is required for the safety claim.
5. **Attribution gate**. Full CTDR must beat the strongest fair baseline on paired service quality at equal or lower total cost. A lower audit token count alone is insufficient.
6. **Cost ledger gate**. Every result has closed token, time, GPU, retry, and failure accounting.
7. **Reproducibility gate**. Public code, manifest schema, model revisions, code hashes, environment lockfile, and replay commands reproduce all reported aggregates without private raw inputs.

## Priority experiment schedule

### P0, no new inference

- Correct the max-uncertainty alias.
- Add the mask-only and signature-only ablations.
- Run metadata-corruption replay and wrong-manifest controls at all three densities.
- Add automated provenance checks that fail if gold labels are read before controller outputs are sealed.

### P1, low-cost model replay

- Run the interface drift panel on the existing three tasks and three checkpoints, labelled as a same-input stress control.
- Capture full per-unit token and action ledgers.
- Verify that row masks are generated from graph manifests and signatures rather than row-level changed-row metadata.

### P2, independent confirmation

- Prepare a new disjoint question cohort and sealed answers.
- Run service-level controllers with the full baseline matrix.
- Add at least one task family outside the current science-MCQA mix if the project scope allows it.
- Freeze the analysis script and publish the exact acceptance decision.

## Claims that should be removed or narrowed now

- Replace “answer accuracy” with “finite audit decision-set recovery” in current result sections.
- Replace “independent panels” with “disjoint input panels” only when the question keys are actually disjoint.
- Do not claim CTDR radius, TV bounds, or decision-region scheduling as stand-alone innovations.
- Do not pool historical maximum-uncertainty scores with the current scorer protocol.
- Do not report the 80–90% savings as end-to-end system savings until the complete ledger is closed.

## Reproducibility checklist

- Branch and commit SHA recorded for every result.
- SHA-256 for worker, scorer, manifest, model revision, and input files.
- Environment package lock and CUDA/PyTorch/Transformers versions.
- Per-job GPU assignment, start/end time, retries, and exit status.
- Per-task, per-checkpoint, per-density, and per-controller aggregates.
- Public artifacts exclude raw questions, answer maps, private matrices, and model weights.

## Bottom line

The candidate is credible as a mechanism study only after the independent drift and service-quality gates are closed. The present evidence is enough to keep the route active and to justify targeted revalidation as a hypothesis. It is insufficient to label the work ACL Main-ready.

