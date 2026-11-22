
## Scope

This note audits whether the current conditional-worker artifacts support a task-level answer-quality or deployment-quality claim. It does not alter historical scores or use the sealed answer map for method selection.

## What is available

Each worker record contains source probabilities, old/new conditional relay probabilities, 16 terminal labels, and token lengths. The private `SEALED.json` files contain one benchmark answer index per question. The current scorer reads that map only after mask construction, scheduling, and replay are complete. The public scorer reports finite-audit metrics such as `exact_set`, `action_error`, `false_drop`, `false_keep`, and incremental token cost.

The current evidence therefore supports an audit certificate result. It establishes whether a method reconstructs the post-edit positive-credit label set and how many post-edit logical tokens it spends.

## Why this is not a normal task-quality evaluation

The 16 terminal labels are model outputs conditional on a fixed pair of hypotheses. They are not a service trace generated after a controller applies a keep/remove graph action. The replay scheduler returns a set of answer labels whose credit exceeds the threshold. It does not produce a unique deployed answer, a graph selected for service, latency, or an end-to-end response.

Consequently, `gold in predicted_set` is a coverage/action diagnostic. It must not be described as answer accuracy. The current full and TCR methods can have identical task outputs because they audit the same post-edit graph while differing only in how much evidence they purchase.

The existing sealed labels were already opened by the scorer for reporting on the development panels. They cannot serve as an independent confirmation set for a main-paper service-quality claim.

## Valid path to a service-quality panel

Freeze a new question cohort and graph-edit manifest before reading labels. For each question and checkpoint, run the same old transcript and new-graph terminal service under each controller. Convert the controller's selected graph into one deterministic answer, for example the highest-credit label under the selected graph, and record exact-match against a newly sealed benchmark answer. Use the same question cohort, model checkpoints, decoding, graph edit, and token accounting for stale, warm/reweighted, random, periodic, uncertainty, TCR, and cold restart. Report per-task accuracy and incremental token cost, plus the paired change in answer and the count of corrected and damaged decisions.

The controller should choose its graph from the audit state only. Gold labels may be read after all controller outputs and schedules are sealed. The public release should contain aggregate and task-level summaries without raw questions, answer arrays, or private weights.

## Current decision

No independent task-level service-quality evidence is present in this checkout. The correct claim boundary remains finite-audit exact-set recovery and post-edit logical-token savings. A new sealed task-quality panel is required before claiming improved end-to-end answer quality, lower wall-clock latency, or deployment superiority.


