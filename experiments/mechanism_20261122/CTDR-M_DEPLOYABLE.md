# CTDR-M deployable contract

## Claim boundary

CTDR-M transports old communication-credit certificates only when a frozen graph manifest and interface fingerprints show that the affected relay interfaces are known. It may over-revalidate. It must never skip a row whose interface signature is missing or changed.

## Frozen inputs

The deployment record contains old and new graph hashes, the edge edit list, node roles, downstream reachability, terminal depth, relay prompt-template hash, tokenizer and checkpoint hash, upstream message-key set, decoding parameters, and old transcript entries with input signatures and costs. These fields are frozen before answer keys or fresh audit outputs are revealed.

## Mask construction

1. Compute candidate relay descendants of each edited edge from the graph manifest.
2. Compare old and new interface fingerprints for every candidate and for every relay with a changed signature.
3. Take the union of graph descendants and changed signatures.
4. If reachability or any signature is unavailable, mark every relay row as affected and perform a full audit.
5. Only unchanged rows with matching signatures may reuse old observations. Changed rows are revalidated with the new interface before the decision-region audit resumes.

The method uses no fresh decision set, answer key, terminal labels, or hidden row mask. A missing field increases cost through conservative fallback.

## Required controls

- source-only edge deletion with unchanged relay signatures
- conditional-channel drift with unchanged source distribution
- equivalent-content formatting change
- degraded manifest and missing signatures
- warm restart, reweighted audit, no-radius replay, random, periodic, maximum-uncertainty, and cold restart

The principal outcome is finite-audit exact decision-set recovery, followed by incremental logical token cost. Normal service quality and latency require a separate task-level evaluation.

## Falsification gates

The source-only panel must have zero false skips. Conditional-drift controls must have zero false skips. CTDR-M must save at least 20% logical tokens relative to cold restart on at least two independent panels and must not increase action error against maximum uncertainty or source-TV. Failure freezes this route as a safety certificate result and prevents an algorithmic superiority claim.

