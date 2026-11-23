# P0-2 decision-state audit

## Finding

`Audit.decision()` uses two distinct states. `None` means the interval audit is
unfinished. A returned empty set means the audit has proved that no label is
active. Previous scorer paths used `set(audit.decision() or [])`, which collapsed
these states and could score an unfinished audit as a valid empty decision.

## Fix

`conditional_real_score.py` now exposes `decision_set(audit, context=...)`.
It raises on `None` and returns a typed set only for a completed decision. The
M60 module ablation, strong baseline matrix and independent ARC scorer use the
same fail-closed semantics. Bounded guards raise if their stopping loop exits
without a certificate.

## Tests

`test_none_semantics.py` verifies both cases. An unfinished audit raises, while
a completed audit with no active labels returns `set()`. Mechanism-directory
compileall and the theory, general-condition and clean-replay tests remain
passing.

## Acceptance

No result is counted as a legal empty decision after an unfinished stopping
loop. A failure now stops the scorer and requires explicit diagnosis.

The strong-baseline matrix records an unfinished controller with
`completed=0`, `exact_set=0` and an explicit incomplete count. It does not
coerce that state into a legal empty prediction, so incomplete-rate reporting
remains separate from decision-set metrics.
