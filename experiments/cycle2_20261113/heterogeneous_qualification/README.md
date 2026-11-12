# Heterogeneous engineering qualification

This isolated engineering extension dispatches a frozen model per worker and
aggregator. It does not modify any historical or running R1 executor. It adds no
learned router and makes no accuracy, novelty or scientific qualification claim.

The unchanged base executor assembles prompts. Sender tokenizers normalize their
own wire messages; recipient tokenizers count incoming evidence and enforce the
actual chat-template input cap. Model identities, fixed routes and adapter code
enter the protocol hash. Any route/model change invalidates the complete prior
cache conservatively. Shared routed backends reject concurrent execution.

Run `test_routed_executor.py` with this directory and the repository `src` on
PYTHONPATH. The tests use synthetic tokenizers, not weights. Real qualification
requires a private pinned deployment plan and an already admitted global budget
reservation. `qualify_routed_executor.py --plan <private-plan.json>` launches its
own bounded child and never terminates unrelated processes. Both weight sets
remain on one GPU; wall occupancy is charged once for that card, not once per
model. All failed calls and load time count. No benchmark questions are read.

The 0.5 GPU-hour ceiling includes a watchdog at 1770 seconds plus termination
grace. Retain actual closed occupancy even if an operating-system incident causes
an overrun; never clip accounting. Such an incident requires review before retry.
The qualification tests 64 physical calls and exact replay on two synthetic tasks.
Format failures are reported, not hidden or recast as successful scientific work.
