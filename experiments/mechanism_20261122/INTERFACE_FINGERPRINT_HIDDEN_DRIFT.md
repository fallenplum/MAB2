# Hidden relay-interface drift

This panel changes one evidence row and independently changes the relay
interface on row 3. A fact-only descendant mask sees only the evidence edit.
The interface-aware worker adds any row with a changed interface fingerprint to
the revalidation set.

| policy | decision exact | answer-action exact | relay tokens |
|---|---:|---:|---:|
| stale | 0.8672 | 0.8203 | 0.00 |
| TCR guard | 1.0000 | 1.0000 | 345.79 |
| fact-only mask, q=.9 | 0.9032 | 0.8710 | 124.81 |
| interface-aware mask, q=.9 | 1.0000 | 1.0000 | 279.45 |
| interface-aware mask, q=1.0 | 1.0000 | 1.0000 | 291.16 |

The cheap fact-only mask is unsafe because the undeclared row changes the
relay output. Adding the interface fingerprint to the changed-row set restores
exact action and decision recovery. At the conservative q=1.0 point, the
interface-aware policy saves 15.8% relay tokens relative to TCR guard.

