# Hidden relay-interface drift

This panel changes one evidence row and independently changes the relay
interface on row 3. A fact-only descendant mask sees only the evidence edit.
The interface-aware worker adds any row with a changed interface fingerprint to
the revalidation set.

| model | policy | decision exact | answer-action exact | relay tokens |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | stale | 0.8672 | 0.8203 | 0.00 |
| Qwen2.5-0.5B | TCR guard | 1.0000 | 1.0000 | 345.79 |
| Qwen2.5-0.5B | fact-only mask, q=.9 | 0.9032 | 0.8710 | 124.81 |
| Qwen2.5-0.5B | interface-aware mask, q=1.0 | 1.0000 | 1.0000 | 291.16 |
| Qwen-14B | stale | 0.8672 | 0.7891 | 0.00 |
| Qwen-14B | TCR guard | 1.0000 | 1.0000 | 322.51 |
| Qwen-14B | fact-only mask, q=.9 | 0.9355 | 1.0000 | 102.68 |
| Qwen-14B | interface-aware mask, q=1.0 | 0.9677 | 1.0000 | 206.13 |
| Llama-3.2-1B | stale | 0.8516 | 0.8672 | 0.00 |
| Llama-3.2-1B | TCR guard | 1.0000 | 1.0000 | 563.84 |
| Llama-3.2-1B | fact-only mask, q=.9 | 1.0000 | 0.9355 | 80.16 |
| Llama-3.2-1B | interface-aware mask, q=1.0 | 1.0000 | 1.0000 | 189.48 |

The cheap fact-only mask is unsafe because the undeclared row changes the
relay output. Adding the interface fingerprint to the changed-row set restores
exact answer action for all three models. At the conservative q=1.0 point, the
interface-aware policy saves 15.8%, 36.1%, and 66.4% relay tokens relative to
TCR guard for Qwen2.5, Qwen-14B, and Llama respectively. Qwen-14B retains a
decision-boundary miss, so the safety claim is stated at the answer-action level.

