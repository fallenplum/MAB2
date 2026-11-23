# Pairwise Gap Certificate under hidden interface drift

The hidden-interface worker files add an undeclared relay-interface change to a
graph edit. Fingerprint-aware changed-row resolution is applied before the
pairwise certificate. The held-out residue contains 31 questions per model.

| model | pairwise action | pairwise tokens | interface-aware mask tokens | reduction |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.000 | 263.10 | 291.16 | 9.6% |
| Qwen-14B | 1.000 | 189.55 | 206.13 | 8.0% |
| Llama-3.2-1B | 1.000 | 174.94 | 189.48 | 7.7% |

The fact-only descendant mask is unsafe under this panel. Pairwise replay uses
the interface fingerprints and keeps the answer-action target exact while
removing additional rows when the pairwise certificate is already separated.
These are worker replay costs and do not replace the Qwen-0.5B service panel.

