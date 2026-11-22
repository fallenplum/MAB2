# Interface-drift stress panel (rows1)

This panel keeps source distributions and terminal prompts fixed. It changes only the new relay interface on graph-descendant row 1 by appending a frozen protocol-v2 suffix. The worker derives changed rows from old/new token fingerprints. The server replay uses the frozen graph manifest for scheduling.

The panel contains 1,152 question-model units from QASC, SciQ, and Commonsense across Qwen-0.5B, Qwen-14B, and Llama-3-8B-Instruct. It is a same-input stress control because it reuses the existing 128-question cohort.

| method | exact-set | action error | post-edit tokens |
|---|---:|---:|---:|
| stale | 0.9418 | 0.0139 | 0.00 |`n| warm_reweighted | 0.9618 | 0.0104 | 22.88 |`n| max_uncertainty | 0.9618 | 0.0104 | 22.88 |`n| random_recheck | 0.9618 | 0.0104 | 37.96 |`n| periodic_recheck | 0.9618 | 0.0104 | 39.86 |`n| source_only_ctdr | 0.9618 | 0.0104 | 22.88 |`n| tcr_guard | 1.0000 | 0.0000 | 170.21 |`n| tcr_token | 1.0000 | 0.0000 | 171.05 |`n| tcr_bound | 0.9158 | 0.0260 | 32.88 |`n| tcr_bound_token | 0.9158 | 0.0260 | 32.88 |`n| tcr_density_gate | 1.0000 | 0.0000 | 170.21 |`n| tcr_density_gate_token | 1.0000 | 0.0000 | 171.05 |`n| cold_restart | 1.0000 | 0.0000 | 949.39 |`n| cold_token | 1.0000 | 0.0000 | 952.71 |`n
TCR guard and token ranking recover the full cold-restart decision set. Warm/reweighted and source-only replay retain the old relay certificate and miss a subset under the changed interface. The worst-case row interval shortcut is retained as a negative control.

The result supports the finite-interface claim that a changed relay signature must enter the revalidation mask. It does not establish end-to-end answer quality or independent-cohort generalization. Costs are post-edit logical query tokens and exclude the already-paid historical transcript.



