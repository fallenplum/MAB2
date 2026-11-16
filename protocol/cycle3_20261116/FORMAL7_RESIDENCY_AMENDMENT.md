# Formal7 pre-admission residency repair

Engineering attempt1 failed on the artificial long fixture second seed: Mistral eager attention requested4.26GiB with1.74GiB free and8.61GiB unallocated reserve. Preserve the failed attempt and charge0.06139995548557231GPUh. No formal question was inferred or scored.

Runtimev2 places the inactive model on CPU before moving the active model to CUDA, and releases unused allocator cache before/after the move. This prevents the inactive Mistral weights from remaining resident during Qwen execution and clears allocations between model switches. Prompts, RNG, precision, eager attention, weights, operator and fixtures remain identical. All transfer time stays in worker occupancy; record model identity for each placement. No other user's process is stopped.

Rerun the same2 artificial fixtures and engineering seeds26111603/26111604. Require full archive, exact CPU replay, independent evidence/notes/cost audit, and identical outputs for the completed short fixture from attempt1. Retain original outputs if a mismatch is found; do not silently accept a changed experiment. Engineering route total remains<=0.5GPUh and formal budget<=8GPUh.
