# P3 CPU cycle006 amendment

Cycle006 is the confirmatory scale-up of the bounded horizon-2 candidate. It uses a bit-mask posterior implementation to evaluate 500 cases per family-condition on five locked seeds. The implementation uses first-seen terminal label branch order, exact floating-point comparison followed by lexical `(kind, index)` tie-breaking, and excludes terminals already valid in the current cache. The implementation identity is recorded in `cycles/cycle_006/IMPLEMENTATION_ID.json`.

The gate is action exactness 1.0 and at least ten percent cost reduction against masked Pairwise. The result remains a CPU algorithm gate and does not authorize GPU or real-model execution by itself.

