# Exact certificate scope and transport counterexamples

Let a terminal answer be generated from the complete serialized prompt, model identity, generation settings and node RNG. Equal request signatures under the frozen deterministic backend imply equal recorded outputs for that coupled execution. Identical canonical evidence sets suffice only when the question, final instruction, model and RNG also match. This is an execution equivalence certificate, not an originality claim.

For edge credit c=reward(G)-reward(G-minus-edge), carrying an old nonzero credit without new reward evaluation requires validity of both the base and ablated execution signatures. Matching only the base trace is insufficient: the ablated continuation can change while the full graph answer stays fixed. A terminal direct-edge deletion can be inspected from already available selected evidence; an upstream deletion can require new relay generation, whose cost must be charged.

A same-seed, same-question zero-credit certificate is not a statement that the edge has zero average credit on future questions or generation seeds. Source selections can change across seeds, and future question text changes the terminal prompt. Do not reuse an exact local certificate as a distribution-shift guarantee or a global edge-removal proof.

The current pilot reports the fraction of realized paired deletions that have an exact certificate. It does not estimate deployment feedback savings from that fraction. Any active-policy comparison must make those same free labels and exact cache rules available to every baseline; otherwise an apparent acquisition advantage may only be an information-budget mismatch.

Counterexamples to reject: (1) same full answer but different ablated answer; (2) equal document IDs but a different question or tokenizer/model identity; (3) certificate at seed1 but different source selection at seed2; (4) a deleted source whose selection was already generated and then incorrectly removed from the cost ledger. These are design constraints, not empirical results from the unscored pilot.
