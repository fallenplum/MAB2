# Heterogeneous executor engineering qualification

The fixed Qwen2.5-3B / Mistral-7B node route completed 64 physical calls on two authored synthetic inputs at one seed. An independent arithmetic implementation checked 260 semantic replay field equalities, all node integrity digests, every route-to-physical-call join, eight graph-cost records, and reuse of unchanged nodes. The replay passed; no malformed transport outputs occurred among these 64 calls.

Actual load-to-exit occupancy was 0.11611894417664088 GPU hours, including model loading and host/storage waits. CUDA event time was separately 84.01500048828125 seconds. The remaining 0.3838810558233591 GPU hours of the 0.5-hour engineering reservation were released. This uses zero formal candidate rounds.

The engineering check supports running this frozen executor in a separately admitted future study. It does not establish task accuracy, general format reliability, novelty, or a scientific candidate pass. No benchmark questions or final test were used. The earlier missing-tokenizer incident and before-run model-input amendment remain recorded.

The independent reviewer initially compared a whole integrity digest across fresh calls; that digest intentionally includes unique call identifiers and physical timing. The reviewer was corrected to verify every digest against its own payload and separately compare semantic outputs. No production output or protocol was altered. The review was a separate implementation by the root agent, not an external human review.
