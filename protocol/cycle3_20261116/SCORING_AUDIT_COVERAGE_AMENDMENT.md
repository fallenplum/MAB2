# Independent scoring audit coverage extension

Recorded before any Formal6 label release. The independent auditor v1 reconstructs main scores and gates but did not check ancillary per-type, per-edge, raw-ranking and projection summaries. Version2 additionally reconstructs those from raw nodes and frozen input metadata. No evaluator, formula, sampling, candidate, threshold or metric changes. Retain v1. Use audit_raw_scoring_v2.py with --questions pointing to the frozen96-row input file. This is a validation-coverage repair, not outcome-based selection.
