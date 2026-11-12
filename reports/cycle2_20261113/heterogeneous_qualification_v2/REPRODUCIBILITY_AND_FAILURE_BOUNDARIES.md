# Qualified-source byte reconstruction and remaining failure semantics

Independent agent review confirmed the 64 successful calls, node-route identities, changed/fresh replay and sender/receiver tokenizer accounting. Its raw receipts and limitations are preserved in external_agent_review.

F1 was a reproducibility gap: qualify_routed_executor.py and PROTOCOL.json were deployed from CRLF working files, whereas the frozen b7434719 Git blob used LF. The seven deployed hashes match the before-run private source manifest and SSH upload receipts. Only those two line endings differ; the execution protocol hash remains independently reproducible. No old commit, protocol, output or run is edited. QUALIFIED_SOURCE_BYTE_MAP.json and materialize_qualified_source_bytes.py reconstruct all seven exact deployed files from the immutable Git commit into a fresh directory, including the two explicit byte transforms. A plain Git checkout alone does not satisfy the original raw source pins.

F2 is a future-run accounting boundary, not a failure observed among these 64 calls: input/template/position-cap rejection may occur before HFBackend assigns a call ID and records a physical-call row. RoutedBackend records a failed attempt and the supervisor retains full occupancy, but a failed-route/physical-call bijection must not be assumed. Missing token or elapsed fields remain unknown, not zero. Real-task preflight should reject oversized inputs before generation and preserve failed-attempt stage information. This qualification did not exercise such a real rejection.

Reconstruct source bytes with a local clone containing b7434719:

    python materialize_qualified_source_bytes.py --repository <clone> --manifest QUALIFIED_SOURCE_BYTE_MAP.json --output <fresh_directory>

Model files, runtime paths and the private plan are deliberately separate. Their hashes and resource controls must be supplied for a new run; this script does not load models or run inference.
