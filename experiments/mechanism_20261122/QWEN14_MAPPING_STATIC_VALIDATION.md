# Qwen-14B service/worker mapping static validation

This artifact validates the corrected mapping diagnostic without loading model weights.

- Input: `arcc_independent_rows2_INPUT.json`, all 128 rows.
- `service_forward_benchmark.build_prompts` and `diagnose_q14_mapping_fixed.prompts` generate identical 23 prompt strings for every row.
- Worker layout is source rows 0:3, old relay rows 3:7, new relay rows 7:11, and terminal rows 11:27.
- Service layout is source rows 0:3, new relay rows 3:7, and terminal rows 7:23.
- Correct correspondence is service source 0:3 to worker 0:3, service new relay 3:7 to worker 7:11, and service terminal 7:23 to worker 11:27.

The validator does not establish numerical logit equivalence. That requires a model-forward run on the server or a local copy of the Qwen-14B weights.

