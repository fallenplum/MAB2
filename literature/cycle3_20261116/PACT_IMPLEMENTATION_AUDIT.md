# PACT implementation boundary audit

Primary repository files are pinned in PACT_CODE_PINS.json; public code was read, not executed or reproduced. No third-party code is redistributed here.

The loader uses supporting-fact titles to put one gold paragraph in each five-document shard (data.py:45-51). Its prompt tells agents this composition (prompts.py:57). That is a gold-informed experimental allocation, distinct from our input-only title-hash partition; it does not directly give agents answer labels. A direct score comparison would confound information allocation.

The implementation alternates two agents and appends outputs after removing complete think-tag blocks (methods/pact.py:100-101). It does not strictly parse and enforce the three public fields. Output tokens are reconstructed from decoded text; ours records generated IDs. Final-answer extraction also differs. The local F1 helper lacks the official special yes/no/noanswer mismatch handling.

For a future matched baseline, preserve the communication concept but explicitly label adaptations: input-only allocation, original model pool, fixed generation limits, official answer metric and raw-token accounting. Report author-code reproduction separately; do not describe a prompt adaptation as an exact replication. No current frozen pilot is altered by this audit.
