# Phase 0/2 real-task and model availability audit

## Local evidence

- HotPotQA raw JSON is present under `项目材料/早期数据集/HotPotQA`. Train has 90,447 records. Fullwiki and distractor dev each have 7,405 records. Sample records contain question, answer, context, supporting facts, type and level. This is a viable long-context evidence task candidate.
- The local 2WikiMultiHopQA directory contains paper, evaluator and an older baseline implementation, but no raw QA JSON/JSONL data file. It cannot be used for a new independent run without obtaining the dataset.
- The local MuSiQue directory contains the original code, evaluator and experiment configurations but no raw MuSiQue-Ans JSONL. The existing `routing_idea2` audit reports a CC BY 4.0 MuSiQue-Ans archive with 19,938 train and 2,417 dev records, but the archive itself is absent from this checkout. The audit is inventory evidence, not fresh task data.
- QASC, Commonsense and SciQ data and cached worker traces are present in `MAB2_recovered`, but they are the prior four-relay MCQA family and do not satisfy the P3 independent-complex-task requirement.

## Model evidence

No local Qwen/Llama/Mistral checkpoint or Hugging Face cache was found under the current workspace. Server-side model and asset availability must be checked in the authorized SSH environment before any GPU launch. Qwen-0.5B was available in the prior P2 server run, but it is a development smoke model and does not satisfy the P3 two-checkpoint target by itself.

## Minimum viable next audit

1. Check the server for MuSiQue-Ans raw data and at least two locally legal checkpoints.
2. If MuSiQue is available, freeze a document-version or supporting-paragraph update using a new question-group manifest, preserving the original evidence and decomposition fields.
3. Build a real parent-input fingerprint for each evidence selector, verifier and answer node. A changed parent output must change every child input digest that consumes it.
4. Run only a small CPU/service smoke on the development split before selecting a second task or checkpoint.

## Gate status

`P3_REAL_TASK_AUDIT = PENDING_SERVER_ASSET_CHECK`. No GPU job was started by this audit.

