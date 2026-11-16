# Second dataset preparation

Primary paper: https://aclanthology.org/2020.coling-main.580/ (Ho et al., COLING2020). Author repository: https://github.com/Alab-NII/2wikimultihop . Author mirror: https://huggingface.co/datasets/xanhho/2WikiMultihopQA . Train revision612bc5039a457880d9e7d84c3b0a4cf154b70e4f, SHA256a0f464f1604edf40ad990b605c61c03d56911c106d09ad32becf6afcf3201d52. No arbitrary dataset loader execution. Only train.parquet downloaded; dev/test absent.

Input-only audit covers167454 rows,167453 normalized question groups; every row has10 documents. Only `_id`, `question`, `context` were projected. Zero exact normalized question overlap with known MAB2 exposures, but cross-project exposure remains unknown and semantic paraphrases were not checked. This is inventory readiness, not independent experimental confirmation.

The mirror schema has canonical answer but no answer_id/evidences_id or alias table. The author repository's2021 alias-aware evaluation cannot be claimed from this schema alone. Before formal use, freeze either explicitly named canonical-answer scoring or obtain and align the updated IDs/alias source. No label was inspected or evaluated during preparation. Keep supporting facts and evidence labels outside policy inputs.
