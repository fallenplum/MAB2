# Exogenous path screen: execution and reproduction

This package implements a controlled development screen, not an online credit-maintenance method. The scientific contract is `protocol/cycle2_exogenous_paths_20261115/SCIENTIFIC_DESIGN_FROZEN_NO_CI.json`; execution placement and budget are in `EXECUTION_FROZEN_V1.json`. Historical experiments retain their own contracts.

## Input and model boundary

HotpotQA distractor train is pinned to revision `1908d6afbbead072334abe2965f91bd2709910ab`. Source IDs, normalized question groups and content hashes exclude all known earlier exposure. Unknown pretraining and cross-project exposure remain unknown. A seeded 512-question pool is capacity-checked using the actual receiving tokenizers, then the first 96 eligible questions are used. No labels, supporting facts, question type or difficulty enter model prompts or the final selection rule. The source descriptor retains historical metadata-column declarations; the current operator accepts only id, question and context.

Exact sample reproduction needs the private exposure union and selection manifest. They are retained for the project owner but not published. A new researcher can rerun the sampling protocol with their own documented exposure list; that is not the identical sample.

The actual run uses local Qwen2.5-3B-Instruct and Mistral-7B-Instruct-v0.3 weights, pinned file hashes, Python 3.11.9, transformers 4.54.0 and torch 2.9.1+cu128. Verify model identities in the private frozen plan rather than resolving a mutable model name. Local CPU tests used Python 3.11.14; real-tokenizer capacity selection and exact archive replay use the server environment. No model or tokenizer downloads occur implicitly in the runtime.

## Tests

From the package directory:

```text
python run_path_tests.py --receipt /new/private/path/test_receipt.json
```

The entrypoint explicitly supplies the sibling `hotpot_selection_v1/select_hotpot.py` and `hotpot_support` adapters. Plain unittest discovery without these adapters is not the supported full-suite invocation. The synthetic tests cover label rejection, duplicate exposure groups, exact capacity ordering, graph information permissions, cache/full-cost separation, overflow without hidden retries, partial-launch cleanup, incomplete-archive label guards and point statistics. They do not prove model efficacy.

## Collection

Freeze source, model, input and scientific contract hashes before executing:

```text
python path_runtime_v3.py --plan /private/FROZEN_PLAN.json --plan-sha <sha256>
```

Four workers receive fixed disjoint 24-question partitions. They first finish two questions each; the supervisor releases remaining questions only if the predeclared blind cost forecasts fit the hard residence limit. No quality is observed for this decision. The supervisor only terminates its own process groups, retains failed/partial outputs, and records observed residence rather than imputing the cap as usage.

Each complete question has INPUT, MATRIX and COMPLETE records. Worker journals retain exact requests, model events and returned generations. All terminal output strings are scored unchanged except outer whitespace stripping. Intermediate messages are evidence notes; final outputs use a shared short-answer instruction for graph and full-context baselines. No JSON repair, substring answer search or answer-dependent parsing is permitted.

## Closure and replay

After every worker exits, verify every hash in ARCHIVE_MANIFEST and reconcile actual GPU residence into the inherited global budget exactly once. CPU-only replay requires the closed original server archive and actual tokenizers:

```text
python replay_path_archive.py --plan /private/FROZEN_PLAN.json --plan-sha <sha256> --output /new/private/CPU_REPLAY
```

Replay reconstructs the operator using recorded generations and checks prompts, graph edges, seeds, cache identities, token counts and costs. It does not regenerate stochastic tokens. Qualification separately included true repeated generation of an artificial short fixture; that does not establish full-dataset repeatability.

## Label release, score and independent arithmetic

The root reviewer must inspect complete coverage, actual cost closure and replay before issuing hash-bound extraction approval. Then:

```text
python extract_path_labels.py --approval /private/LABEL_APPROVAL.json --approval-sha <sha256> --output /new/private/extraction
python evaluate_paths.py --approval /private/SCORING_APPROVAL.json --approval-sha <sha256> --output /new/private/scoring
python audit_path_points.py --selection /private/SELECTION_MANIFEST.json --private /private/scoring/PRIVATE_ARRAYS.json --summary /private/scoring/SUMMARY.json --output /new/private/AUDIT.json
```

Extraction projects only id and answer from the pinned train parquet files and persists only the 96 selected IDs. Arrow may internally decode other values in row-group buffers; this is not a byte-level isolation guarantee. Official validation/test are not opened. The main evaluator validates every terminal condition before calling its gold loader. The independent arithmetic checker uses separate standard-library loops; it is not independent human peer review or independent model execution.

The primary development filter is fixed: D at least 0.05, both seed means positive, and at least 20 questions with nonzero mean-seed contrasts. No confidence intervals or bootstrap are computed. The exploratory descriptive script cannot rescue a failed primary screen or substitute for disjoint initialization, feedback and future-service experiments.

## Cost and public artifacts

Retain actual residence, physical generated calls/tokens/kernel time, complete logical service costs, qualification failures, test failures, archive/replay evidence and source hashes. Shared exact-request caching saves physical collection work; every logical strategy still pays its full permitted work. Kernel time and process residence are different measurements. Unmeasured monetary, human annotation or engineering costs remain unknown.

Public outputs contain aggregate metrics, figures, source and provenance hashes. Raw prompts, answers, question IDs, per-question matrices, credentials, host details and model weights remain private. Publish only to `fallenplum/MAB2`. A stage report is not a project completion, and a candidate failure does not reset the inherited 160 GPU-hour / 5-candidate limits.
