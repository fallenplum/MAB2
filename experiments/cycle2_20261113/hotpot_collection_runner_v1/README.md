# Hotpot 384 题采集编排 CPU staging

**状态：可审源码，不冻结，不授予科学准入或 GPU 使用权限。** 本目录没有正式问题、答案或模型权重。所有本地测试仅使用人工输入及假 backend。原矩阵、durable helper、独立12、图执行器和历史结果均未修改。

实现前设计保留在 DESIGN.md；本文描述目前实际接口。优势是逐请求原文证据、严格前缀放行和完整驻留计费各有独立校验；限制是 8 题耗时外推不构成尾部上界，真实 Linux guard/磁盘与模型运行仍需外部资格和独立复核。CPU 通过对确定性编排逻辑的置信度为高；对未来 384 题能否在预算内完成没有实测结论。

## 可移植 CPU 验证

需要 Python 3.11、NumPy 和本目录 Python 文件；无需 torch/transformers、权重或 benchmark。额外输入只有已公开算子几何元数据 GEOMETRY.json，以及共享 durable_backend.py 源码。输出目录必须不存在：

```text
python -B run_cpu_tests.py --output cpu_replay_new --geometry PATH_TO_GEOMETRY.json --durable-helper PATH_TO_durable_backend.py
```

Windows 初次沙箱测试 7 PASS / 7 ERROR 的原因保留在 CPU_FAILURE_001.json：6 项为 os.link 被沙箱拒绝，1 项为测试桩未提供 Linux SIGKILL。生产原子发布语义未降低。后续在授权 Cstage 文件访问下执行。Windows 会明确跳过 2 项真实 Linux 进程测试；Linux 应执行它们，不得将 skip 算成 PASS。进程测试只启动短时 Python sleep 子进程，并只杀本次自有进程组；其中检查 parent-death 信号配置与 exec 后独立期限，不声称模拟了全部父进程崩溃场景。

## 准备与运行边界

plan_tools.py 的 prepare_plan 只消费已固定的 384 题选择产物，不采样真实来源。九个必填 CLI 参数是 selection、selection-receipt、questions、lengths、protocol、bindings、budget、output、attempt-id。输出 PLAN.json 状态始终 PREPARED_NOT_FROZEN_NOT_ADMITTED；此文件不能直接运行。正式冻结、资源分配和独立 admission 由 root 另行完成，须输出新文件，禁止修改 prepare 产物后冒称原 hash。

parent.py / worker.py 要求 Linux，且计划为 FROZEN_COLLECTION_PLAN；独立 admission 要求 APPROVED_FOR_FROZEN_COLLECTION，绑定 plan SHA、budget SHA、新 attempt、R1 闭合与资格来源验收。未授 cap、占位 pin、改源码、改长度/选择/上下文、重复 attempt 均在模型加载前拒绝。计划中 scientific protocol 的实际 candidate cap 和 per-worker caps 必须与分配预算相等。

预算字段均为显式秒或字节：candidate_cap_seconds、global_available_seconds_for_attempt、worker_caps_seconds、workers[{gpu}]、disk_min_free_bytes、release_guard_seconds、allocation_authorized、prior_candidate_transactions。worker 数为 2–4，GPU 不重复；各 cap 总和不能超过 global 剩余，prior candidate 加各 cap 不得超过 candidate 总额。每笔 prior transaction 有唯一 transaction_id、seconds、already_in_global_ledger=true；本轮 long qualification 记 candidate 一次，已经从 global 剩余中扣除的金额不再扣第二次。代码没有默认 48 GPUh。

## 输入和身份字段

- selection 为原 selector 的 SELECTION_MANIFEST.json；questions 为原五字段 JSONL，保留原顺序及完整上下文。逐行 input_sha256 使用排序 JSON 加末尾换行；collector payload_sha256 使用图算子排序 JSON、不加换行，二者明确分开。
- LENGTHS receipt：status=TOKENIZER_LENGTHS_COMPLETE_NO_LABELS，labels_read=false，server_runtime_equivalence=true，questions_jsonl_sha256、selection_manifest_sha256、collector_source_sha256；payload_hash_algorithm=graph_credit_canonical_utf8_sha256_no_newline；双方 tokenizer_identity_sha256 与 bindings 一致；双方 model_identity_sha256 等于各已验证模型 identity 的图式 canonical SHA。rows 为 id、payload_sha256、qwen_tokens、mistral_tokens。长度必须由服务器对应 tokenizer 对完整 payload 生成；本地 2000 样本预检不能替代。该 receipt 生产工具不在本包，真实题目未读取。
- bindings：collector{path,sha256}、geometry{path,sha256}、source_files[{path,sha256}]、python_import_paths、modules{backend,matrix,durable} 各 path/SHA；models{qwen,mistral} 各 path、完整 identity、tokenizer_identity_sha256；graph_config、expected_matrix_protocol_hash、engineering_admission{path,sha256}。部署应使用服务器实际路径，不能上传本地 E 盘路径当可执行 plan。
- JSON graph_config 仅与默认 dataclass 做 canonical 等价核验；实例始终 GraphConfig(rounds=3)，不会把 JSON list roles 传入而改变 tuple 比较语义。双方模型 identity 完整匹配；HFBackend 的 physical_log=None、矩阵 route_log=None，逐请求原文和 native events 统一由共享 helper 保存。
- engineering acceptance 是独立 root 回执，decision 精确为 PASS_SYNTHETIC_MATRIX_ENGINEERING_FOR_COLLECTION；还需 matrix_protocol_hash、model_identity_sha256、module_sha256={matrix,durable}、candidate_transaction_id、candidate_closed_seconds。evidence 的 qualification、identity、residence、plan、source_manifest、root_closed_review 各为 {path,sha256}；path 必须相对 acceptance 所在 bundle，不可绝对路径或逃逸。每项实际 bytes 核 SHA。本轮资格 transaction/closed seconds 必须在 prior candidate 中恰一次。原资格 REVIEW_REQUIRED 自身不能授准入。

## 放行、持久化与费用

长度 8 层各 48 题，PCG64(261113104) 每层固定 1 题；8 题共 88 条件付费完成后才能预测。余下 376 题保持原选中顺序，各 worker 保持自己被固定分配的子序列；并行完成顺序不承诺全局串行顺序。

预测按 architecture/model/node 使用对应长度层观测与该 worker 自己前缀观测中的较大耗时，另加非负 unit wall 残差，乘至少 1.25。所有已用模型加载、驻留等待与 release 延迟通过 parent 当前 monotonic 计入。每 worker、global sum、candidate 加 prior 均须 fit。缺费用/耗时或前缀 INPUT_REJECTED 不能支持本预测，停止为 INCOMPLETE_RESOURCE_FORECAST_UNSUPPORTED；这是资源外推不支持，不能当作质量结论。成功几何 87 calls 不是未来 hard bound：overflow 可失去复用，真实 hard caps 继续有效。

FORECAST_FROZEN 与 RELEASE_REMAINDER 先写同目录独有临时文件并 fsync，再 hard-link 原子不可覆盖发布。只删除发布后自己的冗余临时链接；磁盘 guard 不删除数据腾空间。放行 hash 绑定 forecast；发布前重验当前余量。父进程从每 worker Popen 前到退出观测累计驻留，求和为 GPU 占用口径，包括加载/等待，不能用并行 wall 或模型 elapsed 代替，也不以 cap 截断实际超时。

每请求由共享 helper 在调用前 fsync START，返回前 fsync 完整 messages/Generation/native events。START 无 terminal 保持 pending 未知；未返回原文不能伪造。RETURNED_JOURNAL 失败时，worker 沿 matrix 异常 cause 提取实际 RAM Generation 与 native events，分别 best-effort 保存，部分文件失败不跳过其余证据。持续磁盘坏或 SIGKILL 仍可能丢失尚未持久化的原文，不能保证恢复。残留 pending 文件保留并进入最终私有快照。

## 输出接口

每题路径为 worker_W/unit_III/{INPUT.json,MATRIX.json,UNIT_RECEIPT.json}；III 是原 selection 0..383 索引，不是完成顺序。INPUT 绑定完整 unit、payload 和 plan SHA；MATRIX 为原 MatrixResult.to_dict；receipt 绑定两文件 hash、协议身份、11 terminal_keys、已知/未知费用、overflow 和纯费用 profile。失败另留 PARTIAL_MATRIX、DURABLE_ERROR_EVIDENCE、FAILED；没有自动 retry/resume。

终态必须覆盖 384×11=4224、唯一 seed/name/mask 与 payload fingerprint。INPUT_REJECTED 保留、不独立评分。parent 的 RAW_OUTPUT_MANIFEST 只 hash 私有原文 bytes，不解读答案；PARENT_RESIDENCE 独立记 attempt 驻留与未知关闭状态；ATTEMPT_RESULT 绑定两者。即使收集完成，score_release_authorized 仍为 false，须独立完整源/账/评分验收。未知物理费用和未闭合驻留不填零。外部独立 evaluator 不属于本 runner。

仅 counts/status/cost 出现在进度；无 raw 文本、逐题效果、正确性或信用。调用链不访问 gold 或 label vault。源码中出现原文仅用于私有持久化，不将生成文本视为题目的 gold。
