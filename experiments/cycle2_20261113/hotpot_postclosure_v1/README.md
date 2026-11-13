# Hotpot 闭账后标签提取与冻结评价调用准备

此目录只有实现、人工 CPU 证据和未授权接口。它不授予真实标签访问、评分、科学准入、GPU 或发布权限；本轮没有读取真实题目、输出、答案或 supporting_facts。所有实际动作都要求 root 从独立审查中取得并另行提供 SHA256。把待审批文件自行 hash 再传给程序，不构成 root 授权。

## 两个相互独立的入口

`postclosure.validate_extraction_release` 只验证 metadata bytes。`extract_vault` 先调用该检查，再调用 reader。真实默认 reader 仅接受固定 revision `1908d6afbbead072334abe2965f91bd2709910ab` 的 distractor/train 原描述符对应的两份 Parquet，逐文件核对完整 bytes SHA 和 footer，使用 `columns=['id','answer']`、`id in selected384` 过滤，按原始 ID 产生恰好 384 个字符串答案。不做归一化、补缺、重新选样或重试；重复、缺失、额外 ID 或非字符串答案拒绝。源描述符仍保持原无标签选样列白名单，新的授权只针对此次答案投影。

`score_after_release` 要求第二份 root `APPROVED_FOR_SCORING` 及独立提供的 hash。它核对第一次授权、提取回执和 vault SHA 血缘，原封不动调用冻结 `evaluation_adapter.evaluate_after_release`。只有 adapter 完成原有全部 384 题/4224 终态的 bind，才可触发 `vault_loader` 读取 vault；wrapper 没有自己的评分、统计公式、阈值、终态筛选或容量例外。统计参数使用 adapter 的冻结默认值。

## 提取授权 schema

Root 新建 JSON，schema 为 `hotpot_postclosure_label_extraction_release_v1`，decision 必须精确为 `APPROVED_FOR_LABEL_EXTRACTION_ONLY`，另行提供该文件 SHA。必填字段：

- `postclosure_source_sha256`：本目录 `postclosure.py` 的批准源码 SHA；整个部署包还应由 root 对 manifest 作外部冻结核验，CLI 自身不是授权来源。
- `evidence_sha256`：恰含 `audit`、`template`、`audit_contract`、`selection_manifest`、`source_descriptor` 五份原字节 SHA。
- `audit_source_sha256`：恰含 `archive_audit.py`、`tokenizer_witness.py`、`audit_cli.py` 三个批准版本 SHA，并与 AUDIT 一致。
- `source_identity`：原描述符的 `dataset_id/revision/config/split` 四字段；`expected_questions=384`、`expected_terminal_conditions=4224`。
- `allowed_columns=['id','answer']`、`output_purpose='PRIVATE_LABEL_VAULT'`、`scoring_authorized=false`、`publication_authorized=false`。
- `unknown_preflight_components_acknowledged`：布尔值；如外部审计保留了已证明没有 native generation 的预检未知费用分量，必须由 root 明确置 true。该字段不恢复未知费用，也不更改输入拒绝的冻结评价规则。

AUDIT 必须是 `REVIEW_REQUIRED`，且 `synthetic_tokenizer_witness=false`（不能用数值 0 替代），明确未读标签。原 template 必须仍为 `REVIEW_REQUIRED`、vault pin 为空、科学合同 FROZEN、完整 384/4224 覆盖、runtime failure 为 0、全部尝试和物理驻留闭账。两者通过 SHA 互相绑定并绑定外部审计合同。Template 的 `bindings.scientific_protocol_sha256` 保持原科学合同，不改写为后期资源执行合同；双合同合法性由已绑定的外部审计负责。

这里不会重做真实 witness/账本审计；它信任 root 对精确 AUDIT、审计源码和上述 metadata 的独立复核。人工 fixture 中的 `synthetic_tokenizer_witness=false` 是纯测试声明，不能证明真实审计或充当正式授权。

## 私有 IO 和执行方式（本轮未执行真实数据）

运行需要 Python、已具备的 pyarrow；不自动安装或升级环境。不导入模型 backend，不加载 weights。所有输出目录必须不存在；无自动恢复、覆盖或重试。POSIX 目录/文件请求 0700/0600；Windows 必须由 root 提供已有私有 ACL 的父目录，代码不声称自动建立 Windows ACL。

默认命令 `python -B postclosure_cli.py` 仅返回 `PREFLIGHT_NO_DATA_IO`，甚至不会打开传入的 `--io`。显式提取：

```text
python -B postclosure_cli.py --mode extract --io PRIVATE_EXTRACTION_IO.json --extraction-approval-sha256 ROOT_TRUSTED_SHA
```

IO JSON 恰含 `approval`、`evidence`、`output`、`parquet_paths_by_sha256`。`evidence` 的五个键同上，值是本地文件路径；其他值分别是授权文件、新私有输出目录、两份 source SHA 到本地路径的映射。相对路径以 IO JSON 所在目录解析，不修改原源描述符来适配机器。成功产生 `LABEL_VAULT.json` 和 `EXTRACTION_RECEIPT.json`。stdout 只输出状态、数量和 hash；异常只输出类型。缺回执或写盘中断不是成功，已有目录不得重用。

Root 审查提取回执后，复制原 `REVIEW_REQUIRED_TEMPLATE.json`，只修改 `decision='APPROVED_FOR_SCORING'`、`label_vault_sha256=提取回执中的 SHA`，追加：

```json
{"postclosure": {
  "schema": "hotpot_postclosure_scoring_binding_v1",
  "extraction_approval_sha256": "root已批准的提取授权SHA",
  "extraction_receipt_sha256": "实际提取回执SHA",
  "audit_sha256": "原AUDIT SHA",
  "template_sha256": "原template SHA",
  "postclosure_source_sha256": "批准wrapper SHA",
  "unknown_preflight_components_acknowledged": false
}}
```

最后一项必须与第一次授权一致。其他全部 template 字段须 canonical JSON 字节语义完全相同，避免改科学字段、unit、metadata 或费用声明。Root 独立批准并提供该新评分回执 SHA 后，才能运行：

```text
python -B postclosure_cli.py --mode score --io PRIVATE_SCORING_IO.json --extraction-approval-sha256 FIRST_ROOT_SHA --score-review-sha256 SECOND_ROOT_SHA
```

评分 IO 恰含提取时的 `approval/evidence/output` 和 `score_review/extraction_receipt/label_vault/archive_root/adapter_path`，不含 Parquet 路径。adapter_path 指向本包 `frozen_dependencies/evaluation_adapter.py`；其三个 helper 的相对结构须保留。只产生私有 `PRIVATE_EVALUATION.json` 和 `EVALUATION_RECEIPT.json`，不自动作公开结论或科学/GPU 准入。

## 测试与边界

```text
python -B -m unittest test_postclosure -v
python -B check_preparation.py --output NEW_CPU_OUTPUT_DIRECTORY
```

测试包括未授权/错 SHA 时 reader 从未调用、真实 witness 声明/闭账检查、两份临时人工 Parquet 的列投影与选中 ID、完整人工 JSON→CLI→私有 vault→拒绝覆盖、第二次授权与 lineage、实际冻结 adapter 在不完整归档时不调用标签回调。另有一项明确标记的函数委托 spy，只检查调用顺序和默认统计参数；它不模拟真实归档通过或科学评分。按 root 指示不重建昂贵的完整 384 题模型 mock；冻结 adapter 已有完整人工测试由其原回执证明。

优势：权限门在 reader 之前；授权、源码、audit、选择和 vault 均绑定原字节；不复制评分逻辑，避免另建统计门。

限制/风险：Parquet 全文件 hash 会读取包含 gold 的不透明字节；Arrow 过滤可能先解码未选行的 id/answer 或整个 row group，仅保证返回 Python 的答案集合恰为选中 384，不能声称行级字节隔离。不投影 question/context/supporting_facts，不打开 official validation/test。依赖可信的 source 描述符与 root 提供的外部 pins，不能从自报 JSON 认证操作者身份，不能证明恶意并发修改下的绝对不可变性。若进程/磁盘失败可能留下部分私有文件；必须保留并独立审查，不能当成完整 vault。
