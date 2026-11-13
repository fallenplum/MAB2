# Hotpot私有选题准备工具

状态：代码与人工测试完成，**没有选择真实题、读取真实题/标签或启动GPU**。默认入口是preflight，不解码题行、不创建选题产物。不下载数据、不调度模型、不建立label vault。

已在现有Python环境的NumPy 1.26.4、PyArrow 23.0.0下验证，无新增安装。正式使用须固定运行环境与Unicode数据库版本；回执记录版本。纯接口、文件路径均参数化，无机器特定数据路径。

## 两种入口

```text
python -B select_hotpot.py --source <source.json> --source-sha256 <SHA256>
  --expected-revision <audited40hexRevision>
  --protocol <frozen_protocol.json> --protocol-sha256 <SHA256>
  --exposures <private_exposure_manifest.json> --exposures-sha256 <SHA256>
```

以上为默认`--mode preflight`，只校验JSON pins、revision对照、两份Parquet字节SHA、footer行数和schema。示例换行仅供阅读，实际命令应按shell规则写成一条或使用参数数组。整文件SHA读取的是不解码的字节，不能误称根本没有接触含标签的文件；**没有投影/解码answer或supporting_facts列**。

真实选择必须额外指定：

```text
--mode select --admission <private_selection_admission.json>
--admission-sha256 <SHA256> --output-directory <new_private_directory>
```

真实选择要求source与scientific protocol均FROZEN，独立准入回执明确批准PRIVATE_NO_LABELS选择，并绑定source/protocol/exposure/selector代码SHA及精确输出目录。该准入不授予GPU；协议中的GPU专项pin/费用还需其独立准入。三份描述/回执只是已可信上游冻结流程的绑定检查，不能凭自制JSON证明人类批准或官方来源。

源文件必须恰为两份官方distractor train Parquet。revision必须是外部审计给出的40位不可变commit，且同时匹配source descriptor和命令行独立pin。**Parquet字节无法反推出Hugging Face revision**；工具验证文档链与本地字节，不重新上网证明来源。`source_audit_sha256`关联主代理已完成的来源审计，不能填虚构值。

## 输入schema

`SCHEMA_EXAMPLES.json`提供不具备执行权限的合并示意，不能直接拿它作生产descriptor。

- source descriptor严格字段：`schema_version/status/dataset_id/revision/config/split/source_audit_sha256/read_columns/parquet_files`。`read_columns`必须恰为`[id,question,context,type,level]`，文件字段为`path/sha256/num_rows`，相对path按descriptor目录解析。
- 官方源schema必须是这五列加answer、supporting_facts；未知新增列、少列、hash/行数不符均拒绝。真实迭代只请求五个允许列；行/嵌套context出现额外字段也拒绝。
- context必须保留原`title`列表及对应`sentences`嵌套列表的文字、空白与顺序，不能按supporting_facts选段。title重复或数组长度错配拒绝。
- ID是官方raw ID，不带collector的`hotpotqa_`前缀。重复ID即使内容相同也拒绝；不通过覆盖字典隐藏冲突。
- protocol接受已经生成的384题候选合同schema，但真实选择时status必须FROZEN，数量、四边、seed和任务必须与代码常量一致。

## 暴露、去重与随机选择

外部暴露表必须汇集所有已记录工程/容量探针（包括capacity2000）、旧实验与明确文献暴露；优先使用可追溯raw IDs。它只含ID或hash，不含答案/题文。工具不扫描其他项目，也不能独立证明外部表已经完整；未知跨项目暴露始终UNKNOWN，不给预训练无污染保证。

question分组严格执行`NFKC → Unicode空白合并 → casefold`，组hash为规范化文本UTF-8的SHA256，不带末尾换行。它不删除标点，也不会把标点前的空白删除。旧`lower+空白`hash不得标成新normalizer；兼容性不符即拒绝，外部应使用raw IDs或重建正确hash。

同question组无论context是否不同，仅保留UTF-8字节序最小ID；任一成员ID、question-group hash或content hash暴露，整组排除。组内非代表ID暴露同样有效。content hash定义为`{question: 原文, context: 原结构}`的canonical JSON（UTF-8、sort_keys、紧凑分隔符、禁止NaN、末尾一个LF）SHA256。完整输入hash则对允许的五字段原始行按同规则计算。精确content重复已蕴含相同question组，因此不需要另造跨组关系。

合格代表按UTF-8 ID升序，PCG64(261113101)置换后取前384；不足384则失败，不换来源或减少N。PCG64(261113102)置换`repeat([1,5,7,9],96)`分配边。生成seed保持26111301/26111302。源行/文件迭代顺序改变不会改变选题结果；环境版本仍必须冻结。

## 私有输出与失败

仅在全部校验和选题完成后创建全新输出目录，禁止覆盖/静默resume。源字节及输入pins在扫描后重检，改变则不输出。写入中断可能留下没有完整回执的目录，应保留为失败尝试，不能当已完成样本集或盲目覆盖。

1. `questions.jsonl`：仅五个allowlisted字段的384行，raw ID与原context不变。后续collector可用已冻结`hotpot_support.collector_payload`转换；不能把标签拼入此文件。
2. `SELECTION_MANIFEST.json`：每题只有ID、question/input/content hash、边和两个生成seed；另含整体pins与seed说明。它仍是私有材料，不进公开仓库。
3. `SELECTION_RECEIPT.json`：来源pins、版本、输出hash、数量与排除支持数，标签解码及GPU准入均为false。独立评分准备阶段另建label vault，不在此工具中实现。

该简单实现一次投影扫描并保存合格源行，内存随源大小增长；内存不足必须失败，不允许改抽样算法或只选已读前缀。预检只读footer且不会承担完整内容扫描。

## 已执行验证

```text
python -B -m unittest -v test_select_hotpot.py
```

12项测试通过。覆盖实际写出的两份人工Parquet的严格投影、默认不解码、非代表暴露整组排除、NFKC/casefold、代表context原样保留、固定随机数与96×4平衡、ID冲突、不合法列/嵌套结构、来源SHA/revision篡改、缺少或错绑准入、禁止覆盖和重复JSON键拒绝。测试包含仅为验证投影而制造的假标签列；没有benchmark或真实gold。

初版有一条测试把“空白合并”误期待成删除标点前空白，已纠正测试预期，算法未因此改变；失败回执保留。尚无独立外部代码审查通过声明。完整SHA与测试经历见DELIVERY_RECEIPT.json。
