# R1 分阶段分析适配器

这套工具把已经完整采集的 R1 图输出转换为开发集分析。它不生成新的模型输出，也不把开发试验解释为独立确认。运行前必须将 `r1_adapter.py` 和相邻的 `r1_policy.py` 提交到干净的项目分支；分析入口会核对实际执行文件与该提交的原始字节。

## 输入与先决条件

- 科学协议为已冻结的 `R1_PROTOCOL.json`，SHA256 为 `5336eacc6713d4e444ddb631032409cd2718ad3b4aaee1387aff5baac40f5ecf`。
- 执行 pin JSON 至少包含正式 `plan_sha256`；计划的全部 390 个单元、384 个正式问题、6 个资格问题及所有图产物必须完整。
- 输入应包含各工作进程的资格和正式完成标记、单元完成标记、图 checkpoint、模型调用流水和驻留时间流水。
- 只有明确签署的 `mab2_cycle2_resource_only_amendment_v1` 证书可以允许资格记录绑定旧计划、正式记录绑定新计划。计划差异仅允许三个工作进程的驻留时间上限重新分配；总量必须仍为 86400 秒，并保留资格耗费。
- 评分器必须是原始 GSM8K `gsm8k_explicit_answer_decimal_v2`，文件 SHA256 为 `da315ade4be7362ca81fc7ed41252cdccaa9271dc32f9a20531356587bb5f37e`。评分输入是聚合器原始生成文本。

以下 CLI 路径均为调用者提供的文件位置；输出目录必须尚不存在。`python` 需指向具备 NumPy 的既定实验环境。

## 第一步：独立 data-owner 进程拆分标签

```text
python r1_adapter.py split-owner
  --source-labels SELECTED_LABELS.jsonl
  --data-receipt DATA_RECEIPT.json
  --selection SELECTION_FROZEN.json
  --plan COLLECTION_PLAN.json
  --execution-pins EXECUTION_PINS.json
  --protocol R1_PROTOCOL.json
  --output RELEASED_LABELS
```

上面的换行仅用于展示；实际调用应使用同一条命令或所在 shell 的续行符。owner 会读取全部已选标签，但不打印答案，产生 I96、C96、S192 三个独立文件以及 `PHASE_LABELS.json`。它核对原始数据 receipt、科学选择清单和正式执行计划，并显式记录 owner 的权限不等于策略的反馈权限。将它作为单独进程运行，保存返回的 descriptor SHA256。

## 第二步：完整性检查与分阶段分析

```text
python r1_adapter.py analyze
  --plan COLLECTION_PLAN.json
  --execution-pins EXECUTION_PINS.json
  --protocol R1_PROTOCOL.json
  --run-root COLLECTION_OUTPUT
  --labels-descriptor RELEASED_LABELS/PHASE_LABELS.json
  --labels-descriptor-sha256 OWNER_RETURNED_SHA256
  --scorer ORIGINAL_SCORING.py
  --repo-root CLEAN_COMMITTED_CHECKOUT
  --output NEW_ANALYSIS_DIRECTORY
  --resource-amendment RESOURCE_AMENDMENT.json
  --data-root DATA_ROOT
```

最后两个参数只用于已冻结的资源修订，证书内相对路径以 `--data-root` 解析；没有修订时省略两者。程序不预设某一正式计划的 SHA，而是通过执行 pin、科学协议和修订证书验证它。

严格顺序为：

1. 核对所有单元、图、节点、缓存来源、调用成本、流水和计划；落盘 `OUTPUTS_FROZEN.json`。
2. 仅打开 I 标签并计算旧背景图的单边动作差；落盘 `PURCHASES_FROZEN.json`。
3. 打开 C 标签；每个策略副本只能通过其已购买边的接口取得反馈；落盘 `ACTIONS_FROZEN.json`。
4. 导出仅含 I/C 的 `R2_IC_INPUTS.npz` 和绑定 receipt。
5. 打开 S 标签，只评分已冻结策略实际选择的动作；落盘聚合结果及 `COMPLETE.json`。

每次标签文件首次读取前均记录阶段和前置冻结文件哈希。各阶段接收的是不同接口；精确均匀基线的每个边子集副本从相同 I 初始化，不能共享其他副本的 C 反馈。它们可以复用底层只读物理评分缓存，但该缓存不扩大策略信息权限。接口和 receipt 是可审计的软件边界，不构成操作系统权限隔离。

图请求中的 `edge` 是置换后的物理边，文件名和菜单顺序仍可能使用模板编号。适配器按照物理边与 `spec.editable` 对齐；不能按文件位置拼接数组。

## 产物与成本口径

`public/R1_RESULTS.json` 只含汇总结果、36 组方法指标、34 个预设对比、题组 bootstrap、逻辑费用和来源哈希。`private/` 保存逐题数组、采购与动作、评分记录、详细费用、阶段 receipt 和独立复算输入；该目录不应发布。

逻辑成本计算每个策略独立部署所需的 I、所购 C 和 S 调用。I 的每题菜单为 9+6×2 次调用；C 基图按冷启动 9 次、每个复检边增加 2 次计费；S 选定图按完整 9 次调用计费。不能把已生成缓存的 2 次增量费用当作未来问题的服务成本。均匀基线费用是随机子集的精确期望，不能乘以枚举副本数。无复检基线保留节省的费用；共同保守上限不代表实际支出相等，也不证明经济最优。

研究物理成本单列，包括资格、全菜单采集、失败及驻留时间。浮点图成本按生产端 Python 3.11 的节点顺序与逐项加法精确重算；整数 token 和调用数精确核对。其他历史实验的 GPU 费用在此工具中保持 `null`，必须由项目总账合并；不得将缺失费用当作零。

CPU 费用记录已测的主分析进程和唯一评分调用时间。未单独测得的导入、子进程或历史成本显式保留未知；评分 CPU 的逻辑复用不意味着发生了新的物理评分调用。

## 独立复算与测试

独立 R1 程序仅使用已经冻结的数组与 receipt，重建全部采购、动作、统计及逻辑费用，不能导入生产策略实现：

```text
python independent_r1_replay.py --analysis-dir NEW_ANALYSIS_DIRECTORY --output NEW_REVIEW.json
```

R2 程序只允许精确的 16 个 I/C 数组键；它核对采购后的动作来源和每个反馈哈希，不读取 S、S 的评分记录或 R1 服务结果：

```text
python r2_ic_diagnostic.py --private-dir NEW_ANALYSIS_DIRECTORY/private --output NEW_R2_REVIEW.json
```

`--synthetic-fixture` 只用于人工集成测试；正式数据不能使用此标记。完整合成测试使用独立生成的 390 单元、反序菜单、异质浮点耗时和原始评分器，不读取真实问题或标签：

```text
python RUN_ADAPTER_TESTS.py --replay-script independent_r1_replay.py --r2-script r2_ic_diagnostic.py --output NEW_TEST_RECEIPT.json
```

每次测试使用新 receipt 文件，保留旧记录；失败应保留已经落盘的阶段证据，修复后使用新的分析输出目录。合成 PASS 仅证明被覆盖的软件行为，不能代替真实数据验收、实验结果或独立确认。
