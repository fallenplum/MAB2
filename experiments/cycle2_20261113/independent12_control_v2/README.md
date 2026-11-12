# 12 次独立采样加聚合：v2 费用核对修复

部署目录可命名为`independent12_control_v2`；源码只使用包内相对依赖，不绑定本地主机路径。请按字节解包/归档，不经过文本重写或LF/CRLF转换；构造器依赖pin比较的是原始字节。先在包目录执行`python -B verify_package.py --root .`，再用不改历史报告的`python -B -m unittest -v test_independent12_executor test_cost_reconciliation`做只读CPU重放。`run_cpu_tests.py`会生成本地测试报告，应只在可写工作副本中使用。

本包可公开的原v1独立审查为`ORIGINAL_V1_REVIEW_PUBLIC.md`和`ORIGINAL_V1_REVIEW_RECEIPT_PUBLIC.json`。独立审查私有目录里的完整检查器、运行日志及封装脚本不在部署白名单中；它们与v2修复者测试是不同证据。

v1 原目录保持不变。v2仅修复同一call_id多状态费用核对，并使用新的control_id/源码哈希/协议身份；信息权限、13槽位、路由、种子与生成策略不变。原14项CPU测试、原2项边界probe及新增9项费用测试均通过。这里的v2修复与测试由修复者执行，不冒充另一个独立审查。

同一(model,call_id)的一次请求须有完整一致seed；已出现的prompt指纹或开始时刻若冲突也使身份不确定。同一身份的费用字段只有所有报告均完整、有效且相等才计一次。缺失/冲突为null，保留每项observed_min/max、missing/invalid/conflict，以及原physical_events。observed范围仅描述报告证据，不能冒充真实费用上下界；known_cost_components只累计无歧义部分，known_cost_ranges的未知上界为null。不同状态不等于重复生成。缺call_id不合并；单请求出现多个身份或跨请求复用身份均明确不确定。返回Generation也不会覆盖冲突的事件报告。峰值的max仅在无歧义的独立请求间汇总，不用于挑选冲突记录。

**入口已实现，仅用于代码审查与 CPU mock 验证；没有实模型资格或 GPU 运行准入。** 原 graph qualification 不覆盖本新 control。冻结依赖来自提交 `b7434719eebb93eb41770bbc9af63222e3a47380`，复制字节不变，未修改原仓库或 Git。

`independent12_executor.py` 提供 `Independent12Executor(router, config=None).execute(question, seed)`。调用者显式提供已构造的冻结 `RoutedBackend`，本模块不会自动加载模型、连接服务或安排 GPU。默认配置与原三轮配置一致：worker 输出 128 tokens、聚合输出 256 tokens、完整输入 8192 tokens；这里 `rounds=3` 仅用于三独立样本的既有节点命名。

```python
from independent12_executor import Independent12Executor, Independent12ExecutionError

# router 由调用者显式提供；本文不包含模型启动命令。
control = Independent12Executor(router)
try:
    result = control.execute(question, seed=99191)
except Independent12ExecutionError as error:
    partial_result = error.partial_result.to_dict()
    # 调用者保存 partial_result 和原后端/占用账本；不得自动重试。
    raise
else:
    output = result.to_dict()
```

运行 CPU 测试使用 `python -B run_cpu_tests.py`；测试会读取本包内原字节冻结依赖，不导入 torch 或 Transformers，不加载权重。可在导入时将 `frozen_dependencies/src` 和 `frozen_dependencies/experiments/cycle2_20261113/heterogeneous_qualification_v2` 加入 Python 模块路径；测试脚本已处理此路径配置。实际集成应使用相同字节依赖，构造器会核对 executor 和 router 源码 pin。

**信息权限与固定顺序**

| 阶段 | 可见信息 | 禁止的信息 |
|---|---|---|
| 每个 worker 样本 | 完整同题、完整显式 context、该角色的原 round-1 提示 | 前样本、自状态、其他 worker 输出、原 gold |
| aggregator | 同题/context、全部 12 个归一化 wire 槽位及 role/sample/node 元数据 | 删槽后挑选候选、失败后替换答案、隐藏重试 |

顺序固定为 role 0 的 sample 1、2、3，然后 role 1、2、3 各三个样本。模型路由 `[Qwen, Mistral, Qwen, Mistral]`，聚合器为 Mistral；完整执行为 **6 次 Qwen + 7 次 Mistral**。三个同角色 worker 的提示逐字相同，等于原 round-1 提示，没有新增 sample 标记。`r{sample}:w{role}` 只承接原路由与 `derive_seed` 的标识，12 个 worker 使用各自确定种子，聚合器仍使用原 `aggregate` 种子标识。

**这不是 no-edge graph。** 原 no-edge 仍有三轮自状态，本入口的 worker 完全没有前序状态。新 protocol identity 记录独立 control 名称、源码与依赖哈希、全部配置、模型 identity、槽位顺序、提示和失败政策；不接受 mask/base 参数，也不读写图缓存。

**失败与成本接口**

`result.slots` 始终保留 12 个 worker 和 1 个 aggregator 的计划槽位。正常槽位保留完整 prompt、种子、原始 Generation、原 normalizer 产出的 wire/parsed/metadata、输入 fingerprint 和输出 integrity digest；`result.nodes` 提供已形成的 NodeRecord。发送 wire 按发送 tokenizer 归一化，聚合接收成本按 Mistral tokenizer 独立统计。完整模型 prompt 的模板化和上限检查仍由原 backend 执行；本入口不截断上下文。

- JSON 格式失败：保留原始付费生成及 `FORMAT_ERROR` wire，不删槽；聚合器仍看见全部 12 项。
- backend 异常：立即停止，不重试；异常携带 `partial_result`，失败槽和未尝试槽可区分；worker 异常后不调用 aggregator。
- 生成已返回但长度或归一化检查失败：保留原 Generation 与费用，记录 `POSTPROCESS_ERROR` 后停止。
- 生成前拒绝没有物理失败行时：记录缺失成本为 `null`，另保留已知成本部分；不将缺失值当 0，也不伪填完整 13-call 逻辑服务。

`physical_cost` 与 `logical_service_cost` 在本无缓存入口中相等。`calls` 明确定义为**已尝试 backend 请求，包含拒绝**；完整成功时均为 13，失败时只能与相同调用语义的账本比较。`returned_generations`、`unattempted_calls` 和未知费用字段用于区分生成、拒绝和未执行。`known_cost_components` 仅为已知部分，不能替代总额。`resident_GPU_hours` 保持 `null`；本 adapter 没有 GPU supervisor，后续实际运行必须由外层提供完整占用账本与授权预算。

`status=COMPLETE` 仅表示全部 13 个槽位完成执行；它可以同时伴随 aggregator 的 `FORMAT_ERROR` 和空 `answer`。调用者必须检查聚合节点的 `message_metadata.format_ok`、`format_failures` 及后续评分规则；不得把 COMPLETE 当作答案有效、质量合格或工程准入。

**验证范围与限制**

优势：沿用原角色提示、节点种子流、sender normalizer 和 backend/router，减少不必要的实现差异；CPU 测试覆盖独立信息边界、完整槽位、6Q+7M、种子、异构计数、格式错误计费、异常无重试、未知成本和新协议隔离。

风险：聚合器需读取 12 候选，输入长度明显不同于原 4-worker 聚合；真实 tokenization、格式率、可重复性、运行资源、上下文上限与真实评分链均未在本入口上实测。测试使用 synthetic mock，不能证明模型质量或科学结论，也不替代实模型工程 qualification。

主测试 23 项（原14+新增9）和原边界探针 2 项均通过。独立探针额外覆盖 normalizer 抛错后的 Generation/费用保留，以及聚合 tokenizer 在第 13 次请求前抛错时保留全部 12 个候选。结果见 `TEST_RESULTS.json`、`CPU_TESTS.log` 和 `INDEPENDENT_EDGE_PROBE.json`。

后续如要实际运行，应另行冻结和准入该 control 的实模型工程检查；当前资源准入为 false。本包不读取 R1 结果或 benchmark，不含真实任务原题答案、权重、主机信息或私有路径。
