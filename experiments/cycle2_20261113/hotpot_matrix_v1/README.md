# 单题适用性矩阵执行接口：CPU staging

**已实现接口与 CPU 验证，未运行实模型工程资格；不采样 benchmark，不读取标签，不评分或选择统计结果。** scientific contract 由外部另行冻结。本模块是新 operator，原 graph / independent12 的资格不能自动转移到本入口。

`HotpotApplicabilityMatrixExecutor(backends).execute(question, seed1, seed2, probe_edge)` 接收两个已由调用者构造的 backend（`qwen`、`mistral`）、一个显式题目、两颗不同整数种子和给定边。模块不加载模型、读数据文件、选择问题、安排算力或创建 supervisor。

```python
from matrix_executor import HotpotApplicabilityMatrixExecutor, MatrixExecutionError

# backends 和完整 question/context 由调用者提供；代码中不含真实启动命令。
matrix = HotpotApplicabilityMatrixExecutor(backends)
try:
    result = matrix.execute(question, seed1, seed2, probe_edge)
except MatrixExecutionError as error:
    private_partial_result = error.partial_result.to_dict()
    raise
else:
    private_result = result.to_dict()
```

导入路径需包含 `frozen_dependencies/src`、`frozen_dependencies/router`、`frozen_dependencies/control` 和本目录；测试脚本已配置。`python -B run_cpu_checks.py` 只运行 synthetic CPU mock，生成测试日志与接口/prompt 合同。不得把该命令当作实模型资格。

**几何与缓存来源**

边索引为原 source-major 的 **0-based** 索引。B=`{1,5,7,9}` 对应 `0→2→1→3→0`，probe edge 必须属于 B。A 为全部 12 边，NO_EDGE 为全 0；两者均保持三轮 self-state，因此 NO_EDGE 不等于 independent12。

| 条件 | seed1 物理请求 | seed2 物理请求 | 每次完整逻辑调用 | 唯一允许原 base |
|---|---:|---:|---:|---|
| A | 13 | 13 | 13 | 无，fresh |
| B | 9 | 9 | 13 | 同 seed 的原 A |
| NO_EDGE | 9 | — | 13 | 同 seed 的原 A |
| A_MINUS_E | 6 | 6 | 13 | 同 seed 的原 A |
| B_MINUS_E | 4 | 4 | 13 | 同 seed 的原 B |
| INDEPENDENT12 | 13 | — | 13 | 无，独立协议 fresh |
| SINGLE_M | 1 | — | 1 | 无，完整原题/context，无候选 |

成功顺序固定为 seed1 的上表七项，再执行 seed2 的 A、B、A_MINUS_E、B_MINUS_E。独立 reachability 计算对四条 probe edge 均得到 **55 + 32 = 87 次物理生成、131 次完整逻辑调用**；随后与冻结 `affected_closure` 逐项核对。`GEOMETRY.json` 保留每条边的闭包和预期模型调用数。

87 是全成功且原 base 可用时的保守缓存估计，**不是失败路径的硬上界**。原 A 或 B 失败时，对应后续条件用相同题目/种子的 fresh 执行，不借用其他 control、另一个 seed 或不完整 base。CPU 用例分别验证过 90、102、88 次请求的输入拒绝路径；这些是 synthetic 控制流测试，不是实测 GPU 次数或预算建议。

**透明观察与输出**

图计算仍由原 `RoutedExecutor` 执行，original base 直接传入。`ObservedGraphExecutor` 仅观察 `_messages` 接收到的已有节点及计数；不修改提示、闭包或缓存判断。`ObservingModelBackend` 将 identity、encode/decode 和 generate 参数及返回的 Generation 原样委托，同时保留已返回原文和原物理事件。测试确认图 protocol hash、backend identity 与直接冻结入口一致。

matrix 另有新协议身份，绑定本模块/几何代码哈希、冻结依赖、顺序、base 规则和单 M 提示。所有完整 graph 输出及原物理/逻辑费用保留；失败时保存已有 NodeRecord、已返回但尚未形成 NodeRecord 的 Generation、route 事件和费用证据。失败后的部分图不会成为后续 base。

上述保留发生在结果对象内，本模块不写运行产物文件。SIGKILL 等进程级终止不能返回 Python 部分结果；未来独立工程入口须另行实现持续落盘、硬终止恢复和 GPU 占用监督。

观察代理尤其处理“模型已返回、route 日志写盘失败”的边界：原图中尚未形成的当前节点输出，仍可从 request trace 恢复。若模型 backend 在返回前自身失败，模块只保留其实际可见事件，不推测内部未返回原文。

每个条件和总矩阵均复用 independent12 v2 的同 call_id 费用核对；一致完整多状态不重复计费，冲突/缺失保持 `null`，跨条件错误复用 call_id 会令总费用未解决并中止。`total_physical_cost.calls` 的口径为**已尝试 backend 请求，包含生成前拒绝**；`returned_generations` 单独标明返回生成数。完整成功时二者均为 87。CUDA event 时间不能代替占用账本，`resident_GPU_hours` 始终为 `null`，须由外层真实运行监督记录。

**单 M 是单独的新 operator**

SINGLE_M 使用固定的强直接问答 system 提示和三字符串 JSON 格式指令。它只看到同题完整 question/context，没有任何候选、worker 输出或 self-state；输出上限固定 256，以 Mistral tokenizer 归一化，保留格式失败。它沿用 `derive_seed(question_id, seed1, "aggregate")` 以避免新增隐含随机流，但拥有独立 protocol identity；没有用 worker0 或原候选聚合器冒充。完整提示写入 `SINGLE_M_PROTOCOL.json`。

**终态、权限与限制**

- 只有冻结 backend 的两类精确输入上限异常可记为 `INPUT_REJECTED` 后继续；不会静默截断、删题或重试。此模块不赋 EM/F1，后续评分层必须按另行冻结的合同处理该终态。
- OOM、日志/存储、tokenizer 其他故障或完整性异常记为 `INCOMPLETE`，停止当前矩阵并抛出携带部分结果的异常；未执行条件保留 `NOT_RUN`，不填成错误答案。
- `COMPLETE` 仅表示控制流完成，仍可能伴随格式错误/空答案；不是答案有效、实模型资格或科学准入标记。
- 不同控制协议没有共享缓存；仅图条件按表共享同 seed 的原 base。caller 运行期间必须独占提供的 backend 对象，不能从旁路并发调用同一模型对象。模块不影响其他进程或任务。
- 完整输出含原 question/context 和逐节点原文，属于调用者的私有结果；不要把这些运行结果纳入公开代码包。

优势是固定地记录所有条件，保留实际缓存节省与完整逻辑费用，并与原算子逐项对照。风险是 12 候选与新单 M 提示的真实上下文容量、实际吞吐、格式率及故障行为仍需独立的人工 synthetic 实模型工程回放；当前 CPU mock 不能证明这些性质，也不能代替科学实验。
