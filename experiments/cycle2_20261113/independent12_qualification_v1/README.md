# Independent12 单独实模型工程资格：准备稿

**本包只准备资格代码与协议，未预留预算、未启动 supervisor、未加载模型或运行 GPU。** CPU 静态检查不能写成资格通过。须先完成 root 独立审查、代码冻结和新的预算预留，再由明确私有启动计划授权运行。

资格对象是独立的 4 角色 × 3 样本加 aggregator 入口；原 graph qualification 不覆盖它。两个人工 synthetic 输入、原角色提示、模型路由、采样配置与 seed 沿用已有 hetero qualification。每题执行两次 fresh，共 **4 次执行 × 13 槽位 = 52 次物理生成：24 Qwen + 28 Mistral**。没有 benchmark、R1 结果或原 gold 读取，正式候选轮次增量为 0，不计算 synthetic 准确率。

**监督器与旧 v2 的精确差异**

`SUPERVISOR_DIFF.patch` 由 CPU 检查直接生成。相同函数的对照使用 `reference_v2/qualify_routed_executor.py` 的原部署源码。

| 部分 | 保留项 / 唯一变化 |
|---|---|
| `gpu_job_guard.py` | 原字节完全复用；SHA256 `f281ffca929966d923d82f7d61227ce5977731b3cb343c5279145d1af970265f` |
| `parent()` | 函数文本仅将 reservation 名称改为 `C2_INDEPENDENT12_ENGINEERING_QUALIFICATION` |
| `main()` | 只在读取 plan 后、分派 parent/child 前新增 `validate_launch_plan(plan)` |
| `sha` / `write` / `pin_sources` | 函数文本原样复用 |
| `child()` | 新增独立采样入口、逐槽检查点、52-call 审计与 fresh 语义比较；不沿用原图边删除实验 |

parent 继续读取真实 GPU 列表和 compute-process 列表，优先选择无其他计算进程的合格卡；允许卡范围由私有 plan 给出，最低空闲显存固定 **32000 MiB**。不会停止其他任务。启动的模型子进程只占一张卡；parent 只终止自己创建的子进程组。

原 parent 单调时钟期限 1770 秒、child 绝对单调时钟期限 1760 秒、终止宽限 10 秒、0.5 GPU 小时上限均保留。guard 在 exec 前及 child 读取 plan 前重新绑定父进程和期限；`SIGKILL` parent-death、捕获父 PID 的竞态检查与 `SIGALRM` 计时器保持原逻辑。本范围仅为一个直接模型进程及其线程；不允许额外 GPU 子进程、脱离监督的后代或权限扩张。本包没有重复先前的 14 项 Linux guard 测试。

新 admission 校验要求：明确启动授权、root 审查收据哈希、冻结提交、**新** reservation 收据、完整源码 pin、固定 control/guard pin、完整模型文件 manifest、私有新输出目录和显式允许卡。运行时禁止 Python 优化模式，防止 `-O` 移除沿用代码中的 assert。收据的真实性和全局预算余额仍须由 root 核验；字段格式检查不能替代预算事实。

**输入布局与私有计划**

公开代码只使用相对源码路径；所有模型、代码根、项目根和输出绝对路径由私有 plan 提供。部署逻辑路径如下：

- `src/graph_credit/{__init__,executor,backend}.py`
- `experiments/cycle2_20261113/heterogeneous_qualification_v2/routed_executor.py`
- `experiments/cycle2_20261113/independent12_control_v2/independent12_executor.py`
- `experiments/cycle2_20261113/independent12_qualification_v1/{qualify_independent12.py,gpu_job_guard.py,PROTOCOL.json}`

`deployment_sources/` 已按上述布局准备八份精确源码，`DEPLOYMENT_SOURCE_MANIFEST.json` 固定字节。它的 `source_commit=null` 表示仍待 root 的实际 Git 冻结；此处文件哈希封存不冒充已经有冻结提交。

`PRIVATE_PLAN.template.json` 故意保留 `launch_authorized=false`、`budget_reservation_confirmed=false`、空 source/model manifest、空允许卡列表和空收据；它不能启动。模板中的 0.5 GPU 小时是拟定上限，不是已经创建的预算预留。root 在审查、冻结、真实预算核验完成后才可生成另外的私有计划，将状态设为 `ROOT_REVIEWED_FROZEN_RESERVED_LAUNCH_APPROVED`；不得把模板中的空值替换成虚构收据。

control 依赖固定为费用合并修复后的 v2，SHA256 `d3ec41ca2df218592ef8b4dc06c7e26441ee2f0f20daad5b70c6df1ae10b9dbc`；不使用原 v1。v2 对同 call_id 的一致完整状态只计费一次，缺失或冲突费用保持未知。此次更新只固定该代码依赖，不把其 CPU 测试称为实模型资格。

启动入口支持 `--plan <PRIVATE_PLAN>`；本说明不附实际执行命令或私有路径。本次没有生成可启动的私有计划，没有预算预留收据，也没有调用启动入口。

**输出、失败与验收范围**

每次运行必须使用不存在的新输出目录。`INDEPENDENT12_IDENTITY.json` 保存模型/路由/control 协议身份、资格源码清单及私有运行时信息；后端与 route 日志名称独立位于该新目录。原资格和原 control 产物均不修改。

每个 fresh 执行在开始前记录 13 个计划槽位，并在每个槽位边界原子更新独立 `*.checkpoint.json`；返回了 Generation 而归一化失败时也保存边界快照。检查点始终标为 `completion_evidence=false`。捕获的执行异常另外保存 `*.partial.json`；完整执行先保存原始结果，再做资格审计。超时信号可能绕过 Python 异常处理，届时检查点、原始日志和 parent 的 `RESIDENT_STATUS.json` 是恢复依据，不能将残留 RUNNING 状态当成已完成。

两次 fresh 比较每个槽位的 messages、parsed、wire、input fingerprint、seed、归一化元数据、接收计数，以及 generation text、输入/输出 token 数和 finish reason。**不跨 fresh 比较包含 call_id / 运行时间的完整 output digest**；每个 digest 只对自身完整记录重新计算。正常完整路径还核对固定 13 槽位、无缓存、模型路由、唯一 call_id、52 次 route/physical/result 一一对应及逐执行/总体费用。

格式错误仍保留付费槽位并按模型报告；不设 synthetic 准确率门或将零格式错误解释为真实任务质量。一次失败不重试，不能产生 PASS。返回生成前拒绝、日志故障及其他未知失败费用保持 `null`，不填 0。`completed_execution_requests` 仅统计已形成完整执行文件的请求，不能作为失败运行的总成本；失败运行还必须结合 partial、原物理日志和占用账本。

child 写出的 `PASS_ENGINEERING_REVIEW_REQUIRED` 只表示子进程内部检查完成；占用费用必须在 parent 退出后从 CLOSED resident 账本另行核验。CUDA event 时间不能替代包含加载、失败、检查点 I/O 和退出阶段的 GPU 占用时间。

**准备检查与限制**

`check_preparation.py` 只做 Python 编译、监督器函数文本对照、guard 原字节 pin、矩阵与原 synthetic/config 一致性、默认模板拒绝，以及手工 ledger fixture 的 fresh 比较和路由/费用正反例。它不启动 parent、child、guard timer、GPU 或真实 tokenizer/model，也不重复 14 项 Linux guard 测试。

优势是监督与资源行为沿用已验证版本，新入口增加可审计部分产物和明确的 fresh 判据。风险是 12 候选聚合更长，checkpoint I/O 增加整体占用，实际模型格式、确定性、容量和完成时间仍未知；0.5 GPU 小时内未完成就应记录失败或未完成，不能自动扩预算或减小验证矩阵。本包不能证明实模型资格已过或科学准入。
