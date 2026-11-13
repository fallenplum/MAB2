# Hotpot 384 题采集 runner / parent：实现前接口设计

状态：设计与 CPU staging；未冻结、未分配 GPU、未读真实题、未运行真实采集。完整读取了当前 PROTOCOL.md / PROTO.json。48 GPUh 只保留为未授权建议，任何执行必须使用另行冻结的实际资源回执。

## 边界与输入

编排器不采样、不评分、不读取 label vault、EM/F1、信用或答案正确性。固定 selection manifest 为唯一 384 题全集；输入 JSONL 只允许 collector 的五字段白名单，逐行 hash 与 manifest 对应。调用已发矩阵执行器 `HotpotApplicabilityMatrixExecutor(...).execute(payload, 26111301, 26111302, probe_edge)`；不得修改矩阵源、控制提示、缓存、单边/seed顺序。collector 只调用冻结 `collector_payload`，保留完整 context。

准备接口：`prepare_plan(selection, questions, length_receipt, protocol, source_bindings, budget_proposal, output)`。只验证已选全集并生成计划，不启动模型。length receipt 绑定相同 collector payload、双方 tokenizer 身份和每题的无标签 question+完整context token 数；不得用回答、gold 或实际正确率决定长度和顺序。源码先实现纯数组规划，真实输入准备由 root 另行授权和冻结。

运行接口：`parent --plan <pinned private plan> --admission <pinned execution receipt> --output <fresh attempt directory>`。计划含 2–4 worker 的 GPU 绑定、每卡 cap、全局 sum cap、disk guard、模型/源码/资格/runtime pins、唯一 attempt/ledger transaction ID；所有必填 pin 和权限状态缺失即在启动前拒绝。当前候选科学合同和未签 cap 均不可执行。

每个 child 接收自己的计划切片和独立输出目录，先启用 parent-death 与 monotonic deadline，再加载两模型；route 固定 Q/M/Q/M，aggregate=M。真实 factory 沿既有 HFBackend 接口构建，模型文件身份逐项绑定；不传预构建 router 给矩阵。所有模型加载和驻留等待均进入 parent 计时。

## 前 8 题与唯一释放屏障

按冻结 manifest 的每题 `max(Q_payload_tokens,M_payload_tokens)`、再按 UTF-8 原始 ID 排序，分为 8 个各 48 题长度层，PCG64(261113104)每层抽 1 题。前缀的 8 题固定完整运行全部 11 条件，随后其余 376 题保持原选入顺序；worker 分配在计划中固定，不按结果或完成速度重新挑题。

所有 worker 在 prefix 结束后保留驻留并等待。前缀费用已是正式轮次费用，不重复加载为免费资格，也不扣除前缀后再重算 N。任一基础设施失败、缺终态、费用未知或来源错配立即停止，不进入剩余题。

parent 仅从不含原文/答案的 cost-only prefix receipt 读取每节点/模型/长度层 request wall 时间、完整 unit wall、一次性加载和 I/O/共享负载记录。余下每题用固定几何的 planned request 清单及对应长度层实际观测耗时预测；缺少所需观测不填零。预测是少量样本的工程外推，不是严格耗时上界。至少乘 1.25；将当前全部 resident 秒、等待及必要余量一起纳入 global 与每 worker cap。完整余量不能 fit 时写 INCOMPLETE_RESOURCE，保留前缀，不改 N/cap、不重选。

仅在 8 个前缀 unit 的 88 个终态闭合、forecast 与父进程当前账共同满足 cap 时，parent 原子写不可覆盖 `FORECAST_FROZEN.json`，再签 `RELEASE_REMAINDER.json` 绑定其 hash。child 只接受本 attempt/plan 的释放记录。没有释放文件不得越过第 8 题；无质量访问、无自动 retry/resume。

## 输出与费用

每题私有保存完整 `MatrixResult.to_dict()` 和内容 hash；失败保存 `MatrixExecutionError.partial_result`。合法 INPUT_REJECTED 保留为终态并继续既定条件，缺 base 的 fresh 语义由原矩阵负责；其他故障停止尝试，不能填成错答或只评分成功子集。runner 不主动提取 answer 到摘要。

物理 native events、矩阵 request ledger、路由记录和 parent residence ledger 保留各自原始证据与身份。必要的前置调用 journal 只透明委托 backend.identity/encode/decode/generate，不改 prompt/RNG/返回值；开始但没有结尾的请求保持未知费用，不把事件 status 当成多次生成。请求计费复用既有 v2 reconciliation 语义，同一物理请求不能因多状态重复计费。

monitor 只发布 counts/status/cost/health：完成题数、terminal condition 数、成功/overflow/不完整数、已知 calls/tokens、未知费用标记、monotonic residence、磁盘/退出状态。禁止原题、raw generation、parsed answer、逐题质量或选边信用。明细只在 private 原始文件，公开 receipt 不含本地路径或问题ID。

总完成需要 384 个唯一 unit、4224 个唯一终态、固定 source/seed/mask 绑定、所有子进程关闭及费用闭合。任何 overflow 标记 COMPLETE_WITH_CAPACITY_FAILURES 并阻断机制升级，但 runner 不评分。原始文件 manifest 冻结之后才可能由独立 evaluator 释放评分；本包不实现该释放器。

## parent 硬停止

Linux production parent 为每 worker 新建自有 session/process group，保存 Popen PID/PGID；deadline/disk/failure 停止只对仍活着且 PGID 匹配的自有组发 SIGKILL，不广杀系统进程。parent 使用 monotonic 时间逐 worker 从 spawn 前到退出观测计占用，global 为这些驻留秒之和；不从 CUDA event 或 token 推 GPUh，不以 cap 截断实测花费。

子进程同时用 parent-death SIGKILL 与独立期限定时器防父进程死亡/卡住；不得直接复用旧 qualification 的固定 1760.5 秒上限，另建编排 guard 并保留旧源。退出时保留未闭合请求与费用未知。disk guard 检查启动前、每题前和 parent poll；无自动扩大磁盘/时间/样本预算。

## CPU 检验计划

使用人工 384 行输入/长度及 fake matrix/backend，不读取真实 benchmark，不加载权重。覆盖：固定 8×48 前缀可复现、剩余原序；严格 8 题屏障；+25% forecast 与 per-worker/global cap；未知耗时拒绝；11-condition 覆盖/重复拒绝；overflow 继续、infra partial 停止；无 retry；新 attempt/源/释放 hash 错配；monotonic 实耗与共享账去重；disk stop；只 kill 自有 PG；monitor 白名单不泄漏 raw 字段。Linux guard 的系统调用用有界 monkeypatch 验证，并明确 Windows CPU 测试不替代真实 Linux 工程资格。
