# MAB2 下一阶段独立代码与科学权限审查

审查基线：提交 `5397a8ea9dd5f5b8f0fc7e1671cd3319d81cf3ad` 的 `heterogeneous_qualification`、`hotpot_support`、`response_mediation_diagnostic`，另复核独立新增的 `heterogeneous_qualification_v2` 预算修复。原三个模块的源码、协议和测试文件与该提交字节一致，详见 `REVIEW_RECEIPT.json`。

## 结论与权限

**旧 v1 资格监督器存在预算阻断问题，不能据其静态检查启动模型。v2 修复经静态复核和 Linux CPU 实测闭环，支持在原 0.5 GPU 小时预留内启动一次有界工程资格检查。**

这项结论只针对工程检查准入。在本次审查截点，协调代理报告的 64 次真实模型资格生成尚未运行；本审查没有执行模型，也没有验收真实重放、格式表现、内存需求或模型驻留费用。没有批准正式候选、数据集实验、科学效果主张或任何额外预算。后续仍须核对完整部署 manifest、64 次实际生成、重放等价证据和闭合的实际费用账本。

本审查未读取新 R1 结果、任何真实题目、原始历史执行缓存、gold 或封存测试；未自行 SSH、访问 GPU 或修改科研仓库。仅运行现有纯合成 CPU 单测，另提供独立 Linux CPU 进程守卫探针；Linux 执行由协调代理完成，本审查只读核实其私有回执。

## 发现一：v1 父进程死亡可绕过驻留上限

**严重性：P1，已由新 v2 修复，旧失败必须保留。**

位置：`heterogeneous_qualification/qualify_routed_executor.py:105` 的 `start_new_session=True` 子进程，以及父进程的定时循环和 `finally` 清理。v1 没有子进程的 parent-death signal，也没有独立期限；直接 `--child` 入口亦没有独立预算守卫。

复现场景：模型子进程已经启动后，监督父进程收到 `SIGKILL`。此信号不会执行 Python `finally`；新的 child session 不随父进程退出。若没有其他机制，该模型进程可以继续运行，故不能保证 0.5 GPU 小时边界。这是源码可识别的逃逸路径，没有通过故意运行模型超时来复现。

正常超时分支使用 `killpg(proc.pid, ...)`，目标是它自己创建的新 process group，未发现会按用户名、GPU 或通配 PID 误杀其他研究进程的代码。

v2 的修复闭环：

- `gpu_job_guard.guard_before_exec` 在 `exec` 前设置 `PR_SET_PDEATHSIG(SIGKILL)`，随后核对预先捕获的监督 PID，关闭 fork／设置 prctl 间的父进程死亡竞态。
- `arm_child_deadline` 验证真实 parent PID 和已经启用的 PDEATHSIG；使用同一系统 `CLOCK_MONOTONIC` 的绝对期限，过期立即退出。
- 子进程在 preexec 和读取 plan／导入模型前均设置 `SIGALRM` 的默认终止动作，显式解除可能继承的信号屏蔽，并设置 `ITIMER_REAL`。因此不依赖 Python 自定义信号回调获得 GIL。
- child 使用监督起点加 1760 秒；parent 仍保留 1770 秒限制及 10 秒终止宽限。两套守卫均不重置绝对时间预算，实际占用仍需如实记账，不可裁剪成上限值。

限制：PDEATHSIG 本身覆盖直接模型进程及其线程，不能承诺杀死未来另起 session 的 GPU 后代。当前支持的串行 HF 后端没有额外模型进程；未来若引入 multiprocessing、服务端或另一个 GPU worker，应重新审查。父进程被 SIGKILL 后，状态文件可能无法写成 CLOSED；即使子进程已经停止，也必须人工或独立恢复账本，不能把缺失 CLOSED 当作零成本或自动重试。

## 发现二：响应分类是答案字符串变化

**严重性：P2，解释边界，不要求改变已冻结分类。**

位置：`response_mediation_scan.py:75` 与 `:80`。`equalanswer` 比较 `parsed['answer']` 的原始 UTF-8 字节哈希。例如人工字符串 `"5"` 和 `"5.0"` 会进入 `changed_receiver_answer`，虽然数值上相等。首尾空白和其他呈现差异也可能进入该类。

因此该类应解释为“receiver answer 字段的字面字符串变化”，不能改称“语义答案改变”或“接收者判断改变”。协调代理已决定保留原分类、协议、代码及输出，通过结果前新增解释附录明确该定义；本审查接受这一处理，不据此重新分组、调门或选择候选。

## 发现三：单题重叠支持不能产生可解释的总体区间

**严重性：P2，条件性报告限制，真实支持数未读取。**

当两类响应在仅一题中同时存在时，代码仍会对该题级单元素向量 bootstrap，产生零宽百分位区间。例如 `[0.2]` 的所有题级重抽均值仍是 0.2，不能据此宣称总体关联已被精确估计。

协调代理已决定若发生 `paired_questions < 2`，保留原始输出，但公开摘要不展示该区间或赋予推断含义；此诊断不作候选准入门。即使支持多于一题，其区间也只是旧开发数据的探索性条件关联，不是因果中介效应或独立确认。

## 未发现阻断错误的部分

异构执行器：`RoutedExecutor._messages` 先选择当前节点，再组装输入；入站消息 token 用接收节点的 tokenizer 计数，当前节点生成后的 wire 规范化使用该节点自己的 tokenizer，即下一跳的 sender tokenizer。完整模型身份、路由、adapter 源码、base executor 配置和代码进入协议身份，改变路由／模型会保守地使旧执行缓存整体失效。并发使用被拒绝，失败没有自动换模型或静默重试。

三轮重放：完整图删去固定一条边时，受影响闭包为 6 个节点。每题完整旧图 13 次、局部重放 6 次、全新新图 13 次，同图缓存复用 0 次，共 32；两个人工任务共 64。真实资格脚本检查新图的局部重放与从头执行在完整 messages、input fingerprint、生成原文、wire、parsed 和 metadata 上一致；这些实际模型检查目前尚无通过证据。仅合成单测通过不能替代它们。

HotpotQA：`answer_metrics` 与仓库中固定官方源文件的实际函数 AST 在 2,006 个人工字符串对上完全一致，并检查源文件和 license 的 Git blob。支持范围严格限于 answer EM／F1／precision／recall，不冒称 supporting-fact 或 joint 指标。上游来源固定为 `hotpotqa/hotpot` 提交 `3635853403a8735609ee997664e1528f4480762a`；本次核对为 vendored 源码及其 blob 证据，没有声称重新检索官方网络仓库。

HotpotQA transport：malformed message 统一计零是明示的未来 HotpotQA 传输约定，区别于原 GSM8K 的 raw-answer 提取。未改变既有评分。collector 白名单只接受输入与元数据字段，拒绝 answer、gold、supporting_facts 等标签；保留所有原序文档和句子，不做支持事实筛选、检索或截断。后续数据准备仍须单独预先冻结划分、重复题分组和完整 prompt 上限检查。

历史响应链：代码检查非受影响非聚合节点 wire 不变；若 receiver 完整 wire 相同，则验证 aggregate input fingerprint、原文和历史信用分数相同。aggregate 数值答案统一用固定原 scorer 提取，再用精确 Decimal 表示比较，保留解析失败类别。题级聚类保持同题 anchor／edge／seed 依赖。

历史条件关联不能解释为干预 receiver answer 的因果效应，也不是无偏信用替代。该 token 估计只针对 off-world baseline 已缓存后的 edited receiver 加 aggregate 后缀；即使 exact-wire 合流可以跳过 aggregate，也不能遗漏银行建设、反馈、服务及失败成本，更不能给候选方法独享该通用缓存优化。

## 实测证据与界限

本地 Python 3.13.3、`-B` 禁止写 pycache 的只读执行：8 项异构路由测试、6 项 HotpotQA 测试、1 项 Decimal 精确表示测试，共 **15 项通过**，unittest 报告 0.076 秒。完整测试名和所测源码 SHA 在 `REVIEW_RECEIPT.json` 中；没有重跑模型或读取真实数据。

Linux Python 3.11.9 CPU 探针：**7 项独立系统原语测试、7 项真实 `gpu_job_guard` 函数测试全部通过**，实际回执耗时 2.2411018749698997 秒，退出码 0。探针确认父进程 SIGKILL、默认 SIGALRM 忙循环终止、正常退出、只终止自有进程组、父 PID 失配、过期期限和继承信号屏蔽解除。独立原语的一次观察为 zombie（Z）：代表已终止执行、尚未被 reaper 回收，已在回执明确区分；真实 guard 的对应观察为进程消失。

探针 SHA256：`5fdb82ed68dff3a964fbc6858f206f94631aad6597148902144f38b5fa67f698`。

真实 guard SHA256：`f281ffca929966d923d82f7d61227ce5977731b3cb343c5279145d1af970265f`，与复核时 v2 源文件一致。

Linux 私有外层回执 SHA256：`125fb986b2f92fa7b6f60f66f990a91585cdbe6b6c5c3f2c44152165cdb7c8c8`。归档只保留脱敏测试字段和内容哈希，不公开服务器命令、主机、凭据或绝对部署路径。

优势：模型／协议与预算边界有明确检查，系统信号行为已有独立实测，Hotpot 指标与既有科学结果隔离。风险：尚缺真实模型重放和格式证据；CPU 测试不能证明 CUDA 故障、显存需求、硬件不可中断状态或完整账本恢复行为。

置信度：固定源码、字符串指标及已实测的 Linux 信号行为为高；真实模型工程资格、科学增益与正式研究准入尚未评估，不提供成功概率。
