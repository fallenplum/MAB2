# 长输入完整矩阵工程资格准备

状态：CPU 准备，未授权启动、未预留预算、未获真实模型资格。Root 独立复核、源码归档和资源封账后才能生成可启动私有 plan。本目录没有读取 benchmark 题、R1 结果或标签。

入口为 `qualify_hotpot_matrix.py --plan <private-plan.json>`。公开模板不可启动。Root 负责服务器部署、预算预留、实际选卡和最终驻留闭账；本包没有远程控制器。

固定两跳人工 QA，47 段干扰文本全部保留；Mistral 低层本地 tokenizer 估计完整用户文本 3744 tokens，不等于服务器 chat-template 输入验证。服务器必须记录完整用户文本的两模型 token 数，Mistral 须在 3500–4000 范围；每次 chat-template 实际计数保留在 Generation 中。所有控制拿到同一完整 context，无裁切或删题。

一次矩阵按 S1 的 A、B、NO_EDGE、A_MINUS_E、B_MINUS_E、INDEPENDENT12、SINGLE_M，再 S2 的 A、B、A_MINUS_E、B_MINUS_E 执行。probe edge=1，seed=26111301/26111302。两次 fresh 矩阵合计成功物理调用 174（Q80/M94），逻辑节点 262；174 是 admission 硬上限。首个非 COMPLETE 条件立即停止，无隐式重试。格式失败保留且计费，不作为准确率门。

矩阵来源固定 `c6a687afa179c826813b11ce3472fe1e8f3166bf` 的 exact bytes，Graph、router、independent12 v2 均有固定 SHA。私有 plan 还必须绑定模型完整文件清单、各完整 backend identity digest，以及预先计算的 matrix protocol hash。后者可以使用既有受审 backend identity 创建仅 identity 的 CPU stub 来构造矩阵获得；不需要新模型推理。源码与模型身份不匹配时不推理。

新增共享 `durable_backend.py` 只观察，原参数对象、seed、config、identity、tokenizer 与返回 Generation 对象均保持不变。原 HFBackend 设 `physical_log=None`，保留其内存 physical_calls；共享层先 fsync START，生成返回后将完整消息、Generation 和本请求 native 事件一起 fsync，再将原对象交给矩阵。journal 是私有原始材料，不是新的物理费用行。成功时按 174 个唯一 call_id 精确核对 raw Generation、matrix trace、route、native，并独立重算全部逻辑/物理费用。

条件完成时另存 PARTIAL checkpoint，矩阵完成时保存完整对象。异常处理沿原矩阵 cause 链提取 DurableBackendError 中已返回但未持久化的 Generation，尝试写入 FAILURE_PARTIAL。持续写盘失败、进程硬杀或 backend 尚未返回会留下 pending START/未知原文或费用，不能宣称全部已付原文均可恢复，未知费用不填零。磁盘故障之后不继续新请求。

父监督沿原 v2：1770 秒 parent、1760 秒 child、0.5 GPUh，装载时间纳入驻留；显存门 32000 MiB，只选择没有实际 compute PID 的 permitted GPU，不停止其他任务。原 `gpu_job_guard.py` exact bytes 复用，原 14 项 Linux guard 测试提供已有实现证据；本次只核字节一致并做 CPU 静态/集成测试，没有把数值计划当作新的 Linux 实测。

相对原 v2 supervisor 的语义变更：reservation 独立；compute occupied 卡严格排除；状态写入 flush/fsync；main 增加 launch/source/model/identity 审批字段校验；parent 捕获 plan bytes 的 SHA，启动前复核，并传给 child，child 先 arm 原 guard 再核同一 plan bytes 后解析。完整源 diff 见 SUPERVISOR_DIFF.patch。parent 进程组清理和 deadlines 未放宽。

优势：固定源码与身份、调用几何、缓存来源、生成原文和逻辑/物理费用可分别审计。限制：CPU mock 不证明实际 BF16/模型/tokenizer/长输入/吞吐可行；单个人工问题也不证明答案质量。吞吐外推只在相同软硬件、此长度和成功缓存几何下成立；384×87 的预计时间不构成资源准入或 benchmark 实测。
