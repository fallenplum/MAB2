# 长合成上下文完整矩阵资格：独立工程准备审查

**结论：所审工作快照未见剩余阻断问题；25项CPU测试通过。此结论只支持进入另行授权的工程实测，不代表174次真实模型调用已完成，不授予科学实验或新预算准入。**

固定对象为 WORKING_MANIFEST SHA256 `a36c282ce8c5310ffc03f077125c05ef3e2dd0b536909c7e31a19a931cf1f695`；入口 `290e8e869b940ae4e028c0a9e2ce2d4bf987a034f6798f8320e10f9ba6a402e6`；算术审核器 `b1daa6690275fafd1ee46302856e58c096ac5c73461bfc6bbaed06ea0a6ae59b`；durable helper `0f39b5e1926fcc0670158e27783d1b0793436f6cb111d71a476a7794116aeade`。清单文件在测试前后逐字节核验且未改动。

## 身份、权限与算子

入口使用固定人工两跳问题及完整长context，不读benchmark或标签，不评分人工答案。Mistral完整用户文本须由实际backend记录并落在3500–4000 token；所有请求仍由冻结backend检查8192输入及128/256输出上限，不能裁切。单问题长度通过不能代替真实Hotpot容量证明。

两次fresh完整矩阵各87次成功物理生成、131逻辑节点，总174/262。共用RequestCap在生成前限制全局174请求，失败无自动重试，工程资格首个非完整条件停止。三轮图、independent12、singleM及缓存规则均来自已固定源码；wrapper不改变identity、tokenizer、消息、seed、config或返回Generation对象。新日志配置是明确的prospective变化：原HFBackend `physical_log=None`，保留原内存物理事件，由统一helper接管持久化；旧资格不能自动覆盖这项新配置。

父子双方绑定相同plan字节SHA。child先安装原Linux guard，再读/核plan，随后核源码、模型文件、backend identity及matrix protocol hash；公开模板仍不可启动。JSON配置用canonical内容核对，再传原默认GraphConfig(rounds=3)，避免JSON列表与Python tuple的表示差异误拒绝，未改生成参数。

## 原文和费用闭合

每次请求先fsync START；原backend返回后，helper把原Generation及原native增量一起fsync，再把同一对象返回矩阵。成功资格按174个唯一call_id核对journal、matrix、route、native，逐字节canonical比对完整Generation，并核helper源码头、请求cap/config/repeat、原始事件区间。131逻辑节点的token、自状态、接收消息、时间和峰值费用单独重算，不只看调用数或两次fresh一致性。

journal事件不产生第二份物理费用。START/回调写失败不得开始native请求；RETURNED写失败保留RAM Generation，poison wrapper阻止继续请求。矩阵异常链中的这份原文可被提取为单独应急证据；它不是新调用。已写完整RETURNED后报告I/O失败也不能重试或双算。

**保证范围止于已返回给helper的Generation。** OOM、SIGKILL或backend内部尚未返回的文本不能保证恢复；持续磁盘故障也可能使应急证据无法落盘。pending START、未知字段和未闭合驻留不能填零。原文/提示/journal属于私有产物；公开审查仅保留聚合结果、源码和哈希，不发布题文或逐请求原文。

## 硬资源边界

资格保留1770秒parent、1760秒child和0.5 GPUh预留。计时从模型装载前启动；child在preexec与main入口安装同一绝对monotonic deadline及PDEATHSIG。parent仅清理自己创建的session/process group；不停止其他任务。选卡检查真实compute PID并只选permitted空闲卡，最少32000 MiB空余。原guard字节与此前已独立Linux验证的版本相同；本次没有自行SSH、启动GPU或重跑Linux系统调用测试。

resident GPU费用须以parent CLOSED回执为准，不能把CUDA事件或条件吞吐替代占用。若parent硬终止，未闭合账仍需另行核查。单个合成长度的吞吐外推不是384题的资源准入证明。

## 实际验证与局限

实际重跑作者21项测试，包括完整JSON child加载mock、174请求、全部输出落盘与费用/raw审核；另写4项独立边界测试：

1. RETURNED已fsync后再抛错，原文仍可读，native只一次，wrapper拒绝继续，异常提取不产生费用行。
2. 两个模型共享一个请求cap，第二模型在native前被拒绝，不增加物理调用。
3. admission callback修改自己的context副本，不能通过别名改调用者或持久化context。
4. callback抛出含overflow文字的ValueError，仍被归为基础设施错误，不能误用输入拒绝继续路径。

共25项，零失败/错误/跳过。没有导入torch或transformers、读取真实数据、模型权重或运行GPU。详见 `audit_001/RECEIPT.json` 与 `CPU_TESTS.log`。早期静态意见及作者的prelaunch失败/修复记录保留；没有覆写科研历史结果。

优势是逐请求原文、完整逻辑费用和物理驻留分离且可追溯，失败不被静默成功化。风险是CPU mock不能证明BF16模型、真实tokenizer、长输入、磁盘负载及实际硬件上的重放表现；这些仍由下一次有上限的真实工程资格检验。置信度：对所审字节版本的CPU控制流和持久化边界为高；对未执行的真实资格不作通过断言。
