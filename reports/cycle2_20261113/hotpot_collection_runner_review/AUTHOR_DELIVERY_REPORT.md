# Hotpot collection runner：可审实现交付

**结论：所要求的 staging 编排实现已完成，独立 CPU 审查无剩余代码阻断；仍未冻结、未分配 GPU、未运行真实采集。**

交付版本是 REVIEW_SNAPSHOT_002.json；其前一版原始字节保存在 review_snapshot_001_source。原矩阵、共享 durable helper、图/独立12算子、科学协议和历史结果未改；本包写入范围仅 Cstage。

作者 CPU 结果为 26 项中的 24 PASS、2 Linux 专属明确 skip，零失败；独立代理重跑同一字节并加 4 个边界，30 项中的 28 PASS、2 同样 skip，零失败，源码 hash 前后相等。独立检查覆盖错误放行阻止第5个 worker-local 单元、费用 profile 不带原文、隐藏 overflow 拒绝及终态顺序改变拒绝。Windows 沙箱初次 hardlink/信号测试桩失败完整保留为历史失败摘要，没有把 skip 算成 PASS。源码可移植入口为 run_cpu_tests.py；真实 Linux guard 结果由 root 另行执行、单独归档，本作者交付尚不将其计作通过。

可复核的实现包括：固定 384 原序索引与 4 边各96；双方服务器 tokenizer 身份/完整 payload 绑定的 8×48 盲前缀；8题完整付费后才冻结 +25% 费用预测；不按效果访问/筛选；逐 worker 模型加载、屏障空闲和退出延迟计入 monotonic residence 总和；当前轮资格费用唯一事务归因；全局/每 worker cap；原子不可覆盖放行；自有进程组终止；只读 disk guard；逐请求 durable raw Generation/native events；异常 cause 中已返回原文的额外 checkpoint；4224唯一终态、原文 hash、初见 receipt 不可改写和独立评分释放边界。

资格验收必须由 root 单独签字并绑定实际 qualification/identity/residence/plan/source_manifest/root_closed_review 字节；REVIEW_REQUIRED 本身不能授准入。科学协议必需 pin 缺失、配置与既有默认算子不符、预算未授、未知/矛盾前缀费用或 overflow 都拒绝运行/放行。JSON list 角色配置仅做 canonical 校验，实际实例仍为 GraphConfig(rounds=3)。

优势是错误/费用/原文证据的边界可独立重放，且不以 cap 伪造实际支出。风险是 8 题预测只作工程外推，不能保证未来尾部耗时；prefix overflow 会使本预测不支持，未来 overflow 可能失去复用；磁盘持续失效或 SIGKILL 后未返回原文可能不可恢复；未关闭驻留保持未知。完整正式收集和质量/机制结论均无新证据。

对本次人工编排逻辑的验证置信度为高；对真实服务器与384题资源可行性尚无本包实测结论。后续需要 root 的 Linux guard/资格闭账与独立资源和科学准入；不得把本报告用于替代这些步骤。
