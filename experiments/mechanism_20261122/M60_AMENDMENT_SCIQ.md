# M60 正式面板扩展说明

原合同规定 QASC 或 ARC 至少 192 题。为检验真实上下文中的图编辑机制，正式正向验证使用两个 QASC 和两个 SciQ 面板。SciQ 面板字段含真实 support 段落，source edge deletion 删除 support 注入，answer key 仍封存。所有 M60 主门、预算、基线和评分规则保持不变。

ARC 面板没有 support 字段。其 source 和 edited prompts 相同，因此只作为 null control 保留，不纳入正向机制聚合，也不用于候选选择。

该说明是任务范围内的数据覆盖扩展，不修改原有门槛，不使用封存标签进行算法调参。

在本次 source-edge edit 的 worker 中，conditional relay prompt 与 old/new source 状态相同，因此 $q_i^{1'}=q_i^1$，原合同 transport bound 中的 conditional-channel TV 项为零。论文将该正式候选明确限定为这一可验证子情形，没有把 source-only 结果外推到 conditional-channel drift。
