# 共享逐请求持久化辅助层

仅 CPU 验证，尚未真实模型资格。`DurableBackend(underlying, model_name, journal_path, context_provider=None, before_request=None, require_fresh_journal=True)` 要求独占 backend 和全新私有 journal。`before_request(context)` 在 START 之前执行，供调用上限准入；metadata callback 必须纯观察，不得改变提示/seed/config。

配置底层 HFBackend 的 `physical_log=None`，内存 physical_calls 仍为原始费用证据。START fsync 后才生成；RETURNED 原文+Generation+native 增量 fsync 后才返回原 Generation 对象；ERROR 记录已知 native 事件，未知原文为 null。journal 不向 physical_calls 添加费用行。

`extract_durable_error_evidence(error)` 沿 `__cause__` 提取包裹异常中仅保留在 RAM 的 Generation，供调用者尝试私有 emergency checkpoint。持久磁盘失败/硬杀可能阻止恢复，pending START 为未知，不是零费用。写入失败使包装层拒绝后续请求。原 backend 输入拒绝在 ERROR 成功落盘时保持原异常，以保留冻结矩阵的分类；包装层写入或 callback 异常不暴露可被误认 overflow 的 cause。

`read_journal(path)` 严格检查成对状态及尾行完整；不支持 resume，也不会修复或丢弃损坏行。身份/tokenizer/参数/Generation 对象透明；不操纵推理随机数。辅助层不做计费结论，费用由原 v2 reconciliation 与外层审计器处理。
