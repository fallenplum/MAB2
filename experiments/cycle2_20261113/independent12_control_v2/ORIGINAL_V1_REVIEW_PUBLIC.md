# 独立二次代码审查

**结论：原版本有1项P2计费缺陷，应修复后再进入单独有预算的合成真实模型工程资格。未发现P1。** 本审查由非原实现者完成；不把原作者标为“独立”的自审作为独立证据。

原14项CPU测试与2项异常probe已在只读源、输出重定向到本目录的条件下重放通过。新增6组检查：前序输出扰动不改变后续worker prompt、独立重建208个节点种子、并发拒绝不清除其他活动节点、聚合格式失败仍计13次完整服务但答案无效、缺失/重排/重复槽位拒绝，以及重复物理日志状态计费。

**P2：同一call_id的多状态日志被当作多次费用相加。** 最小复现只有1次backend请求、1个call_id，但SUCCESS后FAILED两条事件使输入914 tokens被报成1828；elapsed也相加。该路径可由冻结HFBackend触发：_record先将SUCCESS加入physical_calls，再写日志；写盘OSError进入generate的except，以同call_id加入FAILED，未再次生成。未知输出仍为null，但已知输入错误地重复计费。应保留全部原始事件，按物理调用身份核对；一致字段计一次，缺失/冲突字段未知，不猜最新或最大值。

其余所查合同成立：worker仅使用原完整题/context与对应round-1提示，无前样本/自状态/同伴输出；聚合保留全部12槽位及格式失败；固定6Qwen+7Mistral；新协议身份与原graph分离、无mask/base/cache入口；沿用question/node种子流。种子相同只是共同随机数标识，不证明不同提示生成相同或无条件独立。sender归一化和receiver计数方向正确；received-message token是wire内容之和，不是完整聚合模板费用。

优势是信息边界和无重试状态清晰。风险是mock不验证真实tokenizer、12候选输入长度、格式率、实际回放或资源行为。后续工程资格必须另冻synthetic输入、调用/重复次数、真实backend环境、预算与resident-GPU占用监督；加载/失败计费，完整prompt不得截断。任何工程通过均不等于算法科学准入。

原源码SHA：59349b801dda9419cbbe779e2ed3b8dcedbe9ca08e2408fe0253424f2a159892。原树前后哈希一致。零GPU、模型加载、benchmark/R1读取。详细复现见RECEIPT.json和review_checks.py。
