# 资格审核成本差异的精确诊断

原失败报告继续保留。对6资格题90图，仅访问成本数值与节点顺序进行诊断，发现38个派生字段不相等，全部为 gpu_seconds，最大绝对差7.105427357601002e-15。token、call、输入消息数、峰值内存等整数成本字段没有任何差异。

存在两个确定因素：canonical JSON 将 aggregate 键放在 r1/r2 前；服务端模型 identity 固定 Python3.11.9，而本次审核运行于Python3.13.3。Python官方文档确认3.12更改了浮点sum算法：[Python sum 文档](https://docs.python.org/3/library/functions.html#sum)。只把节点恢复为执行顺序，继续用本地3.13内置sum，仍有38处差异；同时使用原执行顺序和显式逐项加法后，所有180个物理/逻辑成本字典的全部字段与原始产物精确相同。

修复只在只读审核器：original_execution_cost固定 r1:w0..3、r2:w0..3、aggregate 顺序；producer_float_sum依据固定模型identity中的Python版本复现原求和算法。原node成本、native ledger和所有产物字节不变。没有增加数值容差，也没有免除其他账目检查。新增合成测试专门验证JSON排序和Python3.11算术差异；7项审核测试全部通过。

另名报告 R1_QUALIFICATION_EVIDENCE_AUDIT_002.json 已通过结构、replay、来源和账目检查，但保持REVIEW_REQUIRED。格式仍需root决定：276物理输出中24个原始schema失败，90个aggregator输出中9个schema失败；3个length结束，aggregator length为0。这是资格格式证据，不是回答正确性成绩；没有读取gold或S评分。

冻结v1 runner的restore/checkpoint路径也使用JSON还原顺序与严格cost比较，存在同类恢复边界。当前资格全新生成不受影响。正式新unit首跑不触发还原成本检查；若将来需要恢复，必须在再运行之前显式做兼容性修订、固定新源码/计划来源并保留已付成本，不能盲目重复 --resume 消耗重试次数。

完整资格稳态吞吐预测已将一次性约200秒模型载入单独计入一次重载。每卡最慢整题执行时间包含fsync等开销。总候选预计18.85 GPUh，加25%稳态余量为23.39 GPUh，加50%为27.93 GPUh；然而worker1在25%余量下超过自身8h固定上限。总预算与逐worker硬限必须同时满足，不能只看总和。

后续处理：正式采集前已冻结总额不变的资源份额修订，三进程改为6.0/9.8/8.2 GPU小时，见 protocol/cycle2_20261113/R1_RESOURCE_AMENDMENT.json。上述8小时风险描述对应保留的原v1计划。资格账本在正式追加前另存独立快照；根执行环境Python3.11.14也完成精确复审，7项仓库测试通过。
