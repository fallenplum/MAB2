# 已解封后的固定描述性补充范围

这是 root 正式解封后的工程描述补充，不是预注册新统计。已知当前 B 路线有失败结果，但本补充不读取 gold、不重算 EM/F1、主 bootstrap 或证据门，也不挑题、种子、边或方法。沿用原 POSTCLOSURE_REPORTING_PLAN 的 11 条件范围。

在首次读取真实 MATRIX 前固定以下处理：

1. 先核 root 提供的 score receipt、scoreReview、evaluation、外部 AUDIT、原 RAW_OUTPUT_MANIFEST 和 ROOT_COST_CLOSURE 精确 SHA。每个 MATRIX 在解析前按 review 与原 manifest 双重核 SHA。按 selection index 0 至 383，每次只持有一个 MATRIX，结束后释放。不读取 INPUT、questions、vault、原 train/val/test、durable 原文或模型。
2. 严格 aggregate 格式检查沿用冻结 scorer 的结构谓词：`message_metadata.format_ok` 为 bool，`parsed` 恰为三个字符串字段；格式结果是该原 bool。因原评分函数成功路径会读取 gold，此处不调用答案指标，只复核上述格式字段，不重新解析原文。源码 SHA 绑定且文件不改。
3. 按条件、模型、worker/aggregate 角色计 `generation.finish_reason`。固定 eos、length、unknown；缺失或未知值均归 unknown，不由 token cap 猜测。完整逻辑口径包含同一生成的逻辑复用，物理 request_trace 口径单列。
4. 完整条件逻辑节点及主要费用调用冻结 adapter `_records`、`_node_costs` 原函数。所有 11 条件完整费用按模型汇总；复用逻辑节点仍计费。四方法 S1 的四字段与已冻结 evaluation 逐项精确一致。输入拒绝完整费用 unknown，不能拿 partial 冒充完整费用。
5. 物理总量引用已核外部 audit，并对 MATRIX request_trace 返回代次的 calls/tokens/CUDA/elapsed 做描述性核对。可复用逻辑调用数只作逻辑与物理次数差，不声称测得驻留节约。物理账本与 parent monotonic 驻留分别报告。
6. 模型装载观测值来自 8 份 MODEL_LOAD_COMPLETE，属于装载区段 wall（包括该源码范围内初始化及相关开销），不拆成精确 GPU 初始化。逐请求 backend elapsed 由原 generation 记录；驻留减去这些观测区段及装载后的差额只列未分解余额，不命名精确等待/CPU 时间，不把 CUDA 差额当并行开销。
7. 原 qualification 已付 0.1783492729746892 GPUh 单列，不重新入账；global/candidate 总量原样引用 root 闭账。不可识别的 barrier等待、精确初始化内核、失败分项费用为 unknown。
8. 仅输出聚合报告、CSV、JSON；无逐题文件、qid、prompt、answer、来源 hash 或私有路径。来源及验证 pins 只写 private receipt/pathmap。先用小人工 fixture 验证，再只执行一次真实流式聚合；若失败保留失败回执，不改原数据或分数。

本补充不自动授权发布或 GPU，不把 B 失败解释为项目终止。优势是补齐原 adapter 未导出的解释性字段并校核费用口径；风险是描述计数不能定位因果机制，余额也不能区分各类系统等待。
