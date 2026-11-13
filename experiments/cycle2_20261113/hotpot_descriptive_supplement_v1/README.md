# 已解封描述性补充：11 条件格式、结束原因与费用

状态：真实单次流式聚合完成，待 root 审阅。未修改 scorer、adapter、主统计、科学门、原归档或 Git。没有读取 gold/vault、INPUT、questions 或原题库；MATRIX 必然含上下文和生成文本，但程序只在内存中访问格式/费用所需结构，不打印、不复制文本，不生成逐题派生文件。

## 产物

`run_001/public/` 是唯一聚合发布候选白名单，7 文件：

- `REPORT.md`、`SUMMARY.json`；
- `aggregate_format.csv`：每条件严格 aggregate 格式成功/失败/输入拒绝；
- `finish_reasons.csv`：完整逻辑与物理返回两个口径，条件×模型×worker/aggregate 的 eos/length/unknown；
- `logical_costs_all_conditions.csv`：全部 11 条件×两模型的完整逻辑服务费用；
- `physical_returned_by_model.csv`：实际返回请求的两模型物理计数/token/时间/峰值；
- `residence_components.csv`：四 worker 的驻留、可观测装载区段、backend elapsed 和未分解余额。

`private/` 与 `run_001/private/` 不发布，保存路径映射、读取前源码固定记录、测试记录、输入 ancestry/pins、运行回执和公开文件 SHA。未生成任何逐题记录。当前未执行发布。

## 方法与核对

读取前范围见 `DESCRIPTIVE_SCOPE.md`。`frozen_dependencies/evaluation_adapter.py` 与 `hotpot_support.py` 是原字节复制，SHA 保持冻结值。实际执行调用 adapter 原 `_records` / `_node_costs`；不调用 scorer 的答案指标、统计 helper 或前沿函数。

格式统计只复核原 scorer 的 bool/三字符串结构谓词，读取原 `format_ok`，不重新解析输出文本或传入虚构 gold。人工测试对格式失败路径调用原函数确认，同时将 `answer_metrics` 设为禁止调用的测试 trap；这仅发生于内存人工模块，冻结源码未改。真实路径从未调用任何答案指标。

按 review 的 384 个固定 index 顺序流式处理，一次只持有一个 MATRIX。每个文件解析前核 review 与原 RAW_OUTPUT_MANIFEST 一致 SHA。实际运行完成 384/4224：

- 全部 11 条件的格式失败合计 59，与原评价全局计数精确相同。
- S1 的 B、NO_EDGE、INDEPENDENT12、SINGLE_M 两模型四主费用字段逐项精确相等。
- 完整逻辑 50304 calls、物理返回 33408 calls，与外部 audit 核对一致；reused 16896 与差额一致。
- 11 条件 aggregate 全部记录为 eos；它只描述原结束原因，不能推断格式必然成功。格式失败仍完整保留。
- 实际正式驻留 32.57857300557617 GPUh。qualification 0.1783492729746892 GPUh 已收费，仅单列；当前候选累计 32.75692227855085 GPUh，不再收费。
- 8 个装载区段 wall 合计 122.783609443577 秒；驻留减装载/backend elapsed 的余额约 2638.383146 秒。余额不是精确等待或 CPU 时间。

费用检查浮点策略：原四方法按相同题序、节点序与加法路径要求精确相等；跨条件汇总与 audit 因求和顺序不同，以 rel_tol=1e-12、abs_tol=1e-8 核对。这是数值序列化校验，不是科学容差或新阈值。峰值分配取 max，不相加。

真实 CPU 流式聚合耗时 10.594 秒；没有测量本进程峰值 RSS，不作具体内存峰值声明。5 个小人工测试全部通过，unittest 耗时 0.047 秒；测试只涉及聚合、复用、原函数费用一致、格式结构、不完整完整费用保持 unknown 和重复物理 identity 拒绝，没有重复冻结主统计测试。

## 使用与限制

```text
python -B supplement.py
python -B -m unittest test_supplement -v
python -B supplement.py --run-authorized-real --private-pathmap private/PATHMAP.json --output NEW_DIRECTORY
```

第一条默认不读数据。最后一条仅说明本次已授权执行的复现接口；本轮只执行过一次真实 stream，输出目录不允许覆盖。所有真实路径位于私有 pathmap，源码不含 E 盘硬路径。输入 SHA 是 root 提供的特定已解封快照，不可自行换成别的实验。来源固定与结果 SHA 见 `DELIVERY_RECEIPT.json` 和私有回执。

优势是逐条件补齐原评价未导出的解释字段，并保留逻辑、物理、驻留三种费用口径。限制是不能从 eos/length、格式失败或费用差额定位质量变化的因果机制；精确 barrier 等待、GPU 初始化内核与独立失败费用仍 unknown。并行 worker 驻留相加，不把并行 wall 当 GPUh，也不把逻辑复用差额当实测驻留节省。

原主要科学结论完全不变；B 路线失败不等于项目停止。该补充不新增科学检验，不自动授予发布、标签或 GPU 准入。
