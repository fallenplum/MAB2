# 历史分阶段反馈：公开代码与汇总结果

**结论只适用于冷启动、均匀 probe、两种响应层、独立 Poisson completion 与声明的 lazy 基图逻辑成本模型。** 评分等价成本比 0 和 0.1 时，最优退化为全部完成；不能泛化为所有公平成本情境无优势。评分比 1 的微弱 oracle 反例完整保留。历史分析不证明未来服务收益或方法新颖性。

这是事后 historical optimistic-oracle screen。公开包提供代码、协议、汇总前沿、数学说明和复现核验；不含原题、答案、逐题/逐响应数组、权重或私人存储路径。私有输入不随包发布，因此公众可检查汇总与方法，完整数据重放需要已有授权输入，不能声称无需数据访问即可端到端复现。

## 接口修订发生在历史结果之后

本次只将数据路径与输出路径参数化。数学实现、冻结协议和科学结果未修改；源码/协议从脚本所在目录读取，结果写入指定的全新目录，已有输出目录会被拒绝。历史 `run_screen.py` 的源码哈希与参数化版本哈希映射见 SOURCE_HASH_MAP.json。该修订不是事前注册，也不是重新选取分析或阈值。

历史结果保存在 HISTORICAL_RESULTS.json 与 HISTORICAL_FRONTIER.csv。原报告中的 RESULTS.json / FRONTIER.csv 指这两份历史文件；新重放目录仍使用原文件名。

## 授权数据重放

已验证环境：Python 3.11.14、NumPy 1.26.4、SciPy 1.11.4，纯 CPU，不使用模型或 GPU。安装依赖后，从任意工作目录运行：

```sh
python run_screen.py --diagnostic-dir /authorized/diagnostic --d2-inputs /authorized/inputs.npz --output replay_new
python verify_replay.py --replay-dir replay_new
```

输入 diagnostic 目录需要原 RESULTS.json、PRIVATE_MEASUREMENT_ROWS.npy、HASHED_NODE_CACHE.npz；D2 的完整文件哈希用于核对，数值分析仅访问 train/tune 的映射元数据，不读取 quality_by_seed。原始数据须符合冻结哈希；脚本拒绝来源不符。命令中的路径仅为占位示例，没有提供或授权获取私有数据。

重放比较分三部分：1,248 行前沿要求逐字节相同；所有科学 JSON 字段要求精确相同；计时允许不同，源码 provenance 仅允许 SOURCE_HASH_MAP.json 声明的 run_screen.py 哈希变化，其余输入/源码哈希须相同。验证器还检查输出拒绝覆盖、零/0.1评分退化以及评分比1反例。已执行回执在 REPLAY_VALIDATION.json。

## 文件与证据边界

- REPORT.md：结果与限制；MATHEMATICAL_NOTE.md：指定类充分证书。
- PROTOCOL.json、SCHEMA_AND_COST_ADDENDUM.json：原历史规范，字节保持不变。
- frontier_math.py、run_screen.py、verify_replay.py：计算及复现检查。
- HISTORICAL_RESULTS.json、HISTORICAL_FRONTIER.csv：聚合结果，非逐题数据。
- LITERATURE.md、INDEPENDENT_MATH_REVIEW.md、NEXT_MECHANISM_ONE_PAGE.md：文献、独审边界及未准入建议。
- SOURCE_HASH_MAP.json、REPLAY_VALIDATION.json、PUBLIC_MANIFEST.json：来源映射、复跑与公开白名单。

token 是记录的资源代理，评分费是假设敏感性；没有实测完整 GPU、wall-time 或物理跨图缓存账。所有方差针对固定银行的抽样随机性；期望预算、连续放松和新题总体误差不能混为一谈。Neyman、HT、增广估计及反馈图并非本项目原创。
