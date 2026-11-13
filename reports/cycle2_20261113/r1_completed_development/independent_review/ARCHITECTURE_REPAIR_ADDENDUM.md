# 架构参照诊断修复后的独立只读复核

原 REPORT.md 与 RECEIPT.json 保留。本附录在 root 冻结并运行新 run_002 后追加，不改历史审查。

已只读核对新 MAPPING_VERIFICATION、RESULTS、COMPLETE：其哈希相互一致，并绑定 root 修复冻结 SHA `74037041cbf81310ea2e2b508d0c86dc5424d4396ef00931d6cdeda1ec6aabc0` 与已审入口 SHA `c9d6636a831993cd5b9a366a5087d0e2a5400f6eea5b5410b3cd93b9a858d7c0`。root 独立执行了原 5+新增 6 项 CPU 测试；真实修复回执确认全部 384 个原始 aggregate 用冻结 scorer 重算，16 组 episode×background 完整逻辑费用与明确 S 轴对应。

**修复结果：A_stay=37.5%，682,553 tokens；B_stay=35.9375%，648,577 tokens。** 显式映射为 change→B→axis0、no_change→A→axis1。原 5 个首轮／投票参照未改变；固定 4 个描述性对比按正确 A/B 标签重算。修复新增 CPU 评分 1,344 次，新增模型调用、GPU 和正式轮次均为 0。

新结果 SHA：`e9b45b8ee331d962e386cdb3560bb559850308a221ad22aa15ddf5c6a39da5b1`；映射核验 SHA：`90ae7548c649d5970cfd6c560333caafa7545cf46d049ad269d981db4921c68f`；COMPLETE SHA：`7393ce594af2a2e7021143072836c1d9b515a220c0dae2148b9dbf7edd0e7002`。

结论：run_002 的 A/B 标识诊断修复已有完整核验回执支持；run_001 的错误 A/B 标签仍应隔离留档。该附录只核验 root 运行结果及绑定，没有由本代理再次读取原始模型文本或重复评分。R1 主候选未进阶、R2 G2 NO_GO 和项目 ACTIVE 的科学判据均不变；架构差异仍不可独立归因于通信，不能替代原主门。
