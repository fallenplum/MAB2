# R1 纯数组策略与探索统计

只实现冻结协议的数值算子，不读取实际题目、答案、模型或磁盘数据。协议 SHA256 固定为 `5336eacc6713d4e444ddb631032409cd2718ad3b4aaee1387aff5baac40f5ecf`。所有测试使用人工数组。测试通过不代表方法有效或服务器执行已通过资格。

## 适配顺序

1. 从冻结协议的原始字节调用 `specs_from_protocol_bytes(raw)`，同时核对协议 SHA 和独立构建的八个图模板。
2. 只释放 I 的评分。每 episode 提供 `(12,6)` 的动作差，列按 `spec.editable` 的**实际边索引升序**。执行器菜单名与顺序可能保留置换前模板编号，必须用 `request.edge` 对齐；可调用 `align_initial_columns(by_physical_edge,spec)`。
3. `freeze_purchases(I_arrays,specs)` 返回不可变购买计划。把 `jsonable_freeze(plan)` 原子持久化，再进入 C。
4. 调用 `update_and_freeze_actions(plan, read_C, persisted_purchase_digest=...)`。C 回调签名为 `(ReplicaPlan,physical_edge)`，仅返回该副本实际购买的一条边在 C12 的成对动作差。返回的值只能为 −1/0/1；未买格保留 I 值及 source_A 来源。将返回动作 freeze 原子持久化，再开放 S。
5. 调用 `grade_service(actions,read_S,persisted_action_digest=...)`。S 回调签名为 `(episode,condition,action)`，返回24个0/1分数；`action=None` 表示 stay。condition 的内核名称为 `change` / `no_change`，分别对应 B / A 服务背景。S 数据只进入评估器，不能更新已冻结动作。
6. `summarize(evaluation)` 给出主比较、全部预定方法、两个 k、辅助比较及变化与不变化之差。所有区间只重采样八个 episode，固定10000次、seed69173；仅探索，不保证八簇覆盖精度。

回调和 SHA 绑定提供接口检查，**不是操作系统安全隔离**。适配器仍必须守住阶段文件权限、题目去重、问题/标签分离、输出哈希链、评分及真实成本。传入 `persisted_*_digest` 只验证绑定，不能自动证明调用者确实已落盘；持久化责任在适配器。

## 随机参考与成本

`uniform_pair_expectation` 对所有 `C(6,k)` 子集等权枚举。每个副本从相同原 I 表独立开始，只用自己的 C 格替换，分别冻结动作后评价；不能先把全部 C 平均或挑最好子集。副本不是独立数据点；统计单位仍为八个 episode、192道服务题。

`conservative_budget(k,method)` 提供每 episode-condition 的逻辑完整上限。K2主动策略为5,210,112 token、144次评分；C反事实增量为48次新调用。随机参考的逻辑上限是每个等权副本的上限，不能乘15。全菜单物理预计算费用由研究账本单独承担。

Source-only没有C费用；current-stay是无需I的额外实用图参照，其初始化权限与机制组不同。Full-revalidation花更高费用，不能拿来声称同低预算优越。k1/k2中重复列出的 source/stay/full 是相同策略在两组展示，不算新实验。实际 token、评分CPU、GPU占用和缓存费用由适配器记录，内核不会把未知费用填0或把预算上限当实际开销。

## 核验

运行 `python -m unittest -v test_r1_policy`。使用 `python RUN_SYNTHETIC_TESTS.py --output receipt_name.json` 可生成有计时与源码哈希的测试凭据；已有同名凭据拒绝覆盖。

测试覆盖：dirty与no-change算子恒等、严格正阈值及实际边索引并列、只访问已购买C、随机枚举的独立副本、未购买格污染反例、角色与增删平衡、置换菜单列对齐、动作冻结后S隔离、episode bootstrap的独立标量核对、完整费用算术。原始数组及冻结计划属于私有派生产物，公开只使用聚合结果。
