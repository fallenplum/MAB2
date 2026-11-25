# E0 真实工作流适用性与收益空间验证

## 1. 一句话结论

官方 FActScore 源码、人工标注数据和 32 个开发组已经下载并锁定。完整 E0 仍暂不能运行，因为官方 `enwiki-20230401.db` 知识源约 20GB，E 盘没有副本，服务器只读访问被当前执行环境拦截。当前状态为 `BLOCKED_KNOWLEDGE_SOURCE_INCONCLUSIVE`，不是正结果，也不是 E1 授权。

## 2. 规约、谱系与资源

本轮继续使用 E0 根 Prompt，并保留 P4 的 `G0_INCONCLUSIVE_INVALIDATING_ISSUES`。官方 `data.zip` 和 `demos.zip` 已通过 ZIP 校验，官方源码已下载。使用 `InstructGPT.jsonl` 的 183 个实体主题，按固定 salt 的主题哈希排序锁定 32 个开发组。主题清单只保存匿名 group_id、主题哈希、源行号和长度，不包含人工标签。

Qasper、LongBench v2、HotpotQA、QASC、SciQ 和历史 M60/P3 轨迹仍不作为替代任务。Qwen2.5-1.5B/3B 只作为待锁定模型资产，不提前选择模型结果。

## 3. F 与 D 主表

FActScore 人工标注数据已经具备，但完整 F/D 服务还需要官方知识源数据库。当前没有运行冷重算、精确缓存、规范化缓存、早停、stale 或强 flat baseline，也没有生成模型调用成本。任何质量或成本数字都会是不完整口径，因此暂不报告。

## 4. 接口、传播与证书

固定阈值证书 CPU 核心已经通过 4/4 测试，区分 `INVALID`、`CERTIFIED`、`UNRESOLVED` 和 `FULL_RESOLVED`。这只验证证书边界，不提供真实工作流结果。32 组已锁定，但在知识源可用前不启动模型调用。

## 5. 剩余收益空间与下一步

当前没有完整 DAG、真实旧/新轨迹或合法成本账本，不能计算 headroom。最小下一步是让 `enwiki-20230401.db` 在本地或授权服务器可读，然后先执行前 4 组 smoke，再按锁定的 32 组运行 F/D 对照。不得读取封存标签替代知识源，不得将旧任务结果并入 E0。

## 6. 证据分类

- `DEVELOPMENT`: 固定阈值证书 CPU 测试、官方数据包完整性、32 组元数据选择。
- `REPLAY`: 无。
- `BLOCKED`: 官方 20GB 知识源和服务器只读访问。
- `INVALID`: 无新的 E0 结果；历史 P4/P3 负结果保持原状态。


