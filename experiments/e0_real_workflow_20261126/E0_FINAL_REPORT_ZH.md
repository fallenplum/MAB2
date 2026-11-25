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


## 7. 32 组开发运行结果

完整锁定运行覆盖 32 组，30 组具有可配对的旧/新固定事实分解，2 组因新版本缺少冻结事实分解被保留为 `INVALID`。旧版本产生 777 个事实事件，新版本产生 1073 个事实事件，warm cache 命中 70 个事件，占新事件的 6.52%。warm 相对新版本冷启动减少 7.28% 输入 token，减少 4.13% verifier latency。后置可比事实准确率为旧 69.79%、新冷启动 68.11%，warm 与新冷启动相同。有效组中只有 1 组最终 PASS/FAIL/UNRESOLVED 动作发生变化。

这些结果是开发数据上的机制诊断。当前工作流表现为按主题检索后逐事实扇出，没有建立非平凡共享动态祖先，因此 E0-C 不把它冒充深层图传播。强 flat、stale、canonical cache、strict gate 和独立 headroom 仍待完成。

## 8. Strict-gate 基线

在同一 32 组开发数据上实际运行了 cold strict-gate 和 canonical-cache strict-gate。cold strict-gate 只产生 30 个有效组事件、33861 个输入 token，canonical strict-gate 产生 18448 个输入 token。相对新版本冷启动的 1259696 个输入 token，节省分别为 97.31% 和 98.54%。但相对人工标签构造的固定 D 目标，两个 strict-gate 方法各有 17/30 个有效组动作错误，而新版本完整冷启动为 7/30 个错误。

这说明证书早停的成本收益伴随明显的动作风险。当前不能把节省数字解释成有效算法收益，也不能把 replay-only 的 stale 或 canonical full 结果当作独立在线结果。强 flat/RAG 和完整 stale 质量对照仍需补齐。

## 9. E0 方向判断

规范化缓存回放在 30 个有效组上保持与完整冷启动相同的动作，同时减少约 7.28% 输入 token。cold strict-gate 和 canonical strict-gate 分别减少约 97.31% 和 98.54% 输入 token，但各有 17/30 个动作错误，完整冷启动为 7/30。group-level flat RAG 基线在 30 个可比较组上全部动作错误。当前工作流没有非平凡共享动态祖先，E0-C 不支持图传播收益主张。

因此 E0 暂不支持正向图优化结论。该结果保留为开发性负机制证据，不能自动进入 E1，也不能声称 ACL 结果。

