# E0 真实工作流适用性与收益空间验证

## 1. 一句话结论

E0 暂不能进入真实工作流实验：已完成 D/E 盘资产清点和固定阈值 CPU 证书验证，但没有核验到规约要求的 FActScore 主任务资产，因此状态为 `INCONCLUSIVE_BLOCKED_ASSET`，不是正结果，也不是 E1 授权。

## 2. 规约、谱系与资源

本轮使用 E0 根 prompt 作为唯一规约，并保留 P4 的 `G0_INCONCLUSIVE_INVALIDATING_ISSUES`。E 盘发现 Qasper、LongBench v2、HotpotQA、Qwen2.5-1.5B/3B 等资产。Qasper 和 LongBench 不是当前冻结的 FActScore 原子事实核查合同，HotpotQA 被 E0 明确排除，因此没有替代主任务。服务器只读探测在当前执行环境被拒绝，未启动远端 GPU。

## 3. F 与 D 主表

没有合法的 32 个实体/文档组、旧/新版本和独立知识源，F/D 的完整重跑、缓存、规范化缓存、早停和 stale 对照均未运行。任何成本或质量数字都会混合不匹配任务，故不报告。

## 4. 接口、传播与证书

已实现独立的固定阈值证书 CPU 核心，区分 `INVALID`、`CERTIFIED`、`UNRESOLVED` 和 `FULL_RESOLVED`，对空任务、三态穷举、阈值边界和非法标签进行了测试。该测试只验证实现边界，不提供真实工作流证据。

## 5. 剩余收益空间与下一步

当前没有固定 DAG、真实旧/新轨迹或合法成本账本，不能计算强基线后的 headroom。下一步只能是获取官方 FActScore 人工标注/知识源资产，或由研究者明确批准替代协议。完成前不得启动 E1、DGBR 或新的算法搜索。

## 6. 证据分类

- `DEVELOPMENT`: 固定阈值证书 CPU 单元/穷举测试。
- `REPLAY`: 无。
- `BLOCKED`: FActScore 主任务资产、远端只读探测和 E0 新计算预算。
- `INVALID`: 无新的 E0 结果；历史 P4/P3 负结果保持原状态。

