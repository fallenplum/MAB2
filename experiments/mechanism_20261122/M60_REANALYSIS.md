# M60 重新分析

## 当前审计结论

原始 M60 报告的约 80% 相对冷重启节省不能归因于 CTDR 漂移半径或决策区域排序。公开代码中，旧 transcript 已被写入 `Audit(new_source, ...)`，因此历史观测复用和新来源分布重加权先于 radius 检查发生。`safe_transport` 在当前四面板上始终与 `Audit.decision()` 已可判定同时发生，不能比普通审计更早停止。

原始 stale 的查询数和 token 数包含编辑前历史成本。审计周期将 stale 的 post-edit 增量固定为 0，并将历史成本、post-edit 查询成本和 full restart 增量分开。

## 接口核验

正式 worker 字节已恢复为既有形式摘要使用的 SHA-256。七个概率行解释为三个 source 行加四个 relay 行 `[none,A,B,C]`，四个概率列始终对应假设 A-D。`m60_schema.py` 在 scorer 入口验证 23 个字段、概率归一化、relay 行数量和 4×4 终端矩阵。

## 当前可支持的主张

在固定四行 relay 接口和有限审计器内，复用旧 transcript 并按新 source 重加权显著低于完全冷重启的新增反馈成本。真实 token 成本排序若有收益，只能作为小幅调度改进。漂移半径的独立收益尚未被证明。

## 未完成的研究问题

- Full CTDR 是否在四面板之外的条件 relay 通道漂移下减少错误删留或新增 token。
- 图编辑造成 relay 条件变化时，source-only TV 半径是否失效。
- 在真实任务质量而非有限接口决策集合上，信用维护是否改善最终回答。

在这些问题完成前，本周期不生成 CTDR 独立算法贡献的论文结论。
