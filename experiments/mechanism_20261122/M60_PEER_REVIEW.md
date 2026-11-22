# M60 独立审稿检查

## 支持的主张

在固定有限接口中，CTDR 能够在 source-edge 图编辑后使用可审计的 transport interval 保留安全决策，并用预算化反馈完成未决的 decision regions。四个 disjoint input panels、12 个 panel-team cells、四类复检对照与 200 个 synthetic soundness seeds 构成中高强度的机制证据。

## 必须限定的主张

1. `action error` 只表示 sealed correct answer 的 membership decision 错误。`exact set` 表示完整有限决策集合相等。两者都不是模型任务准确率。
2. `D*` 是 fresh finite-interface audit reference，不是人类真值或真实任务最优。
3. 79.6%/80.1% 是增量 logical-token accounting 的节省，旧 transcript 已支付，不应表述为 wall-clock 或厂商费用加速。
4. 证明和实验只覆盖 source-edge edit 与固定 conditional relay channel。不能泛化到任意图编辑。
5. 四面板来自两个任务数据源，三团队共享面板题目。不能称为 2304 个独立问题或三任务普适性。

## 文献与复现检查

ACL Anthology 元数据已按官方页面修正。公开分支包含 CTDR、worker、输入 receipt、baseline replay、正式汇总、消融、复现说明和编译 PDF。sealed keys、原始 traces、weights、私有题目数组和 credentials 仍保存在私有审计目录。

## 审稿结论

证据足以支持“有限 graph-conditional credit interface 中的 certificate transport 机制”这一窄主张。证据不足以支持任务准确率提升、公开端到端系统 SOTA、通用图编辑或真实部署墙钟加速。投稿时应保持该范围，避免把机制证据写成系统性能或预测器性能结论。
