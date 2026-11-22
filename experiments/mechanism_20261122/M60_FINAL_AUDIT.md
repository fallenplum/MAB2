# M60 CTDR 最终科研审计

## 研究结论

在固定的四行有限 credit 接口、source-edge edit、conditional relay channel 不变的前提下，CTDR 的区间证书传输与决策区域复检在正式四面板上恢复了 fresh edited-graph finite audit 的完整决策集合。该结论是机制证据，不是任务准确率、端到端 SOTA 或通用图编辑能力的结论。

## 正式证据

- 正向面板为 QASC-A、QASC-B、SciQ-A、SciQ-B。每个面板 192 题，3 个模型团队，共 2304 个 question-team evaluation。三团队共享每个面板题目，因此 2304 不是 2304 个独立样本。
- CTDR action error 为 0，exact decision-set equality 为 100%。Full restart 是有限接口 reference audit，因为它定义了 `D*`，不等价于任务真值。
- Stale action error 为 3.0816%，max-uncertainty 为 6.9010%，periodic 与 random 均为 6.9444%。
- CTDR 增量逻辑 token 为 98.2057，full restart 为 493.1645。按四面板平均节省 79.6136%，按 pooled ledger 节省 80.0866%。旧 transcript 已在图编辑前支付，因此不计入增量成本。
- CTDR safe-transport rate 为 47.8299%。200 个随机有限接口 synthetic seeds 全部通过 exact decision-set soundness replay，该结果用于实现核验，不作为独立任务证据。

## 公平性与泄漏控制

所有方法使用相同的模型侧 vectors、阈值、问题、团队与 post-edit budget。sealed answer key 只用于执行后评分，不参与 query、stop、panel 或 team 选择。ARC 运行是无 evidence null control，因 source/edited prompt identical 而排除出正向 aggregate。早期重复 key 的 SciQ 输入与不兼容格式的重试运行均不计入正式结果，保留在私有审计账本。

## 预注册门与负结果

CTDR 相对 stale 与 max-uncertainty 的 action-set 门、预算门和成本门在正式 QASC/SciQ 面板上满足。移除 transport margin 后退化为 stale，移除 selective ranking 后的 periodic/random 控制不能达到 CTDR 的决策集合。此前 G2 预测器负结果与 M59/M56 探索结果保持原样，不被合并为 M60 独立确认。

## 适用范围与限制

`rho=||s'-s||_1` 的证明只覆盖 source-edge edit 且 conditional relay channel 固定的接口。conditional-channel drift 需要额外的 row-level total-variation 项。实验使用 logical token ledger，不等价于厂商价格或 wall-clock。论文没有宣称任务准确率提升、端到端公开拓扑 baseline 胜出、通用图编辑或部署加速。

## 预算

全局账本最后核验为 98.1870 GPU-hours used、0 reserved、160 GPU-hours cap。未访问 sealed original tests，未新增付费算力，历史负结果未覆盖。
