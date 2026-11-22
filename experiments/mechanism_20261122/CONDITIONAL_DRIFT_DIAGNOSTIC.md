# 条件 relay 漂移诊断结果

## 结果

在 M60 四个冻结面板的 2304 个题目×团队单元上，对 relay 行做固定的条件通道漂移模拟。该结果是低成本机制诊断，不是新的独立模型确认。行索引按接口 schema 解释，A 为 advised A 行，no-advice 行没有被误标为 A。

| 漂移 | source-only CTDR decision error | TCR-row decision error | TCR-row token | cold token |
|---|---:|---:|---:|---:|
| alpha=0.25, A | 0.0017 | 0 | 140.15 | 490.27 |
| alpha=0.25, A-D | 0.0061 | 0 | 301.46 | 477.26 |
| alpha=0.50, A | 0.0048 | 0 | 137.74 | 485.11 |
| alpha=0.50, A-D | 0.0143 | 0 | 291.13 | 455.64 |

Source-only CTDR 保留 0.4783 的 safe rate，但在条件通道变化时仍有决策集合错误。TCR-row 只重查已知受影响 relay 行，复用合法 terminal labels，在四种模拟条件下恢复完整 decision set。它相对 cold restart 节省约 38%–72% 的 post-edit token，具体百分比随条件变化而变。

## 科学含义

该结果支持一个可证伪的机制假设。图编辑对 relay conditional channel 的影响会使 source-only certificate transport 失效，而行级条件感知重验证可以修复这种失效。它尚不足以证明真实模型上的算法收益，因为条件漂移由冻结向量的固定变换构造，未产生新的模型推理。

下一步必须在新的输入和实际模型输出上生成 old/new relay rows，预先固定受影响行的图编辑规则，并与 Warm/Reweighted、source-only CTDR、TCR 和 cold restart 做相同 transcript 与成本口径的对比。若服务器持续不可达，则保留该诊断作为候选方向证据，不将其升级为 ACL 主会结果。
