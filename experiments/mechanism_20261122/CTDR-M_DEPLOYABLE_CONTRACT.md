# CTDR-M 部署合同

CTDR-M 将受影响 relay 行定位为图编辑后信用证书维护的输入。它不读取答案标签、fresh audit table 或事后决策集合。

## 定位规则

旧图和新图在推理前登记每条 relay 行的 interface signature。签名包括 prompt 模板、上游消息键集合、模型和 tokenizer 哈希、解码参数及随机策略。图编辑器另外登记由 edge diff 和下游可达关系得到的 `graph_affected_rows`。检测器把 signature 变化与图影响行取并集。任一字段缺失或格式不确定时回退所有 relay 行，允许多复检但不允许无依据跳过。

## 公平比较

Stale、Warm/Reweighted、Source-TV、Random、Periodic、Max-Uncertainty 和 Full Restart 共享旧 transcript、新图、阈值、模型输出与增量 token 口径。CTDR-M 只使用冻结 graph manifest 和 interface signatures。Oracle row mask 仅用于诊断定位器额外成本，不进入主胜负。

## 证伪面板

1. source-only edge edit，relay interface 不变。
2. 单行和双行 conditional relay drift。
3. 四行 drift，检验稀疏编辑边界。
4. 语义相同但格式或 token 顺序变化，要求保守复检。
5. 缺失 manifest 字段，要求回退全重查。

若图编辑传播到未直接列出的 relay 行，必须通过下游可达关系或该行的上游消息签名变化进入并集。漏标的传播行属于失败样本，报告定位召回和 full-restart fallback 比例，不把 oracle mask 当作主方法输入。

主指标是 exact decision-set mismatch、误删、误留、增量 token、padding token 和 wall-clock。按项目约定不运行置信区间或 bootstrap。

## 预注册门

- source-only 面板 exact-set mismatch 为 0。
- conditional drift 面板 false-skip 为 0。
- 单行或双行 drift 相对冷重启至少节省 20% token。
- 四行 drift 相对冷重启的额外 token 不超过 10%。
- 相对 oracle row mask 的额外 token 不超过 10%。

若 source-only 通过而 conditional drift 失败，冻结 CTDR-M。若四行 drift 的效率门失败，结论限定在稀疏图编辑下的证书维护。
