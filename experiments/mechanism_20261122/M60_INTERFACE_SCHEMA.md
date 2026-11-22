# M60 worker 接口与行语义（审计版）

本文件固定记录 M60 正式 `m60_mcqa_worker.py` 的输入、输出行和 token 账本。它用于离线 replay 与代码身份核对，不改变已封存的 M60 结果。

## 实现身份

- worker SHA-256：`6360F9E870CADB844B6A6741FA12FBAF63A4682E80ACFEB619B918E1D0149230`
- `ctdr.py` SHA-256（本分支修复导入后）：`6AEDC222A597D338FEB27B9AD82EE41AE0113FD43A82695AE927626E97CEFE7E`
- worker 使用 `groups=4`，`source_ids` 是 A、B、C、D 四个输出类别 token。
- 每道题固定生成 23 个请求，3 个 source 行、4 个 relay 行、16 个 terminal 行。

## 概率行布局

| 行号 | 类型 | prompt 条件 | 输出含义 |
|---:|---|---|---|
| 0 | source | `first_fact` | 首个证据条件下对四个 grouped hypotheses 的概率 |
| 1 | source | 无证据 | 空条件基线对四个 grouped hypotheses 的概率 |
| 2 | source | `control_fact` | 控制证据条件下对四个 grouped hypotheses 的概率 |
| 3 | relay | `second_fact`，无第一研究者建议 | 中继条件的四类输出概率 |
| 4 | relay | `second_fact`，第一研究者建议 A | 中继条件的四类输出概率 |
| 5 | relay | `second_fact`，第一研究者建议 B | 中继条件的四类输出概率 |
| 6 | relay | `second_fact`，第一研究者建议 C | 中继条件的四类输出概率 |

`full["0"]` 至 `full["6"]` 是上表的七个概率向量，每个向量长度为 4。`tokens[0]` 至 `tokens[6]` 与这些行一一对应。

## terminal 行布局

terminal 行从 7 开始，使用 `itertools.product(range(4), repeat=2)` 的行优先顺序。对第一研究者建议 `i` 和第二研究者建议 `j`，行号为

```text
row = 7 + 4*i + j,  i,j ∈ {0,1,2,3}
```

`full[str(row)]` 是在该建议对下的最终选项标签整数，`tokens[row]` 是对应请求的 prompt token 长度。完整 16 个 terminal 单元构成 4×4 动作表。

## 四行 relay 语义的边界

当前正式 worker 的四个 relay 行是“无建议、A、B、C”四种条件。虽然每个概率向量的输出类别包含 A、B、C、D 四个 grouped hypotheses，但 worker **没有生成第一研究者建议 D 的第五个 relay 行**。因此，若理论或后续论文将 relay 行定义为 A、B、C、D 四个建议条件，则该理论接口与正式 worker 不一致，不能用当前封存结果支持该更强定义。

本分支不擅自补生成 D 行，也不重算历史结果。后续实验若要研究 D 条件，必须新建独立合同、增加该请求并重新支付模型推理成本。

## 成本与 replay 约定

- `tokens` 是逻辑 prompt token 数，不是厂商计费、GPU 时间或 wall-clock。
- 离线四面板消融把同一历史 transcript 作为已支付观测，所有方法从同一 `full`、`tokens` 和旧动作历史开始。
- post-edit 增量成本仅按新图阶段实际查询的 row/cell token 累加。历史 token 不在增量成本中重复计费。
- `M60_MODULE_ABLATION.json` 和 `m60_module_ablation.py` 记录 Warm Restart、Reweighted Audit、CTDR without Radius、Legacy CTDR、Full CTDR 的逐 panel/team 结果。

## 可复现入口

在本分支 `experiments/mechanism_20261122` 下运行：

```powershell
python m60_module_ablation.py E:\图结构演化\artifacts\private\mechanism_20261122\m60_ablation_manifest.json
```

manifest 只引用已封存 worker JSONL、输入 receipt 和 panel 配置。脚本不调用模型，不读取 answer key，不修改私有原始 trace。
