# FActScore E0 适配器设计

## 官方输入与知识源

官方 `FactScorer` 接收 `topics`、`generations` 和可选的 `atomic_facts`。F 投影使用已下载的 `InstructGPT.jsonl` 中的 `topic`、`output` 和 `annotations[*].model-atomic-facts[*].text`。D 固定阈值服务使用 `human-atomic-facts[*].text` 作为事前冻结的事实列表，只丢弃 `label` 字段，人工标签仅进入后置 scorer。两种合同分开记录，在线 broker 不读取标签。

官方 `register_knowledge_source` 需要 SQLite 数据库中的 `documents(title PRIMARY KEY, text)` 表。默认知识源是 `enwiki-20230401`，当前目标路径为 `E:\图结构演化\data\raw\factscore\enwiki-20230401.db`。

## 最小 smoke

1. 固定 `InstructGPT.jsonl` 中已锁定的前 4 个匿名 group。
2. 只使用官方 `model-atomic-facts` 作为冻结事实清单，不调用在线原子事实生成器。
3. 使用 `retrieval_type="bm25"`，避免首轮引入 GTR 编码器和额外模型下载。
4. 对每个事实记录 topic、fact digest、检索 key、top-k 证据 ID、输入/输出 token、耗时、缓存命中和异常状态。
5. 后置 scorer 使用人工原子事实标签计算 F 的支持率，并把固定单位权重、`tau=0.80` 的 D 证书独立计算。

## 与官方 CLI 的边界

官方 `FactScorer.get_score` 只提供整体分数、响应比例和逐事实决策，不能单独满足 E0 的 F/D 双合同、逐节点成本账本、缓存 closure 和 `PASS/FAIL/UNRESOLVED` 停止记录。外层 adapter 必须拥有 broker、ledger 和后置 scorer，不能把官方整体分数直接当作 E0-D 证书。

当前知识库下载完成前不运行真实 smoke，也不把旧 QASC/SciQ/HotpotQA 轨迹混入 E0。


