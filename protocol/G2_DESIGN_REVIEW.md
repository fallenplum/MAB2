# G2 设计审查与待冻结配置

日期为 2026-11-04。本文依据指定只读参考与无标签校准题提出统计合同建议。根代理已确认200/80/120、两个 Bernoulli 锚图、全部12边、5种子、预算上界、8项alpha网格、参数全惩罚、特征顺序和10000次bootstrap。本次状态为 `PARAMETERS_APPROVED_PENDING_PLAN_AUDIT`，完整计划及实际encoder、设计矩阵和来源清单由根代理进一步审计并冻结。

## 1. 核心问题与允许的结论

令 `Y(x,g,s)` 为原冻结执行器在题目 x、12位原始图 g、节点随机流 s 下的最终正确性，取值为0或1。令 `F*(x,g)=E_s[Y(x,g,s)]`。对边 e 的预测目标为

\[
q_e(x,a)=F^*(x,a_{e\leftarrow1})-F^*(x,a_{e\leftarrow0}).
\]

同一题、同一种子内两个图共享节点随机流，直接从各图真实服务输出评分。预测器在 heldout 标签揭示前给出两个图的质量预测之差。

G2主门检验二阶模型的题外边信用预测是否优于无条件边均值。与使用相同题目表示的一阶模型的比较，另行回答其他边状态是否带来额外预测价值。若只通过前者，结论限定为相对于无条件边均值的预测收益。图上下文归因需要同时审查后一比较。

G1聚合发现允许提出这一实验，G1的200题和标签继续隔离在本预测器之外。当前设计仍为同一3B骨干的固定角色系统，G2提供离线信用预测证据。G3承担完整预算下的在线采集验证。

## 2. 已读参考与输入审计

| 参考 | SHA256 |
|---|---|
| EXECUTION_CONTRACT.md | `3c7154ea281fe6390d7faffffb48c1338bf95fc3cb2b3a6319fd1d189adfc47c` |
| run_pilot.py | `07b8ed84f647973ecb5ca8d4ade05f47637edc5d7de00833699d8b55bdfc9c69` |
| executor.py | `85acab26b3b713900e8099a5d77197af698525f57c2d9d07d1193c68b2c383b1` |
| motivation_v3.json | `5c96c3b20a109baab683376a0f3ac9ac8a384e0e15997554b708fa38a743f761` |
| calibration_questions.jsonl | `4517478a518d09c2c1afd5cf7b3e453e5947c62b128e9c98138512323328b6e1` |

校准文件共有400行、400个唯一ID。全部字段集合为 `id,question,split,task`，全部题目非空，task均为gsm8k、split均为calibration。当前文件未含标签字段。与G1、资格题、在线流和官方test的跨文件排他性，由根代理在数据清单审计中以ID和规范化题干重新验证。

参考执行器提供 source-major 的12条有向边、2轮闭包重执行、原图保留、协议指纹和 `derive_seed(question_id,execution_seed,node)`。当前 `run_pilot.py` 的四单元循环、配对观测键和小样本 `PILOT_ONLY` 标记服务于G1。G2需要独立的图计划和阶段状态，并继续调用原执行接口。

## 3. 题级划分与功效边界

采用训练200、调参80、heldout120。对400题用独立域的SHA键排序后切分，清单在第一次生成前保存并散列。每道题的全部图、种子、别名、嵌入和派生差分属于同一划分。文本相似或图相同均不能将同一题的观测拆到两个划分。

相同题级损失差方差下，120道heldout题的标准误约为80道题的 `sqrt(80/120)=0.816` 倍。这个计算解释增加heldout的价值。现有G1平方交互不能给出G2风险差的方差，也不能推出G2功效。711维ridge依赖正则化与题外检验，200训练题的数量仍构成限制。

80题调参集只用于预先列举的alpha选择。正式heldout使用120题一次性检验。若主门未通过，报告区间和负结果，并据此决定是否暂停G3。任何扩大样本或改特征的后续设计，都建立新的实验清单和新的外部评估依据。题目分配由无标签规则完成。

## 4. 图采样与全12边公平覆盖

每题预抽两个独立锚图，分别命名为 `bernoulli_025` 和 `bernoulli_075`。其12位分别服从p=.25和p=.75的Bernoulli设计。每个锚图执行基图及12个单边翻转邻图。

- split、锚图family和每个bit使用不同的SHA域。两个family使用独立的散列流，避免由同一组uniform数和不同阈值产生嵌套锚图。
- 所有锚图、翻转、执行顺序、别名和seed在生成前预抽。每个family和每条边在主指标中均等权。
- 用 `(question_id,mask,seed,protocol_hash)` 标识原始图执行。两个family产生的重合mask归入同一个执行记录和同一个质量标签，别名表保留其全部评估坐标。
- 计划允许锚图相同或相距较近，按预定分布记录这种实现。覆盖数依据实际唯一mask计算，推断单位仍为题。
- 逻辑边顺序严格沿用 `logical_edges()`。边e无论锚图原来为0还是1，信用方向统一为 `Y(mask1)-Y(mask0)`。

heldout每个family×edge均有120题、600个种子配对差。每题共24个评估坐标、120个种子差。完整支持表列出所有24行。主设计覆盖这两个密度分布。G1的五个固定背景可作为未来预声明的扩展评估。

## 5. 每个原始图标签一次的训练构造

原始评分表每个唯一 `(q,mask,seed)` 保存一行。每个 `(q,mask)` 聚合5个独立seed的质量均值

\[
\bar Y_{qg}=\frac15\sum_{s=1}^5 Y(q,g,s).
\]

模型F的训练表对每个唯一 `(q,mask)` 只保存一行。其特征使用该标签实际来自的图g。令该题唯一图数为 `M_q`，每行训练权重为 `1/M_q`，因此每个训练题总权重为1。

锚图质量可进入多条派生信用差分，别名也可引用同一质量记录。这些都是同一观测的线性使用，其相关性保留在题内。训练过程禁止把锚图基准质量复制成12行额外F样本，也禁止为一个质量标签生成12个独立在线更新。每图一次指F训练样本与质量计数，派生contrast表保留源图ID供审计。

缺失某个seed、图或评分会使该题不完整。完整科学门要求原计划覆盖通过。诊断报告保留缺失记录及已花费用。

## 6. MiniLM、PCA与三个必要预测器

编码器冻结为 `sentence-transformers/all-MiniLM-L6-v2`，具体revision、文件哈希、pooling、L2归一化、tokenizer和最大长度由根代理的资源审计固定。官方模型卡给出384维输出、Apache-2.0标记及默认256 word pieces截断。应报告各划分的截断数，保存实际送入编码器的文本规则。[官方模型卡](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

输入为执行器实际使用的题干与显式context。根代理已确认使用AutoModel、attention-mask mean pooling、L2归一化和CPU，`max_length=256`。对训练200题的嵌入各使用一次，以float64中心化full SVD拟合8维PCA，`whiten=False`。将这组均值和方向应用于调参及heldout。固定符号规则和保存PCA字节，避免不同SVD符号造成系数与重放差异。PCA的中心化和whitening含义见官方文档。[PCA文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

建议锁定 `phi=[PC1,...,PC8,1]`，维数为9。本选型使用原始PC坐标，所有模型共享同一个phi。图特征采用0/1编码

\[
h_2(g)=[1,g_0,\ldots,g_{11},g_ig_j\ (0\leq i<j<12)]\in\mathbb R^{79},
\quad \psi_2=\phi\otimes h_2\in\mathbb R^{711}.
\]

pair列按 `(i,j)` 字典序，`np.kron(phi,h)` 为phi-major。一阶 `h_1=[1,g_0,...,g_11]`、`psi_1=phi⊗h_1` 共117维。

| 方法 | 拟合与预测 | 解释 |
|---|---|---|
| B0 无条件边均值 | 对每边，在训练题×两family×5seed上求有向contrast均值，heldout对所有题和family输出同一个b_e | 无题目表示、无其他边状态 |
| M1 一阶图质量模型 | 唯一图标签拟合117维F，预测 `F(mask1)-F(mask0)` | q可随题目变化，q对其他边状态保持常数 |
| M2 二阶图质量模型 | 同一质量表拟合711维F，预测同样的图差 | q可随题目及其他边状态变化 |

B0精确定义为

\[
\hat b_e=\frac{1}{200\cdot2\cdot5}
\sum_{q\in train}\sum_{a\in\{.25,.75\}}\sum_s
\{Y(q,a_{e\leftarrow1},s)-Y(q,a_{e\leftarrow0},s)\}.
\]

每条contrast通过别名引用同一唯一质量表。B0不使用调参质量重估均值。M1和M2享有相同标签来源、phi、训练题、图权重和alpha搜索次数。另行报告 `qhat=0` 的零预测风险，作为稀疏场景的诊断参照。

## 7. Ridge目标与alpha选择

为与实现代理保持一致，建议显式使用所有参数均受惩罚的ridge，包括phi和h中的常数乘积。目标为

\[
\hat\theta_{k,\alpha}=\arg\min_\theta
\sum_{q\in train}\frac1{M_q}\sum_{g\in M_q}
(\bar Y_{qg}-\psi_k(q,g)^\top\theta)^2
+\alpha\|\theta\|_2^2.
\]

每题总权重1，训练权重之和为200。当前建议不加额外特征scaler，使用float64求解。该惩罚定义应写进模型元数据。若实现改用题均值总损失，即再除以200，则alpha必须同步除以200。软件Ridge的标准目标为平方和加alpha乘L2，需核对具体sample weights和截距设置。[Ridge文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

alpha网格建议为 `[0.0001,0.001,0.01,0.1,1,10,100,1000]`。M1和M2分别训练8个候选，分别在调参80题的24个信用坐标上计算题均权paired credit MSE。每题先对两family和12边平均，再对80题平均。损失差在 `1e-12` 内视作平手，选择较大的alpha。

选定alpha后保留训练200题上相应系数和原PCA，进入heldout评估。调参80题的标签只承担选择作用。主结果使用原始线性信用预测，保存超出理论[-1,1]范围的预测比例作诊断。

生成前，根代理应对实际训练phi、实际唯一图表和权重检查711维X的奇异值、有效秩与条件数，建议rank的相对容差 `1e-10`。同时检查117维X与phi的9维秩。随机图需跨题变化，才能避免所有题共用少于79张图时图特征子空间被固定低秩限制。满经验秩只说明当前设计矩阵的代数性质，泛化仍由heldout检验。

## 8. 独立seed误差与主统计量

对heldout题q、family a、边e，第s个种子得到

\[
d_s=Y(q,a_{e\leftarrow1},s)-Y(q,a_{e\leftarrow0},s),\quad
\bar d=\frac15\sum_s d_s.
\]

预测器仅使用训练与调参标签。heldout各seed因此独立于拟合及alpha选择，5个seed均可用于评价。节点seed派生保留题目ID、执行seed和节点名的域分离。

主比较在同一个 `bar d` 上计算

\[
\Delta_{B0-M2,q}=
\frac1{24}\sum_{a,e}
[(\hat b_e-\bar d)^2-(\hat q_{2,qae}-\bar d)^2].
\]

主点估计为120个题级差的均值，正值表示M2风险较小。由于两方法使用同一观测差，`bar d`平方项精确抵消，因此对固定预测的风险差没有测量噪声的加性偏置。有限5种子仍增加方差。

绝对平方风险另用跨seed无偏估计

\[
R_m^{U}(q,a,e)=\frac1{10}\sum_{s<t}(\hat q_m-d_s)(\hat q_m-d_t).
\]

其期望为 `E[(qhat_m-q_e)^2]`。每组与题级估计允许为负。`R_B0^U-R_M2^U`应与上面的同目标MSE差在 `1e-12` 容差内一致，这是有意义的实现反例检查。先计算这些派生量，再按题聚合。M1与M2的差使用同样公式。

对120个完整题级主差，采用10000次整题paired bootstrap，种子建议为39173。每次对排序后的120题有放回抽取120题，方法、family、边和seed共享同一抽样索引。使用NumPy `default_rng` 和linear分位数，95%区间取 `.025,.975`。所有负题差和负风险估计保留。

## 9. 主门与图上下文归因

G2主门按根代理批准的单一主要比较冻结

```
G2_PASS = 完整执行/划分/预测冻结/评分重放门通过
          且 mean(delta_B0_minus_M2) 的95%题级bootstrap区间下界 > 0
```

下界等于0也不通过。缺失计划支持为 `INCOMPLETE`，完整运行但主区间不满足门槛为 `INSUFFICIENT_PREDICTION_GAIN`。完整负结果按原标准呈现。

同时完整报告 `M1−M2` 的点估计、同样的题级区间和负贡献支持。它是预先列明的归因诊断，不改变主门状态。图上下文收益表述还需要该比较支持。若未来将其升级为确认性第二门，应在heldout揭示前明确固定顺序检验和对应声明范围。

额外指标包括两family×12边的支持、风险差和正负题差数，F质量误差、零预测参照、所有5个seed的观测contrast均为零的坐标上的预测平方值，以及出现非零contrast坐标上的误差与支持数。按观测contrast分组属于描述性分析，主权重固定为全部24坐标。观测零contrast的5种子支持不等同于总体真实信用为零。

## 10. 可持久审计的冻结时序

1. 保存400题排他清单、200/80/120划分、全部锚图、唯一mask和alias、5个seed、执行顺序、模型/协议/来源及预算。根代理对MiniLM与X完成无标签检查后封存G2计划。
2. 生成400题全部计划图输出，形成全局 `ALL_OUTPUTS_FROZEN`，保存raw和执行引用哈希以及可观测时间。各阶段成本完整记录。
3. 独立评分流程只将train200和tune80的质量授权给拟合流程。heldout的参考答案与质量继续由评估流程保管。训练流程的数据加载器按清单拒绝heldout行。
4. 复用已冻结的PCA并拟合模型，完成alpha选择。保存候选损失、选中alpha、PCA/模型字节、source身份及全部heldout图差预测，生成 `PREDICTIONS_FROZEN`。
5. 持久追加 `HELDOUT_LABEL_ACCESS` 记录，保留发生时间与此前预测冻结哈希，然后由独立评分/评估器打开heldout参考答案，完成评分和统计。该记录采用追加或独立文件形式保留历史事件。
6. 独立验收重新计算每个唯一图质量、24个contrast、B0、原始预测差、跨seed风险及题级bootstrap。最终摘要公开聚合数字，题号、原始输出、标签和完整逐题预测保留在私有产物。

若生成与评分采用分批执行，也应保持heldout标签访问晚于完整预测冻结的顺序。推荐全400题先生成的方案，减少阶段间意外反馈路径。

## 11. 预算与执行顺序

| 划分 | 题数 | 唯一图质量样本上界 | 原始图×seed输出上界 | 新物理调用上界 |
|---|---:|---:|---:|---:|
| train | 200 | 5200 | 26000 | 62000 |
| tune | 80 | 2080 | 10400 | 24800 |
| heldout | 120 | 3120 | 15600 | 37200 |
| 总计 | 400 | 10400 | 52000 | 124000 |

物理调用上界依赖明确的base选择。每题每seed首次锚图需9次调用，其12个flip分别以保存的该锚图结果为base，各需一个第二轮worker和一个汇总调用，共24次。第二锚图复用第一轮，至多5次调用，再加其12个flip的24次。合计至多62次，每题5seed至多310次。

所有flip都从所属锚图分支，成本与source_execution_id据此记账。去重aliases会降低上述数值。工程资格、失败、重试、迁移孤立调用和评分成本单列，并进入完整开发投入。52000是图输出上限，124000是该2轮路径的生成调用上界，墙钟仍需真实资源probe。节点生成token上界为22528000，按每次输入上限8192计的保守输入token上界为1015808000，实际token全部以物理账本为准。

## 12. 可直接转成配置的字段

以下数值已与根代理和predictor实现代理统一。encoder revision和实际清单字节待完整计划审计。

```json
{
  "schema": "graph_credit_g2_design_candidate_v1",
  "purpose": "heldout conditional edge-credit prediction",
  "data_partition": "original_calibration_400_only",
  "split": {
    "sizes": {"train": 200, "tune": 80, "heldout": 120},
    "method": "sort_sha256_domain_plus_question_id",
    "domain": "graph_credit/G2/split/v1"
  },
  "graph_design": {
    "anchors": [
      {"family": "bernoulli_025", "p": 0.25},
      {"family": "bernoulli_075", "p": 0.75}
    ],
    "bit_domain": "graph_credit/G2/anchor/v1",
    "domain_fields": ["question_id", "family", "bit_index"],
    "all_single_flip_edges": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "deduplicate": "question_id_mask_seed_protocol",
    "flip_execution_base": "saved_own_anchor_execution",
    "seeds": [2701, 2702, 2703, 2704, 2705]
  },
  "feature_contract": {
    "encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "ROOT_TO_PIN_BEFORE_EXECUTION",
    "normalize_embeddings": true,
    "encoder_loader": "AutoModel",
    "pooling": "attention_mask_mean",
    "encoder_device": "cpu",
    "max_length": 256,
    "pca_components": 8,
    "pca_fit_split": "train",
    "pca_whiten": false,
    "phi_order": "PC1_PC2_PC3_PC4_PC5_PC6_PC7_PC8_CONSTANT1",
    "graph_coding": "binary_0_1",
    "pair_order": "lexicographic_i_lt_j",
    "kronecker_order": "phi_major",
    "additional_feature_scaler": "none",
    "dimensions": {"first_order": 117, "second_order": 711}
  },
  "training": {
    "unit": "unique_question_graph_seedmean5",
    "row_weight": "1_over_unique_graphs_in_question",
    "total_weight_per_question": 1,
    "ridge_loss": "sum_question_mean_unique_graph_squared_error_plus_alpha_l2",
    "penalize_all_coefficients_including_constants": true,
    "alpha_grid": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "alpha_selection": "separate_per_model_min_tune_question_macro_paired_credit_mse",
    "tie_tolerance": 1e-12,
    "tie_break": "largest_alpha",
    "refit_after_tune": false,
    "prediction_clipping": "none"
  },
  "baseline": {
    "name": "unconditional_edge_empirical_mean",
    "fit_split": "train",
    "weights": "equal_question_equal_anchor_equal_seed_per_edge"
  },
  "evaluation": {
    "primary": "B0_minus_M2_paired_credit_mse_risk_difference",
    "weights": "equal_question_equal_anchor_equal_edge",
    "absolute_risk": "cross_seed_unbiased_pair_product",
    "diagnostic": "M1_minus_M2_same_risk_difference",
    "supplementary": "zero_prediction_and_observed_zero_contrast_errors",
    "resampling_unit": "whole_question",
    "bootstrap_samples": 10000,
    "bootstrap_seed": 39173,
    "confidence": 0.95,
    "quantile_method": "linear",
    "minimum_gain": 0,
    "decision": "full_integrity_and_primary_lower95_gt_0",
    "negative_values": "retain"
  },
  "budget_caps": {
    "raw_outputs": 52000,
    "fresh_generation_calls": 124000,
    "engineering_failures_orphans": "separate_actual_ledger"
  },
  "g1_predictor_data_allowed": false,
  "sealed_official_test_used": false,
  "status": "PARAMETERS_APPROVED_PENDING_PLAN_AUDIT"
}
```

## 13. 优势、风险与置信度

优势是题目与随机流独立隔离、全12边均衡覆盖、唯一图质量训练、相同标签来源的117/711维对照，以及同一噪声目标上的题级配对风险差。两密度锚图的预算上界可在执行前计算。

风险集中于200训练题对711维模型的正则化需求、120题heldout面对稀疏信用时的方差、仅两个随机上下文分布的适用范围，以及MiniLM截断和二阶近似的模型偏差。物理调用上界依赖2轮与锚图分支执行，工程实现需要保存这一分支关系。

置信度对代数恒等式、维数和调用上界为高，依据是当前执行器及显式推导。对G2通过的可能性无法给出数值预测，因为尚无题级预测风险差的观测方差。生成前需要根代理完成encoder版本、输入截断、实际设计矩阵和最终配置的审计。
