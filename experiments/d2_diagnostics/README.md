# D2 纯数组数值内核

本目录不读取数据、模型或历史报告，不启动正式实验。`diagnostics.py` 只接受已验证的数组。依赖为 Python 和 NumPy。

## 接口

```python
problem = prepare_problem(X, y, question_ids, variant, d=13_or_79,
                          graph_ids=optional_canonical_graph_ids,
                          difference_X=DX, difference_y=Dy,
                          difference_question_ids=dq)
fits = fit_path(problem)  # 同一次主 SVD 共用全部 8 个 alpha
selected = select_alpha(problem, tune_credit_X, tune_credit_y,
                        tune_question_ids, fits=fits)
beta = selected.best_fit.beta
```

`variant` 为 R0、R1、R1b、R2、R3。非 R3 可省略 difference 参数。d 是图特征维度 13/79；X 是 phi-major 的 9*d 列，最后一个 phi 坐标恒为 1，图特征第一坐标恒为 1。R1 免罚列 8*d，R1b 免罚列 k*d（k=0,...,8）。

目标固定为 `sum(w*(y-X@beta)**2)+alpha*sum(beta_pen**2)`，每个训练题的原 unique 图均权且总权重 1。不会再除训练题数。R2 同时按题中心化 X 和 y；R3 使用调用者提供的每题 24 条预定差分，各权重 1/24。

`fit_path` 返回按输入 alpha 顺序的 `list[FitResult]`，每项包含 ndarray `beta`、float `alpha` 和标量/字符串 `diagnostics`。默认网格为 1e-4、1e-3、1e-2、1e-1、1、10、100、1000。一次主 SVD 之外，R1/R1b 额外对最多 9 列 nuisance 做 SVD，使用加权 FWL 并允许秩缺失。

`select_alpha` 返回 `SelectionResult(best_fit,best_alpha,all_trials)`。`all_trials` 是可直接 JSON 序列化的字典列表，每项为 `alpha` 和 `tune_credit_mse`。评分先在每个 tune 题内均值，再跨题均值；距最小值不超过 1e-12 视为并列，取最大 alpha；返回原训练 fit，不执行 train+tune refit。若自行提供 fits，调用者负责保证它们确实来自同一个 problem，函数只验证 alpha 网格一致。

## 归因与几何辅助接口

- `constant_residual_decomposition(problem, alpha, credit_X=None)`：R0 在固定同一 alpha 下，将 y 分为训练加权均值常数项和残差项，共享分解求解，返回 beta/信用三部分及线性恒等误差。它不是重新调 alpha 后的因果比较。
- `full_pairwise_problem(X,y,q)`：构造全部 i<j 差，权重 1/n_q²，与 R2 的中心化目标完全等价；该辅助目标不属于预定 R3。
- `pairwise_loss_identity(residual,q)`：逐题独立核验中心化方差和全对差目标恒等。
- `contrast_graph_diagnostics(D,weights=None,alpha=None)`：D 是某题的完整唯一图顶点×预定差的有向 incidence 矩阵（形状为差数×顶点数），孤立顶点必须保留列。输出顶点/边/零差数、秩、连通分量、谱 trace、完整谱及可选图算子有效自由度。

所有 fit 的 effective_df 都标记为 prepared_observation_operator。R2 的 df 不含已经 profile 掉的每题自由基准，另外给出 profiled_question_offsets_count。R0/R1/R1b 的原图质量空间、R2 的中心化空间和 R3 的差分空间不同，不可把这些 df 直接解释为统一的完整 F 模型复杂度。图 D 算子的 df 也不同于特征模型的 df。

## 适配器必须验证的边界

1. 历史数据、G2/D1 文件保持原字节；本模块不会自行打开它们。
2. X/y 必须对应 unique canonical 图；可用 graph_ids 去重别名。重复别名若 X/y 冲突则报错，不将不同观测静默平均。
3. R3 适配器必须验证预定 24 个 pair 的身份、方向、D 的顶点映射，以及 DX=D@X、Dy=D@y。本模块只能验证数组形状、题集合和 24 条数目。
4. tune_credit_y 是预定 5-seed mean，不能混入封存测试或新增自适应标签。
5. R0 与历史所选 alpha 的完整预测重放、固定旧 alpha 与重新 tune 的分开展示，由适配器落实。人工测试不代表真实历史结果已重放。

运行纯人工测试：`python -m unittest -v test_diagnostics`。19 项内核测试结果与源码哈希见 TEST_REPORT.json / TEST_OUTPUT.txt。

适配器的人工函数测试：`python -m unittest -v test_run_d2`。9 项测试只导入并调用决策/统计函数，不调用 execute/read_inputs；覆盖 signed 添加与删除、stay、严格零阈值、低边编号并列、候选排除 R0、gain→MSE→名称提名规则、LOSO 防止 held-out seed 参与选边、五 fold 依赖记录，以及 cross-seed 噪声与风险恒等式。结果见 TEST_ADAPTER_REPORT.json。内核和适配器合计 28 项通过；该结果不替代真实 R0 重放或数据来源审计。
