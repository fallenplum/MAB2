# 纯数组统计接口

`applicability_statistics.analyze(em, f1, edge_indices, audit)`不读取文件、题目、gold或模型。由独立adapter在完整采集/费用/身份审计后评分并组装输入；统计模块不能替代这项审计。

`em`和`f1`各为严格四键dict：

- `graphs`: ndarray `(384,2,4)`，轴顺序为题、seed1/seed2、`[A,B,A_MINUS_E,B_MINUS_E]`。
- `no_edge`、`independent12`、`single`: 各`(384,)`，仅seed1。
- EM必须为0/1，F1为有限的[0,1]数；禁止NaN、部分题或静默丢弃失败。

`edge_indices`为384个整数，每个1/5/7/9恰96题；顺序对应冻结selection manifest。不要为了统计排序改变图/标签对应关系。

`audit`严格六键：

```python
audit = {
    'integrity_passed': True,
    'unique_questions': 384,
    'complete_conditions': 4224,
    'runtime_failure_count': 0,
    'input_overflow_conditions': 0,
    'cost_frontier_review': 'NOT_ASSESSED',
}
```

`cost_frontier_review`仅允许NOT_ASSESSED/NOT_DOMINATED/DOMINATED。生产调用必须绑定独立审计回执，不能凭填True创建科学权限。尤其合法overflow已在指定condition的EM/F1正确填0、所有终态来源与费用对应等事实，纯数组模块无法从计数单独重建。

返回`primary_EM`含四个主统计及区间、支持数、方差；`secondary_descriptive_only`含F1和其他描述；`decision`分类不授予GPU、不声称方法成功；`bootstrap`记录PCG64、分位点与完整抽样索引SHA256；`per_edge_primary_means_descriptive`仅供报告，不能挑边救援。

结果尚未绑定实际研究数据；这是待冻结实现。跑合成测试：

```text
python -B -m unittest -v test_statistics.py
```

测试只用人工384行数组，覆盖负交叉积、全零与层内常数的退化、低于5pp的方向支持、F1不可救EM、漏覆盖/运行失败拒绝、边平衡与二元分数检查、容量阻断及机制反向标记。没有任何benchmark访问。
