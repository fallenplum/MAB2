# M60 审计周期合同

## 研究问题

在已经支付旧图 transcript 的前提下，图编辑后信用证书的节省来自历史观测复用、来源分布重加权、漂移半径证书，还是决策区域的实际 token 成本排序。

## 预注册对照

在四个冻结新图面板上，对所有题目×团队单元共享同一旧图 transcript、同一新图输入、同一 sealed 输入和同一 post-edit 成本口径。

1. Warm Restart，历史观测精确复用，单位成本 `Audit.next`。
2. Reweighted Audit，与 Warm Restart 相同，但作为来源重加权模块的显式命名对照。
3. CTDR without Radius，移除漂移半径，保留相同的历史复用和查询器。
4. CTDR legacy，保留当前 radius gate 和单位成本 `Audit.next`。
5. Warm Restart token，使用记录的真实 token 价格，但不改变 `Audit.next` 的决策区域目标。
6. Decision Ranking，按预期决策区域未定宽度除以真实 token 价格选择查询，不使用 radius。
7. Full CTDR，同时使用 radius gate 和真实 token 决策区域排序。

## 通过条件

只有 Full CTDR 在相同决策集合正确性下，相对 Warm Restart、Reweighted Audit、CTDR without Radius 和 Decision Ranking 产生可重复的独立成本或错误下降，才保留算法创新主张。否则把结果降级为机制分解和实现审计。

## 禁止事项

不修改旧门槛，不访问已封存答案进行调参，不把历史 transcript 成本与图编辑后增量混报，不将题目×团队单元称为独立题目，不使用置信区间或 bootstrap。
