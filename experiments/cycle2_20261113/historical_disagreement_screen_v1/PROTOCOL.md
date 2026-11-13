# 唯一历史 CPU 否定筛查：literal r1 分歧 × receiver 入边变化

状态：计算前固定。仅旧 G2 train200/tune80 的派生匿名缓存；不读 exposed120 质量、当前 Hotpot、旧 R1 S、题目、gold、raw 生成，不使用 GPU，不计新的正式候选。至多一次完整实际计算；代码/数据异常保留失败回执，修复需明确记录且不改变本统计设计。

## 可识别的问题

现缓存只有 parsed answer 原字符串 SHA，没有每个 r1 的数值规范化答案。故唯一 Z 是 **seed2701 四个 r1 答案字面不全相同**，normalizer 为恒等映射；不是语义分歧。任一 r1 format_bad，或字面空串的 SHA，令 Z=unknown，保留支持数但不并入一致组。其余四个 seed2702–2705 只用于结果统计，不参与 Z。

两 anchor 随题变化，因此估计的是旧 G2 已冻结随机图背景规则下的关联，不是全体题共用固定 g0→g1。每个 physical edge 的结构变量 D：背景变化是否改变该边目标 receiver 的任何**其他**入边。D 只由 masks 构造，不使用响应。含 aggregate 的完整闭包对各边共同相交，不能作区分信号；D=0 也不意味着信用不会通过 aggregate 背景而变化。

对每题 q、边 e、结果 seed s：c(q,a,e,s)=Q(right)−Q(left)，t(q,e,s)=c(q,anchor1,e,s)−c(q,anchor0,e,s)。唯一结果量 θ(q,e) 为结果四 seed 的六个跨 seed 乘积 t_s*t_u 的均值。它允许为负，不截零。若各 seed 条件独立且 t 无偏，θ 估计该题/边潜在平均信用漂移的平方；不是单次漂移事件率、不是删留损失、不是 G2 原风险或免费实际观测。

## 唯一主对比与分母

每题分别在 D=1/D=0 的边上均值 θ；只对同时含两类边的题定义 d_q=mean(θ|D=1)−mean(θ|D=0)。主对比 I=mean(d_q|Z=1)−mean(d_q|Z=0)，预期方向为正。每题等权，保持题内所有边、anchor、seed 依赖；不能把边当独立样本。

train200 只描述；**tune80 是唯一主要否定对比**，它也是已揭示开发数据，不是新的独立确认。两个 partition 不混合。每个 partition 在两个 Z 组内各自按题重抽 2,000 次，PCG64 seed 分别97531/97532，报告 percentile95%区间。无任何训练拟合、参数选择、阈值搜索、边子集选择或替代 Z。

四个 Z×D cell、unknown、没有 D 内对比的题，全部报告支持数和描述均值。可解释区间的最低报告 guard：两组各至少20个可比较题，且实际 bootstrap 非退化。20 不是充分检验力保证。缺组/少组/退化时 CI 只作描述或置 null，不签正负科学证据；不重新定义 Z。

## 固定结果状态

- 任一数据/来源/图配对/r1一致性断言失败：BLOCKED_DATA_INTEGRITY，不继续统计。
- 主要对比没有两个足量组或 bootstrap 退化：INSUFFICIENT_SUPPORT_FOR_FIXED_LITERAL_PROBE。停止此 proxy 的本次计算，不推广为所有 disagreement 无效。
- 支持 guard 通过且区间上端≤0：EXPLORATORY_EVIDENCE_AGAINST_POSITIVE_INTERACTION，反对本预定正向 surrogate 命题。
- guard通过且下端>0：EXPLORATORY_SIGNAL_ONLY，不授新GPU或正式轮次；仍未证明旧方向错误、主动采样或净服务收益。
- 其他：INCONCLUSIVE，不得将非显著写为等价。

不要求训练对比显著；不能用 train 成绩替代 tune 主结果。即使 θ 交互没有支持，也只否定这一字面分歧/receiver变化/漂移平方目标组合，不否定其他合法信息源或整个研究主题。没有统计结果可用于改当前 Hotpot 合同。

## 数据和实现防护

D2 SHA固定7639dad2581ea52536b7c274d33fd019ad726a9bce9e8e4ec6792c68b77f4963。response缓存与已冻结RESULTS的cacheSHA一致；format缓存与旧SCAN_COMPLETE绑定。只取 train/tune row；D2 quality成员按行流式解码选中部分，不解释其余120题质量。缓存present必须恰好覆盖这280题；同题同seed全部图的r1 answer/prompt/bad必须一致；每个配对只切换对应一边且属于同题。所有原文件字节不改。

输入读取量预计：40MB压缩answer/prompt缓存、0.15MB格式缓存、0.10MB D2匿名数组；不扫描约0.9GB原始输出。CPU预算上限按一次≤120秒设计，普通失败不自动重试。实际输入hash、源hash、协议hash与完整状态在新receipt记录。初次默认sandbox仅文件metadata读取被拒，随后授权范围内只读escalation获准；该错误不是文件不存在或科学失败。
