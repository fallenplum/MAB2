# 指定设计类的方差与退化证书

固定银行 N 行，信用 y_i，响应层 h∈{0,1,2}，h=0 是探测后可认证的 y_i=0。不得在探测前免费知道 h。R 是全银行 receiver 成本，A_h 是第 h 层全部 aggregate+评分成本；a_h=N^-2 sum_{i∈h} y_i²。均匀独立探测概率 q，完成概率 p_h，最终纳入率 r_h=q p_h。

独立 HT 的设计方差为 sum_h a_h(1/r_h−1)。成本 qR+sum_h r_h A_h，外加已探测行触达的基图。若基图组 g 有 m_g 条候选行、成本 b_g≥0，则 B(q)=sum_g b_g[1−(1−q)^m_g]。这是逻辑基图成本模型；不重建物理执行复用。

R>0 且 B 单调时，固定 r_h 最便宜的 q=max_h r_h；令最大层为top，另一层low，r_top=q，r_low=tq，0<t≤1。定义 G(q)=B(q)+qR，则预算约束是 G(q)+q(A_top+t A_low)=b，方差 V=(a_top+a_low/t)/q−sum_h a_h。

在 q<1 段，令 D=G′(q)+A_top+t A_low>0，有 q′(t)=−q A_low/D，故

    dV/dt = [a_top A_low t² − a_low(G′(q)+A_top)] / (q t² D).

由于 B′(q)=sum_g b_g m_g(1−q)^(m_g−1)≥B′(1)=sum_{g:m_g=1}b_g，若两个top/low方向都满足

    a_top A_low ≤ a_low(R+B′(1)+A_top),

则全段 dV/dt≤0。q=1 饱和段 dV/dt=−a_low/t²≤0，交界连续，所以该指定类所有正预算的最优可取 t=1，即 p1=p2=1。此证书是充分非必要条件；不满足绝不推出必有收益。它是基础微分推导，无原创定理主张。

RESULTS.json记录了两方向的未除N² margin。train/tune 的评分等价比0和0.1均为正；比1的低风险层方向为负，须依赖数值优化结果并保留其小幅空间。基图成本是按探测付费，不能改成按最终完成付费。

直接SRS均值方差：S_y²(1/n−1/N)。期望基图费：sum_g b_g[1−C(N−m_g,n)/C(N,n)]。公式在整数n正确；代码使用gamma连续延拓，只是连续放松。

全银行探测后的分层SRS方差：sum_h (N_h/N)² S_h²(1/n_h−1/N_h)。在固定aggregate预算下，最优 n_h=min{N_h,sqrt[(N_h/N)² S_h²/(lambda c_h)]}；必须先支付全银行R与全部基图费。真实层方差来自历史结果，所以是oracle。

这些量均针对固定银行的抽样随机性。它们不处理题分布漂移、独立新题误差、图政策遗憾或未知proxy偏差，也不是全策略下界。新增使用 hard budget、无放回自适应选择或条件化概率后，必须重新推导设计与方差。
